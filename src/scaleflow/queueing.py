"""Queue-aware collaborative inference simulation.

The simulator is deliberately separate from the online scheduler.  It replays
historical per-model observations (or an explicitly configured synthetic
profile) under a deterministic arrival process and models node queues and
network costs with a small discrete-event simulation.  Results are simulation
or historical replay artifacts; they are not measurements of an online vLLM
deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from math import isfinite, log
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any, Mapping, Sequence

from scaleflow.baseline import percentile


class QueueingConfigError(ValueError):
    """Raised when a queueing experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    quality: float
    service_time_ms: float
    source: str
    observation_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class NodeProfile:
    node_id: str
    capacity: int
    model_ids: tuple[str, ...]
    service_time_multiplier: Mapping[str, float]
    transmission_ms: float
    propagation_ms: float


@dataclass(frozen=True, slots=True)
class Candidate:
    model_id: str
    node_id: str
    quality: float
    estimated_compute_ms: float
    transmission_ms: float
    propagation_ms: float
    service_time_multiplier: float = 1.0

    @property
    def round_trip_network_ms(self) -> float:
        return 2.0 * (self.transmission_ms + self.propagation_ms)

    @property
    def one_way_network_ms(self) -> float:
        return self.transmission_ms + self.propagation_ms


@dataclass(frozen=True, slots=True)
class Arrival:
    request_id: str
    sequence: int
    arrival_time_ms: float
    observations: Mapping[str, Mapping[str, Any]]


@dataclass(slots=True)
class _ServerState:
    available_at_ms: float = 0.0
    pending_jobs: int = 0


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueueingConfigError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result) or result < minimum:
        raise QueueingConfigError(f"{name} must be finite and >= {minimum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QueueingConfigError(f"{name} must be a positive integer")
    return value


def _model_id(item: Mapping[str, Any], index: int) -> str:
    value = item.get("model_id", item.get("id"))
    if not isinstance(value, str) or not value:
        raise QueueingConfigError(f"models[{index}].model_id must be non-empty")
    return value


def _build_candidates(
    profiles: Mapping[str, ModelProfile],
    nodes: Sequence[NodeProfile],
) -> list[Candidate]:
    return [
        Candidate(
            model_id=model_id,
            node_id=node.node_id,
            quality=profiles[model_id].quality,
            estimated_compute_ms=profiles[model_id].service_time_ms
            * node.service_time_multiplier[model_id],
            transmission_ms=node.transmission_ms,
            propagation_ms=node.propagation_ms,
            service_time_multiplier=node.service_time_multiplier[model_id],
        )
        for node in nodes
        for model_id in node.model_ids
    ]


def _network_scenario_nodes(
    nodes: Sequence[NodeProfile],
    scenario: Mapping[str, Any],
) -> list[NodeProfile]:
    overrides = scenario.get("overrides", scenario.get("nodes", {}))
    if not isinstance(overrides, Mapping):
        raise QueueingConfigError("network scenario overrides must be an object")
    result: list[NodeProfile] = []
    for node in nodes:
        override = overrides.get(node.node_id, {})
        if not isinstance(override, Mapping):
            raise QueueingConfigError(
                f"network override for node {node.node_id} must be an object"
            )
        result.append(
            NodeProfile(
                node_id=node.node_id,
                capacity=node.capacity,
                model_ids=node.model_ids,
                service_time_multiplier=node.service_time_multiplier,
                transmission_ms=_number(
                    override.get("transmission_ms", node.transmission_ms),
                    f"{node.node_id}.transmission_ms",
                ),
                propagation_ms=_number(
                    override.get("propagation_ms", node.propagation_ms),
                    f"{node.node_id}.propagation_ms",
                ),
            )
        )
    return result


def _parse_profiles(
    config: Mapping[str, Any],
    model_records: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> tuple[dict[str, ModelProfile], list[NodeProfile], list[Candidate]]:
    model_items = config.get("models")
    if not isinstance(model_items, list) or not model_items:
        raise QueueingConfigError("models must be a non-empty list")
    profiles: dict[str, ModelProfile] = {}
    for index, item in enumerate(model_items):
        if not isinstance(item, Mapping):
            raise QueueingConfigError(f"models[{index}] must be an object")
        model_id = _model_id(item, index)
        if model_id in profiles:
            raise QueueingConfigError(f"duplicate model_id: {model_id}")
        records = list((model_records or {}).get(model_id, ()))
        historical_latencies = [
            _number(record["latency_ms"], f"{model_id}.latency_ms")
            for record in records
            if isinstance(record, Mapping) and record.get("latency_ms") is not None
        ]
        configured_service = item.get("service_time_ms")
        if configured_service is None and not historical_latencies:
            raise QueueingConfigError(
                f"{model_id} needs service_time_ms when no historical records are supplied"
            )
        service_time = (
            fmean(historical_latencies)
            if historical_latencies
            else _number(configured_service, f"{model_id}.service_time_ms", minimum=0.000001)
        )
        observation_scale = 1.0
        concurrency_reference = item.get("concurrency_reference_ms")
        if concurrency_reference is not None:
            reference_ms = _number(
                concurrency_reference,
                f"{model_id}.concurrency_reference_ms",
                minimum=0.000001,
            )
            if historical_latencies:
                observation_scale = reference_ms / service_time
            service_time = reference_ms
        configured_quality = item.get("quality")
        if configured_quality is None and records:
            configured_quality = fmean(
                float(bool(record.get("correct"))) for record in records
            )
        if configured_quality is None:
            raise QueueingConfigError(
                f"{model_id} needs quality when no historical correctness is supplied"
            )
        quality = _number(configured_quality, f"{model_id}.quality")
        if quality > 1.0:
            raise QueueingConfigError(f"{model_id}.quality must be <= 1")
        if records and concurrency_reference is not None:
            source = "historical_phase7_jsonl_calibrated_by_phase9_concurrency"
        elif records:
            source = "historical_phase7_jsonl"
        else:
            source = "synthetic_config"
        profiles[model_id] = ModelProfile(
            model_id,
            quality,
            service_time,
            source,
            observation_scale,
        )

    node_items = config.get("nodes")
    if not isinstance(node_items, list) or not node_items:
        raise QueueingConfigError("nodes must be a non-empty list")
    nodes: list[NodeProfile] = []
    for index, item in enumerate(node_items):
        if not isinstance(item, Mapping):
            raise QueueingConfigError(f"nodes[{index}] must be an object")
        node_id = item.get("node_id", item.get("id"))
        if not isinstance(node_id, str) or not node_id:
            raise QueueingConfigError(f"nodes[{index}].node_id must be non-empty")
        capacity = _positive_int(item.get("capacity", 1), f"nodes[{index}].capacity")
        configured_models = item.get("model_ids", item.get("models"))
        if configured_models is None:
            configured_models = list(profiles)
        if not isinstance(configured_models, list) or not configured_models:
            raise QueueingConfigError(f"nodes[{index}].model_ids must be non-empty")
        model_ids = tuple(str(value) for value in configured_models)
        unknown = set(model_ids) - set(profiles)
        if unknown:
            raise QueueingConfigError(
                f"node {node_id} references unknown models: {sorted(unknown)}"
            )
        multipliers = item.get("service_time_multiplier", {})
        if not isinstance(multipliers, Mapping):
            raise QueueingConfigError(f"nodes[{index}].service_time_multiplier must be an object")
        parsed_multipliers = {
            model_id: _number(
                multipliers.get(model_id, 1.0),
                f"{node_id}.{model_id}.service_time_multiplier",
                minimum=0.000001,
            )
            for model_id in model_ids
        }
        network = item.get("network", item)
        if not isinstance(network, Mapping):
            raise QueueingConfigError(f"nodes[{index}].network must be an object")
        nodes.append(
            NodeProfile(
                node_id=node_id,
                capacity=capacity,
                model_ids=model_ids,
                service_time_multiplier=parsed_multipliers,
                transmission_ms=_number(
                    network.get("transmission_ms", 0.0),
                    f"{node_id}.transmission_ms",
                ),
                propagation_ms=_number(
                    network.get("propagation_ms", 0.0),
                    f"{node_id}.propagation_ms",
                ),
            )
        )
    return profiles, nodes, _build_candidates(profiles, nodes)


def generate_arrivals(
    queue_config: Mapping[str, Any],
    rate_per_second: float,
    *,
    seed: int,
    observations: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[Arrival]:
    """Generate a reproducible Poisson or periodic arrival stream."""

    rate = _number(rate_per_second, "rate_per_second", minimum=0.000001)
    request_count = _positive_int(queue_config.get("request_count", 100), "request_count")
    process = str(queue_config.get("process", "poisson"))
    rng = Random(seed)
    sampling_seed = int(queue_config.get("request_sampling_seed", seed))
    sampling_rng = Random(sampling_seed)
    model_ids = list(observations or {})
    if not model_ids:
        model_ids = [str(item.get("model_id", item.get("id"))) for item in queue_config.get("models", [])]
    if not model_ids:
        raise QueueingConfigError("cannot generate requests without model observations")
    record_ids: list[str] = []
    record_indices_by_model: dict[str, dict[str, int]] = {}
    if observations:
        first_records = list(observations[model_ids[0]])
        record_ids = [str(record.get("sample_id", index)) for index, record in enumerate(first_records)]
        if len(set(record_ids)) != len(record_ids):
            raise QueueingConfigError("historical records contain duplicate sample_id values")
        record_indices_by_model[model_ids[0]] = {
            sample_id: index for index, sample_id in enumerate(record_ids)
        }
        for model_id in model_ids[1:]:
            ids = [str(record.get("sample_id", index)) for index, record in enumerate(observations[model_id])]
            if len(set(ids)) != len(ids):
                raise QueueingConfigError("historical records contain duplicate sample_id values")
            if set(ids) != set(record_ids) or len(ids) != len(record_ids):
                raise QueueingConfigError("historical model record sample_id sets do not match")
            record_indices_by_model[model_id] = {
                sample_id: index for index, sample_id in enumerate(ids)
            }
    if not record_ids:
        record_ids = [f"synthetic-{index:06d}" for index in range(request_count)]
    sampling_order = list(range(len(record_ids)))
    sampling_rng.shuffle(sampling_order)
    start_time = _number(queue_config.get("start_time_ms", 0.0), "start_time_ms")
    arrivals: list[Arrival] = []
    now = start_time
    for sequence in range(request_count):
        if sequence:
            if process == "poisson":
                now += -log(max(rng.random(), 1e-15)) * 1000.0 / rate
            elif process in {"periodic", "constant"}:
                now += 1000.0 / rate
            else:
                raise QueueingConfigError(
                    "arrival process must be poisson, periodic, or constant"
                )
        record_index = sampling_order[sequence % len(sampling_order)]
        request_id = record_ids[record_index]
        request_observations = {
            model_id: dict(
                list(observations[model_id])[record_indices_by_model[model_id][request_id]]
            )
            for model_id in model_ids
        } if observations else {}
        arrivals.append(
            Arrival(
                request_id=request_id,
                sequence=sequence,
                arrival_time_ms=now,
                observations=request_observations,
            )
        )
    return arrivals


def _candidate_key(candidate: Candidate) -> tuple[str, str]:
    return candidate.model_id, candidate.node_id


def _quality_threshold(queue_config: Mapping[str, Any]) -> float:
    slo = queue_config.get("slo", {})
    if not isinstance(slo, Mapping):
        raise QueueingConfigError("slo must be an object")
    value = slo.get("min_quality", queue_config.get("min_quality", 0.0))
    quality = _number(value, "slo.min_quality")
    if quality > 1.0:
        raise QueueingConfigError("slo.min_quality must be <= 1")
    return quality


def _choose_candidate(
    strategy: Mapping[str, Any],
    candidates: Sequence[Candidate],
    queue_config: Mapping[str, Any],
    node_states: Mapping[str, list[_ServerState]],
    arrival_time_ms: float,
) -> Candidate:
    strategy_type = str(strategy.get("type", strategy.get("policy", "queue_aware")))
    if strategy_type in {"fixed", "fixed_model", "fixed_node"}:
        model_id = strategy.get("model_id", strategy.get("fixed_model_id"))
        node_id = strategy.get("node_id", strategy.get("fixed_node_id"))
        matches = [
            candidate
            for candidate in candidates
            if candidate.model_id == model_id
            and (node_id is None or candidate.node_id == node_id)
        ]
        if not matches:
            raise QueueingConfigError(
                f"fixed strategy cannot find candidate {model_id!r}/{node_id!r}"
            )
        return matches[0]

    threshold = _quality_threshold(queue_config)
    eligible = [candidate for candidate in candidates if candidate.quality >= threshold]
    if not eligible:
        if bool(queue_config.get("allow_quality_fallback", False)):
            eligible = list(candidates)
        else:
            raise QueueingConfigError(
                "no model/node candidate satisfies slo.min_quality; "
                "set allow_quality_fallback to document a quality relaxation"
            )
    if strategy_type in {"quality_only", "quality", "quality_first"}:
        # Quality-only intentionally does not inspect queue state, network, or
        # expected compute time.  It picks the least capable model that meets
        # the quality target, with configuration order as the final tie-break.
        return min(
            enumerate(eligible),
            key=lambda item: (item[1].quality, item[0]),
        )[1]
    if strategy_type not in {"queue_aware", "queue", "queue_and_quality"}:
        raise QueueingConfigError(f"unsupported queueing strategy type: {strategy_type}")

    def predicted_total(candidate: Candidate) -> float:
        states = node_states[candidate.node_id]
        next_available = min(state.available_at_ms for state in states)
        node_arrival = arrival_time_ms + candidate.one_way_network_ms
        return (
            candidate.round_trip_network_ms
            + max(0.0, next_available - node_arrival)
            + candidate.estimated_compute_ms
        )

    return min(
        eligible,
        key=lambda candidate: (
            predicted_total(candidate),
            -candidate.quality,
            _candidate_key(candidate),
        ),
    )


def _historical_or_synthetic_observation(
    arrival: Arrival,
    model: ModelProfile,
    seed: int,
) -> dict[str, Any]:
    observation = dict(arrival.observations.get(model.model_id, {}))
    if observation:
        service_time = _number(
            observation.get("latency_ms"),
            f"{model.model_id}.latency_ms",
            minimum=0.000001,
        )
        correct = bool(observation.get("correct", observation.get("outcome") == "correct"))
        return {
            "service_time_ms": service_time * model.observation_scale,
            "correct": correct,
            "source": "historical",
        }
    # Synthetic correctness is sampled from the configured quality.  Deriving
    # the draw from request/model identity keeps a synthetic request outcome
    # identical when two strategies select the same candidate.
    digest = sha256(f"{seed}:{arrival.sequence}:{model.model_id}".encode()).digest()
    synthetic_random = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return {
        "service_time_ms": model.service_time_ms,
        "correct": synthetic_random < model.quality,
        "source": "synthetic",
    }


def _queue_length(states: Sequence[_ServerState]) -> int:
    return sum(max(0, state.pending_jobs - 1) for state in states)


def _schedule_on_node(
    states: list[_ServerState],
    node_arrival_ms: float,
    compute_ms: float,
) -> tuple[float, float, int, int]:
    for state in states:
        if state.available_at_ms <= node_arrival_ms:
            state.available_at_ms = node_arrival_ms
            state.pending_jobs = 0
    before = _queue_length(states)
    server_index = min(range(len(states)), key=lambda index: states[index].available_at_ms)
    state = states[server_index]
    start_ms = max(node_arrival_ms, state.available_at_ms)
    queue_wait_ms = max(0.0, start_ms - node_arrival_ms)
    state.pending_jobs += 1
    state.available_at_ms = start_ms + compute_ms
    after = _queue_length(states)
    return queue_wait_ms, start_ms, before, after


def _default_strategies(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    first = candidates[0]
    return [
        {
            "name": "fixed",
            "type": "fixed",
            "model_id": first.model_id,
            "node_id": first.node_id,
        },
        {"name": "quality-only", "type": "quality_only"},
        {"name": "queue-aware", "type": "queue_aware"},
    ]


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty queueing run")
    total = [float(row["total_latency_ms"]) for row in rows]
    queue = [float(row["queue_wait_ms"]) for row in rows]
    compute = [float(row["compute_ms"]) for row in rows]
    transmission = [float(row["transmission_ms"]) for row in rows]
    propagation = [float(row["propagation_ms"]) for row in rows]
    request_count = len(rows)
    chosen_models: dict[str, int] = {}
    chosen_nodes: dict[str, int] = {}
    for row in rows:
        chosen_models[str(row["model_id"])] = chosen_models.get(str(row["model_id"]), 0) + 1
        chosen_nodes[str(row["node_id"])] = chosen_nodes.get(str(row["node_id"]), 0) + 1
    return {
        "request_count": request_count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / request_count,
        "slo_satisfaction_rate": sum(bool(row["slo_met"]) for row in rows) / request_count,
        "latency_slo_satisfaction_rate": sum(bool(row["latency_slo_met"]) for row in rows) / request_count,
        "quality_slo_satisfaction_rate": sum(bool(row["quality_slo_met"]) for row in rows) / request_count,
        "quality_loss_vs_oracle_rate": fmean(float(row["quality_loss_vs_oracle"]) for row in rows),
        "latency_ms": {
            "mean": fmean(total),
            "p50": percentile(total, 50),
            "p95": percentile(total, 95),
        },
        "latency_decomposition_ms": {
            "queue_wait": {"mean": fmean(queue), "p95": percentile(queue, 95)},
            "compute": {"mean": fmean(compute), "p95": percentile(compute, 95)},
            "transmission": {"mean": fmean(transmission), "p95": percentile(transmission, 95)},
            "propagation": {"mean": fmean(propagation), "p95": percentile(propagation, 95)},
            "total": {"mean": fmean(total), "p50": percentile(total, 50), "p95": percentile(total, 95)},
        },
        "queue_length": {
            "mean_before": fmean(float(row["queue_length_before"]) for row in rows),
            "max_before": max(int(row["queue_length_before"]) for row in rows),
            "mean_after": fmean(float(row["queue_length_after"]) for row in rows),
            "max_after": max(int(row["queue_length_after"]) for row in rows),
        },
        "model_invocations": chosen_models,
        "node_invocations": chosen_nodes,
        "sources": sorted({str(row["input_source"]) for row in rows}),
    }


def simulate_strategy(
    arrivals: Sequence[Arrival],
    profiles: Mapping[str, ModelProfile],
    nodes: Sequence[NodeProfile],
    candidates: Sequence[Candidate],
    strategy: Mapping[str, Any],
    queue_config: Mapping[str, Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Replay one strategy over a fixed arrival stream."""

    node_states = {
        node.node_id: [_ServerState() for _ in range(node.capacity)]
        for node in nodes
    }
    slo = queue_config.get("slo", {})
    if not isinstance(slo, Mapping):
        raise QueueingConfigError("slo must be an object")
    deadline_ms = _number(
        slo.get("deadline_ms", queue_config.get("deadline_ms", 1e18)),
        "slo.deadline_ms",
    )
    require_correctness = bool(slo.get("require_correctness", True))
    rows: list[dict[str, Any]] = []
    for arrival in sorted(arrivals, key=lambda item: (item.arrival_time_ms, item.sequence)):
        candidate = _choose_candidate(strategy, candidates, queue_config, node_states, arrival.arrival_time_ms)
        observation = _historical_or_synthetic_observation(
            arrival,
            profiles[candidate.model_id],
            seed,
        )
        node_arrival_ms = arrival.arrival_time_ms + candidate.one_way_network_ms
        compute_ms = float(observation["service_time_ms"]) * candidate.service_time_multiplier
        queue_wait_ms, start_ms, queue_before, queue_after = _schedule_on_node(
            node_states[candidate.node_id], node_arrival_ms, compute_ms
        )
        transmission_ms = candidate.transmission_ms * 2.0
        propagation_ms = candidate.propagation_ms * 2.0
        total_ms = queue_wait_ms + compute_ms + transmission_ms + propagation_ms
        correct = bool(observation["correct"])
        latency_slo_met = total_ms <= deadline_ms
        quality_slo_met = correct if require_correctness else candidate.quality >= _quality_threshold(queue_config)
        oracle_correct = any(
            bool(arrival.observations.get(model_id, {}).get("correct"))
            for model_id in profiles
        ) if arrival.observations else True
        rows.append(
            {
                "request_id": arrival.request_id,
                "sequence": arrival.sequence,
                "arrival_time_ms": arrival.arrival_time_ms,
                "arrival_rate_per_second": queue_config.get("_active_rate_per_second"),
                "model_id": candidate.model_id,
                "node_id": candidate.node_id,
                "queue_wait_ms": queue_wait_ms,
                "compute_ms": compute_ms,
                "transmission_ms": transmission_ms,
                "propagation_ms": propagation_ms,
                "total_latency_ms": total_ms,
                "node_arrival_time_ms": node_arrival_ms,
                "service_start_time_ms": start_ms,
                "service_finish_time_ms": start_ms + compute_ms,
                "queue_length_before": queue_before,
                "queue_length_after": queue_after,
                "quality": candidate.quality,
                "correct": correct,
                "quality_slo_met": quality_slo_met,
                "latency_slo_met": latency_slo_met,
                "slo_met": latency_slo_met and quality_slo_met,
                "quality_loss_vs_oracle": float(oracle_correct and not correct),
                "strategy": str(strategy.get("name", strategy.get("type", "unnamed"))),
                "input_source": str(observation["source"]),
            }
        )
    return rows


def run_queue_experiment(
    config: Mapping[str, Any],
    model_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run all configured load points and strategies.

    The returned object contains a flat ``per_request`` list and a nested
    ``scenarios`` aggregate, making it suitable for JSONL plus JSON output.
    """

    queue_config = config.get("queueing", config)
    if not isinstance(queue_config, Mapping):
        raise QueueingConfigError("queueing must be an object")
    profiles, nodes, candidates = _parse_profiles(config, model_records)
    rates = queue_config.get("rates_per_second", queue_config.get("rate_per_second", [0.1]))
    if isinstance(rates, (int, float)) and not isinstance(rates, bool):
        rates = [rates]
    if not isinstance(rates, list) or not rates:
        raise QueueingConfigError("rates_per_second must be a non-empty list")
    strategies = queue_config.get("strategies")
    if strategies is None:
        strategies = _default_strategies(candidates)
    if not isinstance(strategies, list) or not strategies:
        raise QueueingConfigError("strategies must be a non-empty list")
    seed = int(queue_config.get("seed", config.get("project", {}).get("seed", 42)))
    observations = model_records
    network_scenarios = queue_config.get("network_scenarios", [{"name": "baseline"}])
    if not isinstance(network_scenarios, list) or not network_scenarios:
        raise QueueingConfigError("network_scenarios must be a non-empty list")
    all_rows: list[dict[str, Any]] = []
    scenario_reports: list[dict[str, Any]] = []
    for network_index, network_scenario in enumerate(network_scenarios):
        if not isinstance(network_scenario, Mapping):
            raise QueueingConfigError("each network scenario must be an object")
        network_name = str(network_scenario.get("name", f"network-{network_index}"))
        scenario_nodes = _network_scenario_nodes(nodes, network_scenario)
        scenario_candidates = _build_candidates(profiles, scenario_nodes)
        for rate_index, rate_value in enumerate(rates):
            rate = _number(rate_value, "rates_per_second", minimum=0.000001)
            active_config = dict(queue_config)
            active_config["_active_rate_per_second"] = rate
            if "models" not in active_config:
                active_config["models"] = config.get("models", [])
            arrivals = generate_arrivals(
                active_config,
                rate,
                # Keep the same arrivals across network scenarios so a
                # comparison attributes changes to communication parameters.
                seed=seed + rate_index,
                observations=observations,
            )
            strategy_reports: dict[str, Any] = {}
            for strategy_index, strategy in enumerate(strategies):
                if not isinstance(strategy, Mapping):
                    raise QueueingConfigError("each strategy must be an object")
                name = str(strategy.get("name", strategy.get("type", f"strategy-{strategy_index}")))
                if name in strategy_reports:
                    raise QueueingConfigError(f"duplicate strategy name: {name}")
                rows = simulate_strategy(
                    arrivals,
                    profiles,
                    scenario_nodes,
                    scenario_candidates,
                    strategy,
                    active_config,
                    seed=seed + rate_index * 1000 + strategy_index,
                )
                for row in rows:
                    row["strategy"] = name
                    row["network_scenario"] = network_name
                    row["arrival_rate_per_second"] = rate
                all_rows.extend(rows)
                strategy_reports[name] = _summarize_rows(rows)
            scenario_reports.append(
                {
                    "network_scenario": network_name,
                    "arrival_rate_per_second": rate,
                    "request_count": len(arrivals),
                    "strategies": strategy_reports,
                }
            )
    metadata = {
        "kind": "queue_aware_collaborative_inference_simulation",
        "result_type": "simulation_or_historical_replay",
        "online_vllm_measurement": False,
        "seed": seed,
        "model_profiles": {
            model_id: {
                "quality": profile.quality,
                "mean_service_time_ms": profile.service_time_ms,
                "historical_observation_scale": profile.observation_scale,
                "source": profile.source,
            }
            for model_id, profile in profiles.items()
        },
        "nodes": [
            {
                "node_id": node.node_id,
                "capacity": node.capacity,
                "model_ids": list(node.model_ids),
                "transmission_ms_one_way": node.transmission_ms,
                "propagation_ms_one_way": node.propagation_ms,
            }
            for node in nodes
        ],
        "network_scenarios": [
            str(scenario.get("name", f"network-{index}"))
            for index, scenario in enumerate(network_scenarios)
        ],
        "delay_definition": {
            "transmission_ms": "round_trip (twice configured one-way link value)",
            "propagation_ms": "round_trip (twice configured one-way link value)",
            "queue_wait_ms": "time from node arrival until a server starts computation",
            "compute_ms": "historical per-request latency or synthetic service_time_ms",
            "total_latency_ms": "queue_wait + compute + transmission + propagation",
        },
    }
    return {"metadata": metadata, "scenarios": scenario_reports, "per_request": all_rows}


def write_queue_outputs(
    report: Mapping[str, Any],
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    """Write flat JSONL request records and the aggregate JSON report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in report["per_request"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {key: value for key, value in report.items() if key != "per_request"}
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "Arrival",
    "Candidate",
    "ModelProfile",
    "NodeProfile",
    "QueueingConfigError",
    "generate_arrivals",
    "run_queue_experiment",
    "simulate_strategy",
    "write_queue_outputs",
]
