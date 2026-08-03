"""Offline confidence analysis and cascade replay helpers."""

from collections.abc import Sequence
from hashlib import sha256
from itertools import product
from math import ceil, isfinite, prod, sqrt
from random import Random
from statistics import fmean
from typing import Any


ALWAYS_ACCEPT = "always_accept"
ALWAYS_ESCALATE = "always_escalate"
Threshold = float | str


def split_sample_ids(
    sample_ids: Sequence[str],
    *,
    development_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    if not 0 < development_count < len(sample_ids):
        raise ValueError("development_count must leave both splits non-empty")
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: sha256(f"{seed}:{sample_id}".encode()).hexdigest(),
    )
    return ordered[:development_count], ordered[development_count:]


def area_under_roc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    positive_count = sum(labels)
    if positive_count == 0:
        raise ValueError("average precision requires at least one positive label")

    ordered = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positives = 0
    observed = 0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        true_positives += group_positives
        observed += end - index
        result += (group_positives / positive_count) * (
            true_positives / observed
        )
        index = end
    return result


def _percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _confidence_statistics(
    confidence: Sequence[float],
    correct: Sequence[bool],
    low_confidence_fraction: float,
) -> dict[str, float]:
    auroc = area_under_roc(confidence, correct)
    not_correct = [not label for label in correct]
    error_auprc = average_precision(
        [-value for value in confidence],
        not_correct,
    )
    ordered = sorted(
        zip(confidence, not_correct, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    cumulative_errors = 0
    observed = 0
    aurc = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        cumulative_errors += sum(failed for _, failed in ordered[index:end])
        observed += end - index
        aurc += (cumulative_errors / observed) * ((end - index) / len(ordered))
        index = end
    overall_error_rate = fmean(float(value) for value in not_correct)

    low_count = max(1, ceil(len(ordered) * low_confidence_fraction))
    low_records = sorted(
        zip(confidence, not_correct, strict=True),
        key=lambda item: item[0],
    )[:low_count]
    low_error_rate = fmean(float(failed) for _, failed in low_records)
    return {
        "auroc": auroc,
        "error_auprc": error_auprc,
        "overall_error_rate": overall_error_rate,
        "low_confidence_error_rate": low_error_rate,
        "low_confidence_error_lift": low_error_rate - overall_error_rate,
        "aurc": aurc,
        "random_aurc": overall_error_rate,
        "aurc_improvement": overall_error_rate - aurc,
    }


def _confidence_interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "lower": _percentile(values, 2.5),
        "upper": _percentile(values, 97.5),
    }


def _point_biserial(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    numeric_correct = [1.0 if value else 0.0 for value in correct]
    confidence_mean = fmean(confidence)
    correct_mean = fmean(numeric_correct)
    confidence_offsets = [value - confidence_mean for value in confidence]
    correct_offsets = [value - correct_mean for value in numeric_correct]
    denominator = sqrt(
        sum(value * value for value in confidence_offsets)
        * sum(value * value for value in correct_offsets)
    )
    if denominator == 0:
        return 0.0
    return sum(
        left * right
        for left, right in zip(confidence_offsets, correct_offsets, strict=True)
    ) / denominator


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
    }


def _confidence_deciles(
    confidence: Sequence[float],
    correct: Sequence[bool],
) -> list[dict[str, float | int]]:
    ordered = sorted(zip(confidence, correct, strict=True), key=lambda item: item[0])
    group_count = min(10, len(ordered))
    groups: list[dict[str, float | int]] = []
    for group_index in range(group_count):
        start = group_index * len(ordered) // group_count
        end = (group_index + 1) * len(ordered) // group_count
        group = ordered[start:end]
        group_confidence = [item[0] for item in group]
        groups.append(
            {
                "index": group_index + 1,
                "count": len(group),
                "confidence_min": min(group_confidence),
                "confidence_max": max(group_confidence),
                "mean_confidence": fmean(group_confidence),
                "accuracy": fmean(float(item[1]) for item in group),
                "error_rate": fmean(float(not item[1]) for item in group),
            }
        )
    return groups


def analyze_confidence(
    records: Sequence[dict[str, Any]],
    *,
    low_confidence_fraction: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("confidence analysis requires records")
    if not 0 < low_confidence_fraction < 1:
        raise ValueError("low_confidence_fraction must be between 0 and 1")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    scorable: list[tuple[float, bool]] = []
    for record in records:
        confidence = record.get("confidence")
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("record has invalid correctness label")
        if (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and isfinite(confidence)
            and 0 <= confidence <= 1
            and record.get("outcome") != "inference_failure"
        ):
            scorable.append((float(confidence), correct))
    confidence = [item[0] for item in scorable]
    correct = [item[1] for item in scorable]
    if len(scorable) < 2 or all(correct) or not any(correct):
        failure_reason = (
            "requires_at_least_two_valid_scores"
            if len(scorable) < 2
            else "requires_both_correctness_classes"
        )
        empty_interval = {"lower": None, "upper": None}
        gate_checks = {
            "auroc_ci_lower_above_chance": False,
            "low_confidence_error_lift_ci_lower_above_zero": False,
            "aurc_improvement_ci_lower_above_zero": False,
        }
        return {
            "request_count": len(records),
            "scorable_count": len(scorable),
            "invalid_confidence_count": len(records) - len(scorable),
            "auroc": None,
            "error_auprc": None,
            "overall_error_rate": (
                fmean(float(not value) for value in correct) if correct else None
            ),
            "low_confidence_error_rate": None,
            "low_confidence_error_lift": None,
            "aurc": None,
            "random_aurc": None,
            "aurc_improvement": None,
            "point_biserial_correlation": (
                _point_biserial(confidence, correct) if confidence else None
            ),
            "confidence_by_correctness": {
                "correct": _distribution(
                    [value for value, label in scorable if label]
                ),
                "not_correct": _distribution(
                    [value for value, label in scorable if not label]
                ),
            },
            "confidence_deciles": (
                _confidence_deciles(confidence, correct) if confidence else []
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "confidence_intervals_95": {
                "auroc": dict(empty_interval),
                "low_confidence_error_lift": dict(empty_interval),
                "aurc_improvement": dict(empty_interval),
            },
            "gate_checks": gate_checks,
            "failure_reason": failure_reason,
            "passed": False,
        }

    metrics = _confidence_statistics(
        confidence,
        correct,
        low_confidence_fraction,
    )

    bootstrap_values = {
        "auroc": [],
        "low_confidence_error_lift": [],
        "aurc_improvement": [],
    }
    random = Random(bootstrap_seed)
    for _ in range(bootstrap_iterations):
        indices = [random.randrange(len(scorable)) for _ in scorable]
        sampled_confidence = [confidence[index] for index in indices]
        sampled_correct = [correct[index] for index in indices]
        if all(sampled_correct) or not any(sampled_correct):
            continue
        sampled = _confidence_statistics(
            sampled_confidence,
            sampled_correct,
            low_confidence_fraction,
        )
        for name in bootstrap_values:
            bootstrap_values[name].append(sampled[name])
    if not all(bootstrap_values.values()):
        raise ValueError("bootstrap samples did not contain both correctness classes")

    intervals = {
        name: _confidence_interval(values)
        for name, values in bootstrap_values.items()
    }
    gate_checks = {
        "auroc_ci_lower_above_chance": intervals["auroc"]["lower"] > 0.5,
        "low_confidence_error_lift_ci_lower_above_zero": (
            intervals["low_confidence_error_lift"]["lower"] > 0
        ),
        "aurc_improvement_ci_lower_above_zero": (
            intervals["aurc_improvement"]["lower"] > 0
        ),
    }
    return {
        "request_count": len(records),
        "scorable_count": len(scorable),
        "invalid_confidence_count": len(records) - len(scorable),
        **metrics,
        "point_biserial_correlation": _point_biserial(confidence, correct),
        "confidence_by_correctness": {
            "correct": _distribution(
                [value for value, label in scorable if label]
            ),
            "not_correct": _distribution(
                [value for value, label in scorable if not label]
            ),
        },
        "confidence_deciles": _confidence_deciles(confidence, correct),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "confidence_intervals_95": intervals,
        "gate_checks": gate_checks,
        "failure_reason": None,
        "passed": all(gate_checks.values()),
    }


def analyze_confidence_models(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
    *,
    development_count: int,
    split_seed: int,
    low_confidence_fraction: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(records_by_model) < 2:
        raise ValueError("confidence analysis requires at least two models")
    if any(not records for records in records_by_model):
        raise ValueError("confidence analysis cannot use empty model results")

    indexed_models: list[dict[str, dict[str, Any]]] = []
    model_ids: list[str] = []
    for model_index, records in enumerate(records_by_model):
        indexed: dict[str, dict[str, Any]] = {}
        ids: set[str] = set()
        for record in records:
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"model[{model_index}] has invalid sample_id")
            if sample_id in indexed:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            model_id = record.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"model[{model_index}] has invalid model_id")
            ids.add(model_id)
            indexed[sample_id] = record
        if len(ids) != 1:
            raise ValueError("each input must contain exactly one model_id")
        indexed_models.append(indexed)
        model_ids.append(next(iter(ids)))
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_id values must be unique")

    sample_ids = [record["sample_id"] for record in records_by_model[0]]
    expected_ids = set(sample_ids)
    baseline = indexed_models[0]
    common_fields = (
        "dataset_index",
        "question",
        "prompt",
        "reference_answer",
        "experiment_config",
    )
    for model_index, indexed in enumerate(indexed_models[1:], start=1):
        if set(indexed) != expected_ids:
            raise ValueError(f"sample_id mismatch for model[{model_index}]")
        for sample_id in sample_ids:
            for field in common_fields:
                if indexed[sample_id].get(field) != baseline[sample_id].get(field):
                    raise ValueError(f"{field} mismatch for sample {sample_id}")

    development_ids, evaluation_ids = split_sample_ids(
        sample_ids,
        development_count=development_count,
        seed=split_seed,
    )
    model_analyses: list[dict[str, Any]] = []
    for model_id, indexed in zip(model_ids, indexed_models, strict=True):
        analysis = analyze_confidence(
            [indexed[sample_id] for sample_id in development_ids],
            low_confidence_fraction=low_confidence_fraction,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        analysis["model_id"] = model_id
        model_analyses.append(analysis)

    intermediate = model_analyses[:-1]
    passed_model_ids = [
        analysis["model_id"] for analysis in intermediate if analysis["passed"]
    ]
    return {
        "scope": "development_confidence_gate",
        "evaluation_outcomes_read": False,
        "split": {
            "method": "sha256_seed_sample_id",
            "seed": split_seed,
            "development_count": len(development_ids),
            "evaluation_count": len(evaluation_ids),
            "development_sample_ids": development_ids,
            "evaluation_sample_ids": evaluation_ids,
        },
        "intermediate_model_ids": model_ids[:-1],
        "terminal_model_id": model_ids[-1],
        "passed_intermediate_model_ids": passed_model_ids,
        "failed_intermediate_model_ids": [
            analysis["model_id"]
            for analysis in intermediate
            if not analysis["passed"]
        ],
        "all_intermediate_models_failed": not passed_model_ids,
        "models": model_analyses,
    }


def _index_replay_records(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
    sample_ids: Sequence[str],
) -> tuple[list[dict[str, dict[str, Any]]], list[str]]:
    if len(records_by_model) < 2:
        raise ValueError("cascade replay requires at least two models")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("cascade sample_id values must be unique")
    expected = set(sample_ids)
    indexed_models: list[dict[str, dict[str, Any]]] = []
    model_ids: list[str] = []
    for records in records_by_model:
        indexed = {record["sample_id"]: record for record in records}
        if len(indexed) != len(records):
            raise ValueError("cascade input contains duplicate sample_id")
        if not expected.issubset(indexed):
            raise ValueError("cascade input is missing requested sample_id values")
        ids = {record.get("model_id") for record in records}
        if len(ids) != 1 or not isinstance(next(iter(ids)), str):
            raise ValueError("each cascade input must contain one model_id")
        indexed_models.append(indexed)
        model_ids.append(next(iter(ids)))
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("cascade model_id values must be unique")
    return indexed_models, model_ids


def _is_valid_confidence(record: dict[str, Any]) -> bool:
    confidence = record.get("confidence")
    return (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and isfinite(confidence)
        and 0 <= confidence <= 1
    )


def _can_be_accepted(record: dict[str, Any]) -> bool:
    return record.get("outcome") in {"correct", "incorrect"}


def _accepts_threshold(record: dict[str, Any], threshold: Threshold) -> bool:
    if not _can_be_accepted(record) or not _is_valid_confidence(record):
        return False
    if threshold == ALWAYS_ACCEPT:
        return True
    if threshold == ALWAYS_ESCALATE:
        return False
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"invalid threshold candidate: {threshold}")
    return float(record["confidence"]) >= float(threshold)


def replay_cascade(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
    sample_ids: Sequence[str],
    thresholds: Sequence[Threshold],
) -> dict[str, Any]:
    if len(thresholds) != len(records_by_model) - 1:
        raise ValueError("one threshold is required for each intermediate model")
    indexed_models, model_ids = _index_replay_records(records_by_model, sample_ids)
    model_invocations = [0 for _ in model_ids]
    model_returns = [0 for _ in model_ids]
    eligible_counts = [0 for _ in thresholds]
    accepted_counts = [0 for _ in thresholds]
    per_request: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        cumulative_latency = 0.0
        decisions: list[dict[str, Any]] = []
        final_index = len(model_ids) - 1
        for model_index, indexed in enumerate(indexed_models):
            record = indexed[sample_id]
            model_invocations[model_index] += 1
            cumulative_latency += float(record["latency_ms"])
            if model_index == len(thresholds):
                decisions.append(
                    {
                        "model_id": model_ids[model_index],
                        "action": "return",
                        "reason": "terminal_model",
                    }
                )
                final_index = model_index
                break

            if _can_be_accepted(record) and _is_valid_confidence(record):
                eligible_counts[model_index] += 1
            if _accepts_threshold(record, thresholds[model_index]):
                accepted_counts[model_index] += 1
                decisions.append(
                    {
                        "model_id": model_ids[model_index],
                        "action": "return",
                        "reason": "confidence_threshold_met",
                    }
                )
                final_index = model_index
                break
            reason = (
                "operational_failure"
                if not _can_be_accepted(record) or not _is_valid_confidence(record)
                else "confidence_below_threshold"
            )
            decisions.append(
                {
                    "model_id": model_ids[model_index],
                    "action": "escalate",
                    "reason": reason,
                }
            )

        final_record = indexed_models[final_index][sample_id]
        model_returns[final_index] += 1
        per_request.append(
            {
                "sample_id": sample_id,
                "final_model_id": model_ids[final_index],
                "final_outcome": final_record["outcome"],
                "correct": bool(final_record["correct"]),
                "model_call_count": final_index + 1,
                "escalation_count": final_index,
                "cumulative_latency_ms": cumulative_latency,
                "decisions": decisions,
            }
        )

    latencies = [float(item["cumulative_latency_ms"]) for item in per_request]
    correct_count = sum(bool(item["correct"]) for item in per_request)
    outcomes = {
        outcome: {
            "count": sum(item["final_outcome"] == outcome for item in per_request),
            "sample_ids": [
                item["sample_id"]
                for item in per_request
                if item["final_outcome"] == outcome
            ],
        }
        for outcome in (
            "correct",
            "incorrect",
            "parse_failure",
            "inference_failure",
        )
    }
    return {
        "request_count": len(sample_ids),
        "model_order": model_ids,
        "thresholds": list(thresholds),
        "correct_count": correct_count,
        "accuracy": correct_count / len(sample_ids),
        "outcomes": outcomes,
        "latency_ms": {
            "mean": fmean(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "mean_model_calls": fmean(
            float(item["model_call_count"]) for item in per_request
        ),
        "model_invocations": model_invocations,
        "model_invocation_rates": [
            count / len(sample_ids) for count in model_invocations
        ],
        "model_returns": model_returns,
        "model_return_rates": [count / len(sample_ids) for count in model_returns],
        "stage_eligible_counts": eligible_counts,
        "stage_accepted_counts": accepted_counts,
        "stage_acceptance_rates": [
            accepted / eligible if eligible else 0.0
            for accepted, eligible in zip(
                accepted_counts,
                eligible_counts,
                strict=True,
            )
        ],
        "terminal_invocation_count": model_invocations[-1],
        "terminal_invocation_rate": model_invocations[-1] / len(sample_ids),
        "latency_scope": "offline_replay_estimate",
        "latency_exclusions": [
            "model_loading",
            "model_switching",
            "queueing",
            "concurrency_interference",
        ],
        "per_request": per_request,
    }


def build_threshold_candidates(
    confidence: Sequence[float],
    *,
    quantile_step: float,
) -> list[Threshold]:
    if not confidence:
        raise ValueError("threshold candidates require confidence values")
    if not 0 < quantile_step < 1:
        raise ValueError("quantile_step must be between 0 and 1")
    values = [float(value) for value in confidence]
    if not all(isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("threshold candidates require finite confidence values")

    numeric: list[float] = []
    quantile_index = 1
    while quantile_index * quantile_step < 1:
        value = _percentile(values, quantile_index * quantile_step * 100)
        if not numeric or value != numeric[-1]:
            numeric.append(value)
        quantile_index += 1
    return [ALWAYS_ACCEPT, *numeric, ALWAYS_ESCALATE]


def _threshold_sort_key(threshold: Threshold) -> tuple[int, float]:
    if threshold == ALWAYS_ACCEPT:
        return (0, 0.0)
    if threshold == ALWAYS_ESCALATE:
        return (2, 0.0)
    return (1, float(threshold))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["correct_count"] >= right["correct_count"]
        and left["total_cumulative_latency_ms"]
        <= right["total_cumulative_latency_ms"]
        and left["terminal_invocation_count"] <= right["terminal_invocation_count"]
    )
    strictly_better = (
        left["correct_count"] > right["correct_count"]
        or left["total_cumulative_latency_ms"]
        < right["total_cumulative_latency_ms"]
        or left["terminal_invocation_count"] < right["terminal_invocation_count"]
    )
    return no_worse and strictly_better


def search_pareto_thresholds(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
    sample_ids: Sequence[str],
    *,
    quantile_step: float,
) -> dict[str, Any]:
    indexed_models, model_ids = _index_replay_records(records_by_model, sample_ids)
    candidate_sets = [
        build_threshold_candidates(
            [
                float(indexed[sample_id]["confidence"])
                for sample_id in sample_ids
                if _can_be_accepted(indexed[sample_id])
                and _is_valid_confidence(indexed[sample_id])
            ],
            quantile_step=quantile_step,
        )
        for indexed in indexed_models[:-1]
    ]
    target_correct_count = sum(
        bool(indexed_models[-1][sample_id]["correct"])
        for sample_id in sample_ids
    )
    frontier: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    feasible_count = 0

    for thresholds in product(*candidate_sets):
        correct_count = 0
        total_latency = 0.0
        terminal_invocations = 0
        for sample_id in sample_ids:
            final_index = len(indexed_models) - 1
            for model_index, threshold in enumerate(thresholds):
                record = indexed_models[model_index][sample_id]
                total_latency += float(record["latency_ms"])
                if _accepts_threshold(record, threshold):
                    final_index = model_index
                    break
            if final_index == len(indexed_models) - 1:
                total_latency += float(indexed_models[-1][sample_id]["latency_ms"])
                terminal_invocations += 1
            correct_count += bool(indexed_models[final_index][sample_id]["correct"])

        candidate = {
            "thresholds": list(thresholds),
            "correct_count": correct_count,
            "accuracy": correct_count / len(sample_ids),
            "total_cumulative_latency_ms": total_latency,
            "mean_cumulative_latency_ms": total_latency / len(sample_ids),
            "terminal_invocation_count": terminal_invocations,
            "terminal_invocation_rate": terminal_invocations / len(sample_ids),
        }
        if correct_count >= target_correct_count:
            feasible_count += 1
            selection_key = (
                candidate["total_cumulative_latency_ms"],
                candidate["terminal_invocation_count"],
                -candidate["correct_count"],
                tuple(_threshold_sort_key(value) for value in thresholds),
            )
            if selected is None or selection_key < selected["_selection_key"]:
                selected = {**candidate, "_selection_key": selection_key}

        if any(_dominates(item, candidate) for item in frontier):
            continue
        frontier = [
            item for item in frontier if not _dominates(candidate, item)
        ]
        if not any(
            item["correct_count"] == candidate["correct_count"]
            and item["total_cumulative_latency_ms"]
            == candidate["total_cumulative_latency_ms"]
            and item["terminal_invocation_count"]
            == candidate["terminal_invocation_count"]
            for item in frontier
        ):
            frontier.append(candidate)

    if selected is None:
        raise ValueError("no threshold combination preserves terminal-model accuracy")
    selected.pop("_selection_key")
    frontier.sort(
        key=lambda item: (
            item["mean_cumulative_latency_ms"],
            -item["accuracy"],
            item["terminal_invocation_rate"],
        )
    )
    return {
        "model_order": model_ids,
        "quantile_step": quantile_step,
        "candidate_counts_per_stage": [len(values) for values in candidate_sets],
        "candidate_combination_count": prod(len(values) for values in candidate_sets),
        "target_model_id": model_ids[-1],
        "target_correct_count": target_correct_count,
        "target_accuracy": target_correct_count / len(sample_ids),
        "feasible_candidate_count": feasible_count,
        "selected": selected,
        "pareto_frontier": frontier,
    }


def _random_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "lower": _percentile(values, 2.5),
        "upper": _percentile(values, 97.5),
    }


def random_acceptance_baseline(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
    sample_ids: Sequence[str],
    *,
    acceptance_rates: Sequence[float],
    seeds: Sequence[int],
) -> dict[str, Any]:
    if len(acceptance_rates) != len(records_by_model) - 1:
        raise ValueError("one random acceptance rate is required per intermediate model")
    if not seeds:
        raise ValueError("random baseline requires at least one seed")
    if not all(0 <= rate <= 1 for rate in acceptance_rates):
        raise ValueError("random acceptance rates must be between 0 and 1")
    indexed_models, model_ids = _index_replay_records(records_by_model, sample_ids)
    run_accuracy: list[float] = []
    run_latency: list[float] = []
    run_terminal_rates: list[float] = []
    stage_rates: list[list[float]] = [[] for _ in acceptance_rates]

    for seed in seeds:
        random = Random(seed)
        correct_count = 0
        total_latency = 0.0
        terminal_count = 0
        eligible = [0 for _ in acceptance_rates]
        accepted = [0 for _ in acceptance_rates]
        for sample_id in sample_ids:
            final_index = len(indexed_models) - 1
            for model_index, acceptance_rate in enumerate(acceptance_rates):
                record = indexed_models[model_index][sample_id]
                total_latency += float(record["latency_ms"])
                if not _can_be_accepted(record) or not _is_valid_confidence(record):
                    continue
                eligible[model_index] += 1
                if random.random() < acceptance_rate:
                    accepted[model_index] += 1
                    final_index = model_index
                    break
            if final_index == len(indexed_models) - 1:
                total_latency += float(indexed_models[-1][sample_id]["latency_ms"])
                terminal_count += 1
            correct_count += bool(indexed_models[final_index][sample_id]["correct"])
        run_accuracy.append(correct_count / len(sample_ids))
        run_latency.append(total_latency / len(sample_ids))
        run_terminal_rates.append(terminal_count / len(sample_ids))
        for model_index in range(len(acceptance_rates)):
            stage_rates[model_index].append(
                accepted[model_index] / eligible[model_index]
                if eligible[model_index]
                else 0.0
            )

    return {
        "scope": "confidence_independent_random_acceptance",
        "model_order": model_ids,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "target_stage_acceptance_rates": list(acceptance_rates),
        "mean_stage_acceptance_rates": [fmean(values) for values in stage_rates],
        "accuracy": _random_summary(run_accuracy),
        "mean_cumulative_latency_ms": _random_summary(run_latency),
        "terminal_invocation_rate": _random_summary(run_terminal_rates),
        "mean_terminal_invocation_rate": fmean(run_terminal_rates),
        "confidence_used": False,
    }
