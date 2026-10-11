"""Event-ordered replay of measured RAG stage chains on a fixed topology.

Real model outputs are immutable. Placement can change queue/network latency
and payload transfers, but never invent a change in answer correctness.
"""

from collections import Counter, deque
import heapq
from math import isfinite
from statistics import fmean

from scaleflow.baseline import percentile
from scaleflow.queueing import generate_arrivals
from scaleflow.rag import json_bytes


def transfer(payload: dict, source: str, destination: str, links: list[dict]) -> dict:
    size = len(json_bytes(payload))
    if source == destination:
        return {"source": source, "destination": destination, "payload_bytes": size,
                "transmitted_bytes": 0, "transmission_ms": 0.0, "propagation_ms": 0.0}
    link = next((link for link in links if (link["source"], link["destination"]) == (source, destination)
                 or link.get("bidirectional", True) and (link["destination"], link["source"]) == (source, destination)), None)
    if link is None:
        raise ValueError(f"no fixed topology link {source} -> {destination}")
    bandwidth, propagation = float(link["bandwidth_mbps"]), float(link["propagation_ms"])
    if not isfinite(bandwidth) or bandwidth <= 0 or not isfinite(propagation) or propagation < 0:
        raise ValueError("link bandwidth must be finite positive and propagation finite nonnegative")
    return {"source": source, "destination": destination, "payload_bytes": size,
            "transmitted_bytes": size, "transmission_ms": size * 8 / (bandwidth * 1000), "propagation_ms": propagation}


def summarize_replay(rows: list[dict]) -> dict:
    return {"task_count": len(rows), "task_success_rate": fmean(r["correct"] for r in rows),
            "latency_ms": {"mean": fmean(r["total_latency_ms"] for r in rows), "p95": percentile([r["total_latency_ms"] for r in rows], 95)},
            "communication_bytes": {"mean_per_task": fmean(r["communication_bytes"] for r in rows), "total": sum(r["communication_bytes"] for r in rows)},
            "latency_decomposition_mean_ms": {field: fmean(r[field] for r in rows) for field in ("queue_wait_ms", "compute_ms", "transmission_ms", "propagation_ms")},
            "stage_node_counts": dict(Counter(f"{s['stage']}:{s['node_id']}" for r in rows for s in r["stages"]))}


def replay_workflow(records: list[dict], simulation: dict, scenario: dict, mode: str, policy: str, rate: float) -> list[dict]:
    """Transfer arrivals join FCFS queues; completion events release dependencies."""
    nodes = {n["node_id"]: n for n in simulation["nodes"]}
    for node in nodes.values():
        if not isinstance(node["capacity"], int) or isinstance(node["capacity"], bool) or node["capacity"] < 1:
            raise ValueError("node capacity must be a positive integer")
        if any(not isfinite(float(v)) or v <= 0 for v in node["service_time_multiplier"].values()):
            raise ValueError("service_time_multiplier must be finite positive")
    client = simulation["client_node"]
    if client not in nodes:
        raise ValueError("client_node is missing")
    selected = [row for row in records if row["mode"] == mode]
    by_id = {row["sample_id"]: row for row in selected}
    if len(by_id) != len(selected) or not selected:
        raise ValueError("replay measurements must contain distinct samples for each mode")
    if policy not in {"fixed", "network_aware"}:
        raise ValueError("unknown RAG placement policy")
    for row in selected:
        if not row["stages"] or row["stages"][-1]["stage"] != "answer":
            raise ValueError("RAG stage chain must end with answer")
        for stage in row["stages"]:
            if not isfinite(float(stage["compute_ms"])) or stage["compute_ms"] < 0:
                raise ValueError("measured compute_ms must be finite nonnegative")
    # Reuse Phase 12's deterministic arrival process/sample-ID alignment.
    arrivals = generate_arrivals(simulation["arrival"], rate_per_second=rate, seed=simulation["seed"], observations={"workflow": selected})
    means = {}
    for row in selected:
        for stage in row["stages"]:
            key = stage["stage"], stage["model_id"]
            means.setdefault(key, []).append(stage["compute_ms"])
    means = {key: fmean(values) for key, values in means.items()}
    queues = {node: deque() for node in nodes}
    busy = {node: [None] * profile["capacity"] for node, profile in nodes.items()}
    events = []
    serial = 0
    def push(time: float, priority: int, kind: str, data: tuple) -> None:
        nonlocal serial
        heapq.heappush(events, (time, priority, serial, kind, data))
        serial += 1
    tasks = []
    for arrival in arrivals:
        record = by_id[arrival.request_id]
        task = {"request_id": f"{arrival.sequence}:{arrival.request_id}", "sample_id": arrival.request_id,
                "arrival_time_ms": arrival.arrival_time_ms, "mode": mode, "policy": policy,
                "network_scenario": scenario["name"], "arrival_rate_per_second": rate,
                "correct": record["correct"], "pipeline_success": record["pipeline_success"],
                "predicted_answer": record["predicted_answer"], "gold_answer": record["gold_answer"],
                "stages": [], "transfers": [], "record": record,
                "result_type": "historical_replay_of_real_rag_stages_with_simulated_queue_and_network"}
        tasks.append(task)
        push(arrival.arrival_time_ms, 1, "ready", (len(tasks)-1, 0, client))

    def multiplier(node: str, model: str) -> float:
        return float(nodes[node]["service_time_multiplier"].get(model, 1.0))

    def estimated_start(node: str, now: float) -> float:
        slots = [now if job is None else max(now, job["estimated_finish_ms"]) for job in busy[node]]
        for task_id, stage_id, _, _ in queues[node]:
            stage = tasks[task_id]["record"]["stages"][stage_id]
            server = min(range(len(slots)), key=slots.__getitem__)
            slots[server] += means[stage["stage"], stage["model_id"]] * multiplier(node, stage["model_id"])
        return min(slots)

    def choose(task_id: int, stage_id: int, source: str, now: float) -> str:
        task = tasks[task_id]
        stage = task["record"]["stages"][stage_id]
        candidates = [node for node, p in nodes.items() if stage["model_id"] in p["model_ids"]]
        if not candidates:
            raise ValueError(f"no node supports stage model {stage['model_id']}")
        if mode != "multi_model_rag":
            candidates = [simulation["single_node"]]
        elif policy == "fixed":
            candidates = [simulation["fixed_stage_nodes"][stage["stage"]]]
        if any(c not in nodes or stage["model_id"] not in nodes[c]["model_ids"] for c in candidates):
            raise ValueError("fixed node does not support stage model")
        def cost(node: str) -> tuple:
            message = transfer(stage["input"], source, node, scenario["links"])
            node_arrival = now + message["transmission_ms"] + message["propagation_ms"]
            start = max(node_arrival, estimated_start(node, now))
            end = start + means[stage["stage"], stage["model_id"]] * multiplier(node, stage["model_id"])
            if stage_id == len(task["record"]["stages"]) - 1:
                # Use mean output bytes for prediction; final answer not available at decision time.
                byte_count = fmean(len(json_bytes(r["stages"][-1]["output"])) for r in selected)
                link_cost = transfer({}, node, client, scenario["links"])
                if link_cost["transmitted_bytes"]:
                    end += link_cost["propagation_ms"] + link_cost["transmission_ms"] * byte_count / link_cost["payload_bytes"]
            return end, message["transmitted_bytes"], node
        return min(candidates, key=cost)

    def start_queued(node: str, now: float) -> None:
        for server in range(len(busy[node])):
            if busy[node][server] is not None or not queues[node]:
                continue
            task_id, stage_id, queued_at, queue_length = queues[node].popleft()
            stage = tasks[task_id]["record"]["stages"][stage_id]
            compute = stage["compute_ms"] * multiplier(node, stage["model_id"])
            result = {"stage": stage["stage"], "model_id": stage["model_id"], "node_id": node,
                      "queue_length_on_arrival": queue_length, "node_arrival_ms": queued_at,
                      "service_start_ms": now, "service_finish_ms": now + compute,
                      "queue_wait_ms": now - queued_at, "compute_ms": compute,
                      "input_bytes": len(json_bytes(stage["input"])), "output_bytes": len(json_bytes(stage["output"]))}
            tasks[task_id]["stages"].append(result)
            busy[node][server] = {"estimated_finish_ms": now + means[stage["stage"], stage["model_id"]] * multiplier(node, stage["model_id"])}
            push(now + compute, 0, "finish", (task_id, stage_id, node, server))

    while events:
        now, _, _, kind, data = heapq.heappop(events)
        if kind == "ready":
            task_id, stage_id, source = data
            node = choose(task_id, stage_id, source, now)
            stage = tasks[task_id]["record"]["stages"][stage_id]
            message = transfer(stage["input"], source, node, scenario["links"])
            message["stage"] = stage["stage"]
            tasks[task_id]["transfers"].append(message)
            push(now + message["transmission_ms"] + message["propagation_ms"], 2, "enqueue", (task_id, stage_id, node))
        elif kind == "enqueue":
            task_id, stage_id, node = data
            queues[node].append((task_id, stage_id, now, len(queues[node])))
            start_queued(node, now)
        elif kind == "finish":
            task_id, stage_id, node, server = data
            busy[node][server] = None
            start_queued(node, now)
            record = tasks[task_id]["record"]
            if stage_id + 1 < len(record["stages"]):
                push(now, 1, "ready", (task_id, stage_id + 1, node))
            else:
                message = transfer(record["stages"][-1]["output"], node, client, scenario["links"])
                message["stage"] = "answer_return"
                tasks[task_id]["transfers"].append(message)
                tasks[task_id]["completed_at_ms"] = now + message["transmission_ms"] + message["propagation_ms"]
    for task in tasks:
        task.pop("record")
        task["queue_wait_ms"] = sum(s["queue_wait_ms"] for s in task["stages"])
        task["compute_ms"] = sum(s["compute_ms"] for s in task["stages"])
        task["transmission_ms"] = sum(t["transmission_ms"] for t in task["transfers"])
        task["propagation_ms"] = sum(t["propagation_ms"] for t in task["transfers"])
        task["communication_bytes"] = sum(t["transmitted_bytes"] for t in task["transfers"])
        task["total_latency_ms"] = task["completed_at_ms"] - task["arrival_time_ms"]
    return tasks


def run_rag_replay(records: list[dict], simulation: dict) -> dict:
    rows, scenarios = [], []
    for scenario in simulation["network_scenarios"]:
        for rate in simulation["arrival_rates_per_second"]:
            metrics = {}
            for mode, policy in (("direct", "fixed"), ("single_node_rag", "fixed"), ("multi_model_rag", "fixed"), ("multi_model_rag", "network_aware")):
                result = replay_workflow(records, simulation, scenario, mode, policy, rate)
                rows.extend(result)
                metrics[f"{mode}:{policy}"] = summarize_replay(result)
            scenarios.append({"network_scenario": scenario["name"], "arrival_rate_per_second": rate, "metrics": metrics})
    return {"metadata": {"result_type": "historical_replay", "real_online_distributed_measurement": False,
                         "real_inputs": "measured vLLM query/answer outputs and CPU retrieval latencies",
                         "synthetic_assumptions": "fixed topology, service multipliers, capacity, Poisson arrivals; non-contending links",
                         "prediction": "in-sample stage means; greedy estimated stage finish including incoming link and final return; no correctness oracle",
                         "communication": "exact compact sorted UTF-8 JSON payload on cross-node transitions incl. final return; protocol overhead excluded; corpus preplaced",
                         "nodes": simulation["nodes"], "network_scenarios": simulation["network_scenarios"],
                         "arrival": simulation["arrival"], "seed": simulation["seed"]},
            "scenarios": scenarios, "per_request": rows}
