import json

import pytest

from scaleflow.queueing import generate_arrivals, run_queue_experiment


def _config() -> dict:
    return {
        "project": {"seed": 7},
        "models": [
            {"model_id": "small", "quality": 0.8, "service_time_ms": 10.0},
            {"model_id": "large", "quality": 0.95, "service_time_ms": 100.0},
        ],
        "nodes": [
            {
                "node_id": "edge",
                "capacity": 1,
                "model_ids": ["small", "large"],
                "network": {"transmission_ms": 2.0, "propagation_ms": 3.0},
            }
        ],
        "queueing": {
            "process": "periodic",
            "request_count": 3,
            "rate_per_second": 1000.0,
            "seed": 7,
            "slo": {
                "deadline_ms": 50.0,
                "min_quality": 0.8,
                "require_correctness": True,
            },
            "strategies": [
                {
                    "name": "fixed-small",
                    "type": "fixed",
                    "model_id": "small",
                    "node_id": "edge",
                },
                {"name": "quality-only", "type": "quality_only"},
                {"name": "queue-aware", "type": "queue_aware"},
            ],
        },
    }


def test_arrival_stream_is_reproducible_and_preserves_event_order() -> None:
    first = generate_arrivals(
        {"process": "poisson", "request_count": 8},
        4.0,
        seed=19,
        observations={"small": [{"sample_id": f"s{i}"} for i in range(8)]},
    )
    second = generate_arrivals(
        {"process": "poisson", "request_count": 8},
        4.0,
        seed=19,
        observations={"small": [{"sample_id": f"s{i}"} for i in range(8)]},
    )
    assert first == second
    assert [item.sequence for item in first] == list(range(8))
    assert [item.arrival_time_ms for item in first] == sorted(
        item.arrival_time_ms for item in first
    )


def test_queue_updates_and_latency_decomposition_are_explicit() -> None:
    report = run_queue_experiment(_config())
    rows = [row for row in report["per_request"] if row["strategy"] == "fixed-small"]
    assert [row["queue_wait_ms"] for row in rows] == pytest.approx([0.0, 9.0, 18.0])
    assert [row["queue_length_before"] for row in rows] == [0, 0, 1]
    assert [row["queue_length_after"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["transmission_ms"] == pytest.approx(4.0)
        assert row["propagation_ms"] == pytest.approx(6.0)
        assert row["total_latency_ms"] == pytest.approx(
            row["queue_wait_ms"]
            + row["compute_ms"]
            + row["transmission_ms"]
            + row["propagation_ms"]
        )


def test_node_capacity_allows_parallel_service_before_queueing() -> None:
    config = _config()
    config["nodes"][0]["capacity"] = 2
    report = run_queue_experiment(config)
    rows = [row for row in report["per_request"] if row["strategy"] == "fixed-small"]
    assert [row["queue_wait_ms"] for row in rows] == pytest.approx([0.0, 0.0, 8.0])
    assert rows[0]["queue_length_after"] == 0
    assert rows[1]["queue_length_after"] == 0
    assert rows[2]["queue_length_before"] == 0


def test_slo_separates_deadline_and_quality_conditions() -> None:
    config = _config()
    config["queueing"]["slo"]["deadline_ms"] = 25.0
    config["queueing"]["slo"]["require_correctness"] = False
    report = run_queue_experiment(config)
    rows = [row for row in report["per_request"] if row["strategy"] == "fixed-small"]
    assert [row["latency_slo_met"] for row in rows] == [True, False, False]
    assert all(row["quality_slo_met"] for row in rows)
    assert [row["slo_met"] for row in rows] == [True, False, False]


def test_network_scenarios_reuse_arrivals_for_communication_comparison() -> None:
    config = _config()
    config["queueing"]["network_scenarios"] = [
        {"name": "baseline"},
        {
            "name": "slow-link",
            "overrides": {"edge": {"transmission_ms": 20.0, "propagation_ms": 10.0}},
        },
    ]
    report = run_queue_experiment(config)
    assert [scenario["network_scenario"] for scenario in report["scenarios"]] == [
        "baseline",
        "slow-link",
    ]
    baseline = [
        row
        for row in report["per_request"]
        if row["network_scenario"] == "baseline" and row["strategy"] == "fixed-small"
    ]
    slow = [
        row
        for row in report["per_request"]
        if row["network_scenario"] == "slow-link" and row["strategy"] == "fixed-small"
    ]
    assert [row["request_id"] for row in baseline] == [row["request_id"] for row in slow]
    assert [row["arrival_time_ms"] for row in baseline] == [row["arrival_time_ms"] for row in slow]
    assert all(slow_row["total_latency_ms"] > base_row["total_latency_ms"] for base_row, slow_row in zip(baseline, slow, strict=True))


def test_quality_only_and_queue_aware_are_deterministic_and_can_diverge() -> None:
    config = _config()
    config["queueing"]["request_count"] = 30
    config["queueing"]["rate_per_second"] = 20.0
    first = run_queue_experiment(config)
    second = run_queue_experiment(config)
    assert first == second
    rows = first["per_request"]
    assert {row["strategy"] for row in rows} == {
        "fixed-small",
        "quality-only",
        "queue-aware",
    }
    quality_rows = [row for row in rows if row["strategy"] == "quality-only"]
    assert {row["model_id"] for row in quality_rows} == {"small"}
    queue_rows = [row for row in rows if row["strategy"] == "queue-aware"]
    assert all(row["quality"] >= 0.8 for row in queue_rows)


def test_historical_records_supply_latency_and_correctness_without_online_claim() -> None:
    config = _config()
    config["models"][0]["concurrency_reference_ms"] = 6.0
    records = {
        "small": [
            {"sample_id": "a", "latency_ms": 4.0, "correct": True},
            {"sample_id": "b", "latency_ms": 6.0, "correct": False},
            {"sample_id": "c", "latency_ms": 8.0, "correct": True},
        ],
        "large": [
            {"sample_id": "a", "latency_ms": 40.0, "correct": True},
            {"sample_id": "b", "latency_ms": 60.0, "correct": True},
            {"sample_id": "c", "latency_ms": 80.0, "correct": True},
        ],
    }
    records["large"] = list(reversed(records["large"]))
    report = run_queue_experiment(config, records)
    assert report["metadata"]["online_vllm_measurement"] is False
    profile = report["metadata"]["model_profiles"]["small"]
    assert profile["source"] == "historical_phase7_jsonl_calibrated_by_phase9_concurrency"
    assert profile["mean_service_time_ms"] == pytest.approx(6.0)
    assert profile["historical_observation_scale"] == pytest.approx(1.0)
    rows = [row for row in report["per_request"] if row["strategy"] == "fixed-small"]
    assert {row["input_source"] for row in rows} == {"historical"}
    assert {row["compute_ms"] for row in rows} <= {4.0, 6.0, 8.0}


def test_missing_quality_target_candidate_is_rejected() -> None:
    config = _config()
    config["queueing"]["slo"]["min_quality"] = 0.99
    with pytest.raises(ValueError, match="no model/node candidate"):
        run_queue_experiment(config)
