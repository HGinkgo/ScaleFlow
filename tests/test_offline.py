from hashlib import sha256

import pytest

from scaleflow.offline import (
    ALWAYS_ACCEPT,
    ALWAYS_ESCALATE,
    analyze_confidence,
    analyze_confidence_models,
    area_under_roc,
    average_precision,
    build_threshold_candidates,
    random_acceptance_baseline,
    replay_cascade,
    search_pareto_thresholds,
    split_sample_ids,
)


def _record(
    index: int,
    *,
    model_id: str = "model-small",
    confidence: float,
    correct: bool,
) -> dict[str, object]:
    outcome = "correct" if correct else "incorrect"
    return {
        "sample_id": f"sample-{index:03d}",
        "dataset_index": index,
        "question": f"question {index}",
        "prompt": f"prompt {index}",
        "reference_answer": str(index),
        "model_id": model_id,
        "outcome": outcome,
        "correct": correct,
        "success": True,
        "confidence": confidence,
        "latency_ms": 10.0,
        "experiment_config": {"dataset": "fixture"},
    }


def _failed_record(
    index: int,
    *,
    model_id: str,
    outcome: str,
    confidence: float | None,
    latency_ms: float,
) -> dict[str, object]:
    record = _record(
        index,
        model_id=model_id,
        confidence=confidence if confidence is not None else 0.0,
        correct=False,
    )
    record["outcome"] = outcome
    record["success"] = outcome != "inference_failure"
    record["confidence"] = confidence
    record["latency_ms"] = latency_ms
    return record


def test_split_sample_ids_uses_seeded_hash_without_labels() -> None:
    sample_ids = [f"sample-{index:03d}" for index in range(10)]
    expected = sorted(
        sample_ids,
        key=lambda sample_id: sha256(f"42:{sample_id}".encode()).hexdigest(),
    )

    development, evaluation = split_sample_ids(
        sample_ids,
        development_count=6,
        seed=42,
    )

    assert development == expected[:6]
    assert evaluation == expected[6:]
    assert set(development).isdisjoint(evaluation)


def test_area_under_roc_handles_ties() -> None:
    assert area_under_roc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert area_under_roc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_error_average_precision_uses_low_confidence_direction() -> None:
    confidence = [0.9, 0.8, 0.2, 0.1]
    not_correct = [False, False, True, True]

    assert average_precision([-value for value in confidence], not_correct) == 1.0


def test_confidence_gate_passes_separable_development_records() -> None:
    records = [
        _record(
            index,
            confidence=0.1 + index / 1000,
            correct=False,
        )
        for index in range(20)
    ] + [
        _record(
            index,
            confidence=0.8 + index / 1000,
            correct=True,
        )
        for index in range(20, 100)
    ]

    analysis = analyze_confidence(
        records,
        low_confidence_fraction=0.2,
        bootstrap_iterations=200,
        bootstrap_seed=42,
    )

    assert analysis["passed"] is True
    assert analysis["auroc"] == 1.0
    assert analysis["error_auprc"] == pytest.approx(1.0)
    assert analysis["point_biserial_correlation"] > 0
    assert len(analysis["confidence_deciles"]) == 10
    assert analysis["confidence_by_correctness"]["correct"]["count"] == 80
    assert analysis["confidence_by_correctness"]["not_correct"]["count"] == 20
    assert analysis["gate_checks"] == {
        "auroc_ci_lower_above_chance": True,
        "low_confidence_error_lift_ci_lower_above_zero": True,
        "aurc_improvement_ci_lower_above_zero": True,
    }


def test_confidence_gate_rejects_constant_scores() -> None:
    records = [
        _record(index, confidence=0.5, correct=index % 2 == 0)
        for index in range(100)
    ]

    analysis = analyze_confidence(
        records,
        low_confidence_fraction=0.2,
        bootstrap_iterations=100,
        bootstrap_seed=42,
    )

    assert analysis["passed"] is False
    assert analysis["auroc"] == pytest.approx(0.5)
    assert analysis["gate_checks"]["auroc_ci_lower_above_chance"] is False


def test_confidence_gate_marks_single_class_as_not_evaluable() -> None:
    records = [
        _record(index, confidence=0.9, correct=True)
        for index in range(20)
    ]

    analysis = analyze_confidence(
        records,
        low_confidence_fraction=0.2,
        bootstrap_iterations=20,
        bootstrap_seed=42,
    )

    assert analysis["passed"] is False
    assert analysis["failure_reason"] == "requires_both_correctness_classes"
    assert analysis["auroc"] is None
    assert all(check is False for check in analysis["gate_checks"].values())


def test_aurc_is_invariant_to_order_within_confidence_ties() -> None:
    records = [
        _record(0, confidence=0.9, correct=True),
        _record(1, confidence=0.5, correct=True),
        _record(2, confidence=0.5, correct=False),
        _record(3, confidence=0.1, correct=False),
    ]
    reordered = [records[0], records[2], records[1], records[3]]

    first = analyze_confidence(
        records,
        low_confidence_fraction=0.25,
        bootstrap_iterations=20,
        bootstrap_seed=42,
    )
    second = analyze_confidence(
        reordered,
        low_confidence_fraction=0.25,
        bootstrap_iterations=20,
        bootstrap_seed=42,
    )

    assert first["aurc"] == second["aurc"]


def test_multi_model_confidence_report_uses_only_development_split() -> None:
    records_by_model = []
    for model_index in range(3):
        records_by_model.append(
            [
                _record(
                    index,
                    model_id=f"model-{model_index}",
                    confidence=(0.1 if index < 4 else 0.9),
                    correct=index >= 4,
                )
                for index in range(20)
            ]
        )

    report = analyze_confidence_models(
        records_by_model,
        development_count=10,
        split_seed=42,
        low_confidence_fraction=0.2,
        bootstrap_iterations=100,
        bootstrap_seed=42,
    )

    assert report["split"]["development_count"] == 10
    assert report["split"]["evaluation_count"] == 10
    assert report["scope"] == "development_confidence_gate"
    assert all(model["request_count"] == 10 for model in report["models"])
    assert report["intermediate_model_ids"] == ["model-0", "model-1"]
    assert report["terminal_model_id"] == "model-2"
    assert report["evaluation_outcomes_read"] is False


def test_multi_model_confidence_report_rejects_common_config_mismatch() -> None:
    first = [
        _record(index, model_id="model-a", confidence=0.9, correct=index % 2 == 0)
        for index in range(10)
    ]
    second = [
        _record(index, model_id="model-b", confidence=0.9, correct=index % 2 == 0)
        for index in range(10)
    ]
    second[0]["experiment_config"] = {"dataset": "different"}

    with pytest.raises(ValueError, match="experiment_config mismatch"):
        analyze_confidence_models(
            [first, second],
            development_count=6,
            split_seed=42,
            low_confidence_fraction=0.2,
            bootstrap_iterations=10,
            bootstrap_seed=42,
        )


def test_multi_model_gate_removes_non_evaluable_intermediate_without_stopping() -> None:
    always_correct = [
        _record(index, model_id="always-correct", confidence=0.9, correct=True)
        for index in range(100)
    ]
    separable = [
        _record(
            index,
            model_id="separable",
            confidence=0.1 if index < 20 else 0.9,
            correct=index >= 20,
        )
        for index in range(100)
    ]
    terminal = [
        _record(index, model_id="terminal", confidence=0.8, correct=True)
        for index in range(100)
    ]

    report = analyze_confidence_models(
        [always_correct, separable, terminal],
        development_count=60,
        split_seed=42,
        low_confidence_fraction=0.2,
        bootstrap_iterations=100,
        bootstrap_seed=42,
    )

    assert report["failed_intermediate_model_ids"] == ["always-correct"]
    assert report["passed_intermediate_model_ids"] == ["separable"]
    assert report["all_intermediate_models_failed"] is False


def test_replay_cascade_escalates_low_confidence_and_operational_failures() -> None:
    small = [
        {**_record(0, model_id="small", confidence=0.9, correct=True), "latency_ms": 10.0},
        {**_record(1, model_id="small", confidence=0.1, correct=False), "latency_ms": 10.0},
        _failed_record(
            2,
            model_id="small",
            outcome="parse_failure",
            confidence=0.99,
            latency_ms=10.0,
        ),
    ]
    medium = [
        {**_record(0, model_id="medium", confidence=0.9, correct=True), "latency_ms": 20.0},
        {**_record(1, model_id="medium", confidence=0.9, correct=True), "latency_ms": 20.0},
        _failed_record(
            2,
            model_id="medium",
            outcome="inference_failure",
            confidence=None,
            latency_ms=20.0,
        ),
    ]
    terminal = [
        {**_record(index, model_id="terminal", confidence=0.8, correct=True), "latency_ms": 50.0}
        for index in range(3)
    ]

    report = replay_cascade(
        [small, medium, terminal],
        ["sample-000", "sample-001", "sample-002"],
        [0.5, 0.5],
    )

    assert [item["final_model_id"] for item in report["per_request"]] == [
        "small",
        "medium",
        "terminal",
    ]
    assert [item["cumulative_latency_ms"] for item in report["per_request"]] == [
        10.0,
        30.0,
        80.0,
    ]
    assert report["model_invocations"] == [3, 2, 1]
    assert report["accuracy"] == 1.0


def test_threshold_candidates_include_both_boundary_modes() -> None:
    candidates = build_threshold_candidates(
        [0.1, 0.4, 0.9],
        quantile_step=0.5,
    )

    assert candidates[0] == ALWAYS_ACCEPT
    assert candidates[-1] == ALWAYS_ESCALATE
    assert 0.4 in candidates


def test_pareto_search_selects_lowest_latency_quality_preserving_threshold() -> None:
    small = [
        {
            **_record(
                index,
                model_id="small",
                confidence=0.9 if index < 3 else 0.1,
                correct=index < 3,
            ),
            "latency_ms": 10.0,
        }
        for index in range(4)
    ]
    terminal = [
        {**_record(index, model_id="terminal", confidence=0.8, correct=True), "latency_ms": 50.0}
        for index in range(4)
    ]

    search = search_pareto_thresholds(
        [small, terminal],
        [f"sample-{index:03d}" for index in range(4)],
        quantile_step=0.5,
    )

    assert search["target_correct_count"] == 4
    assert search["selected"]["correct_count"] == 4
    assert search["selected"]["thresholds"] == [0.9]
    assert search["selected"]["mean_cumulative_latency_ms"] == 22.5
    assert search["selected"]["mean_cumulative_latency_ms"] < 50.0
    assert search["pareto_frontier"]


def test_random_baseline_matches_acceptance_rate_without_confidence() -> None:
    small = [
        _record(index, model_id="small", confidence=0.99, correct=index % 2 == 0)
        for index in range(100)
    ]
    terminal = [
        _record(index, model_id="terminal", confidence=0.01, correct=True)
        for index in range(100)
    ]

    baseline = random_acceptance_baseline(
        [small, terminal],
        [f"sample-{index:03d}" for index in range(100)],
        acceptance_rates=[0.75],
        seeds=range(1000, 1200),
    )

    assert baseline["seed_count"] == 200
    assert baseline["mean_stage_acceptance_rates"][0] == pytest.approx(0.75, abs=0.02)
    assert baseline["mean_terminal_invocation_rate"] == pytest.approx(0.25, abs=0.02)
    assert baseline["accuracy"]["lower"] <= baseline["accuracy"]["mean"]
    assert baseline["accuracy"]["mean"] <= baseline["accuracy"]["upper"]


def test_random_baseline_forces_invalid_confidence_to_terminal_model() -> None:
    small = [_record(0, model_id="small", confidence=0.9, correct=True)]
    small[0]["confidence"] = None
    terminal = [_record(0, model_id="terminal", confidence=0.8, correct=False)]

    baseline = random_acceptance_baseline(
        [small, terminal],
        ["sample-000"],
        acceptance_rates=[1.0],
        seeds=[42],
    )

    assert baseline["mean_terminal_invocation_rate"] == 1.0
    assert baseline["accuracy"]["mean"] == 0.0
