from hashlib import sha256

import pytest

from scaleflow.routing import (
    build_routing_examples,
    classification_metrics,
    extract_request_features,
    fit_rule_router,
    fit_tfidf_router,
    predict_rule,
    predict_tfidf,
    verify_phase8_split,
)


def test_verify_phase8_split_matches_saved_hash_order() -> None:
    sample_ids = [f"gsm8k-test-{index:04d}" for index in range(8)]
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: sha256(f"42:{sample_id}".encode()).hexdigest(),
    )
    split_report = {
        "split": {
            "method": "sha256_seed_sample_id",
            "seed": 42,
            "development_count": 5,
            "evaluation_count": 3,
            "development_sample_ids": ordered[:5],
            "evaluation_sample_ids": ordered[5:],
        }
    }

    development, evaluation = verify_phase8_split(
        sample_ids,
        split_report,
        seed=42,
        development_count=5,
    )

    assert development == ordered[:5]
    assert evaluation == ordered[5:]

    changed = dict(split_report)
    changed["split"] = dict(split_report["split"])
    changed["split"]["development_sample_ids"] = ["wrong"] + ordered[1:5]
    with pytest.raises(ValueError, match="Phase 8 split"):
        verify_phase8_split(sample_ids, changed, seed=42, development_count=5)


def test_request_features_are_deterministic_and_question_only() -> None:
    question = "A box has 12 items, receives 4 more, then computes 2 * 3."
    features = extract_request_features(question)

    assert features == extract_request_features(question)
    assert features["char_count"] == len(question)
    assert features["number_count"] == 4
    assert features["operator_count"] >= 1
    assert features["keyword_count"] >= 1
    with pytest.raises(TypeError):
        extract_request_features(question, {"confidence": 0.9})  # type: ignore[call-arg]


def _model_records(
    model_label: str,
    correctness: list[bool],
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "dataset_index": index,
            "question": f"question {index}",
            "prompt": f"prompt {index}",
            "reference_answer": "4",
            "model_id": model_label,
            "correct": correct,
            "outcome": "correct" if correct else "incorrect",
            "success": True,
            "latency_ms": float(index + 1),
            "experiment_config": {"fingerprint": "fixed"},
        }
        for index, correct in enumerate(correctness)
    ]


def test_routing_labels_keep_non_monotonic_and_none_correct_samples() -> None:
    records_by_model = {
        "2B": _model_records("2B", [True, False, False, True, False]),
        "4B": _model_records("4B", [False, True, False, False, False]),
        "9B": _model_records("9B", [False, False, True, False, False]),
    }

    examples = build_routing_examples(
        records_by_model,
        ["2B", "4B", "9B"],
        [f"sample-{index}" for index in range(5)],
    )

    assert [example["label"] for example in examples] == [
        "2B",
        "4B",
        "9B",
        "2B",
        "9B",
    ]
    assert [example["none_correct"] for example in examples] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert examples[3]["non_monotonic"] is True
    assert all(
        set(example["correctness"].keys()) == {"2B", "4B", "9B"}
        for example in examples
    )


def _router_examples() -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, label in enumerate(["2B"] * 4 + ["4B"] * 3 + ["9B"] * 2):
        features = {
            "char_count": float(20 + index * 20),
            "word_count": float(4 + index * 4),
            "number_count": float(1 + index),
            "operator_count": float(index // 2),
            "keyword_count": float(index // 3),
        }
        examples.append(
            {
                "sample_id": f"router-{index}",
                "question": f"question about {label} complexity {index}",
                "label": label,
                "features": features,
                "records": {
                    "2B": {"latency_ms": 10.0, "correct": label == "2B"},
                    "4B": {"latency_ms": 20.0, "correct": label != "2B"},
                    "9B": {"latency_ms": 30.0, "correct": label == "9B"},
                },
            }
        )
    return examples


def test_rule_router_uses_only_development_examples_and_is_deterministic() -> None:
    examples = _router_examples()
    weights = {
        "char_count": 0.01,
        "word_count": 1.0,
        "number_count": 4.0,
        "operator_count": 2.0,
        "keyword_count": 3.0,
    }

    router = fit_rule_router(
        examples,
        weights=weights,
        threshold_quantiles=[0.50, 0.60, 0.70, 0.80, 0.90],
    )

    assert router["fit_sample_ids"] == [example["sample_id"] for example in examples]
    assert router["development_target_met"] is True
    assert predict_rule(router, examples[0]["question"], examples[0]["features"]) in {
        "2B",
        "4B",
        "9B",
    }
    assert predict_rule(
        router,
        examples[0]["question"],
        examples[0]["features"],
    ) == predict_rule(
        router,
        examples[0]["question"],
        examples[0]["features"],
    )


def test_assignment_metrics_report_history_and_router_overhead_separately() -> None:
    from scaleflow.routing import evaluate_assignments

    examples = _router_examples()
    assignments = ["2B", "2B", "4B", "4B", "9B", "9B", "9B", "9B", "9B"]
    metrics = evaluate_assignments(
        examples,
        assignments,
        router_overhead_ms=[1.0] * len(examples),
    )

    assert metrics["request_count"] == len(examples)
    assert metrics["call_counts"] == {"2B": 2, "4B": 2, "9B": 5}
    assert metrics["call_ratios"] == pytest.approx(
        {"2B": 2 / 9, "4B": 2 / 9, "9B": 5 / 9}
    )
    assert metrics["latency_ms"]["historical"]["mean"] < metrics["latency_ms"]["with_router"]["mean"]
    assert metrics["router_overhead_ms"]["mean"] == pytest.approx(1.0)
    assert set(metrics["outcome_counts"]) == {
        "correct",
        "incorrect",
        "parse_failure",
        "inference_failure",
    }
    assert metrics["none_correct_count"] == 0


def test_matched_random_baseline_preserves_router_call_counts() -> None:
    from scaleflow.routing import matched_random_baseline

    examples = _router_examples()
    assignments = ["2B", "2B", "4B", "4B", "9B", "9B", "9B", "9B", "9B"]
    baseline = matched_random_baseline(
        assignments,
        examples,
        seeds=[1000, 1001, 1002],
    )

    assert baseline["seed_count"] == 3
    assert baseline["call_counts"] == {"2B": 2, "4B": 2, "9B": 5}
    assert all(
        run["call_counts"] == baseline["call_counts"]
        for run in baseline["runs"]
    )
    assert baseline["accuracy"]["lower"] <= baseline["accuracy"]["mean"]
    assert baseline["accuracy"]["mean"] <= baseline["accuracy"]["upper"]


def test_classification_metrics_preserve_three_class_confusion_counts() -> None:
    metrics = classification_metrics(
        ["2B", "4B", "9B", "9B"],
        ["2B", "9B", "9B", "4B"],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {
        "2B": {"2B": 1, "4B": 0, "9B": 0},
        "4B": {"2B": 0, "4B": 0, "9B": 1},
        "9B": {"2B": 0, "4B": 1, "9B": 1},
    }


def test_tfidf_router_has_fixed_three_class_predictions() -> None:
    examples = _router_examples()
    router = fit_tfidf_router(
        examples,
        tfidf_config={
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_features": 100,
            "sublinear_tf": True,
            "lowercase": True,
        },
        logistic_config={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        },
    )

    predictions = [
        predict_tfidf(router, example["question"])
        for example in examples
    ]
    assert set(predictions) <= {"2B", "4B", "9B"}
    assert predictions == [
        predict_tfidf(router, example["question"])
        for example in examples
    ]
    assert router["fit_sample_ids"] == [example["sample_id"] for example in examples]
