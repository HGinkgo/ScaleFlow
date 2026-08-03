"""CPU-only, pre-inference text routing analysis."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
import re
from random import Random
from statistics import fmean
from time import perf_counter_ns
from typing import Any

from scaleflow.baseline import percentile


_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_OPERATOR_PATTERN = re.compile(r"[+\-*/=×÷]")
_KEYWORD_PATTERNS = (
    re.compile(r"\b(add|added|sum|total|together|more|increase)\b"),
    re.compile(r"\b(subtract|difference|remain|remaining|less|decrease)\b"),
    re.compile(r"\b(multiply|multiplied|times|product|each|per)\b"),
    re.compile(r"\b(divide|divided|quotient|ratio|percent|percentage)\b"),
)


def verify_phase8_split(
    sample_ids: Sequence[str],
    split_report: dict[str, Any],
    *,
    seed: int,
    development_count: int,
) -> tuple[list[str], list[str]]:
    """Recompute and verify the exact split persisted by Phase 8."""

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Phase 8 split input has duplicate sample_id values")
    split = split_report.get("split")
    if not isinstance(split, dict):
        raise ValueError("Phase 8 split report has no split object")
    if split.get("method") != "sha256_seed_sample_id":
        raise ValueError("Phase 8 split method does not match")
    if split.get("seed") != seed:
        raise ValueError("Phase 8 split seed does not match")
    if split.get("development_count") != development_count:
        raise ValueError("Phase 8 split development count does not match")
    evaluation_count = len(sample_ids) - development_count
    if split.get("evaluation_count") != evaluation_count:
        raise ValueError("Phase 8 split evaluation count does not match")

    ordered = sorted(
        sample_ids,
        key=lambda sample_id: sha256(f"{seed}:{sample_id}".encode()).hexdigest(),
    )
    development = ordered[:development_count]
    evaluation = ordered[development_count:]
    if split.get("development_sample_ids") != development:
        raise ValueError("Phase 8 split development sample IDs do not match")
    if split.get("evaluation_sample_ids") != evaluation:
        raise ValueError("Phase 8 split evaluation sample IDs do not match")
    return development, evaluation


def extract_request_features(question: str) -> dict[str, float]:
    """Extract only request-text features available before model execution."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    lowercase = question.lower()
    return {
        "char_count": float(len(question)),
        "word_count": float(len(_WORD_PATTERN.findall(question))),
        "number_count": float(len(_NUMBER_PATTERN.findall(question))),
        "operator_count": float(len(_OPERATOR_PATTERN.findall(question))),
        "keyword_count": float(
            sum(bool(pattern.search(lowercase)) for pattern in _KEYWORD_PATTERNS)
        ),
    }


def build_routing_examples(
    records_by_model: Mapping[str, Sequence[dict[str, Any]]],
    model_order: Sequence[str],
    development_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Construct labels while retaining every model's historical outcome."""

    if list(model_order) != ["2B", "4B", "9B"]:
        raise ValueError("routing model order must be 2B, 4B, 9B")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    expected_ids: set[str] | None = None
    for label in model_order:
        records = records_by_model.get(label)
        if records is None or not records:
            raise ValueError(f"missing records for routing model {label}")
        model_index: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in model_index:
                raise ValueError(f"invalid or duplicate sample_id for model {label}")
            if not isinstance(record.get("question"), str):
                raise ValueError(f"missing question for model {label}, {sample_id}")
            if not isinstance(record.get("correct"), bool):
                raise ValueError(f"missing correctness for model {label}, {sample_id}")
            model_index[sample_id] = record
        ids = set(model_index)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("routing model sample_id sets do not match")
        indexed[label] = model_index

    assert expected_ids is not None
    if set(development_ids) - expected_ids:
        raise ValueError("development IDs are absent from routing records")

    examples: list[dict[str, Any]] = []
    for sample_id in development_ids:
        records = {label: indexed[label][sample_id] for label in model_order}
        correctness = {
            label: bool(records[label]["correct"]) for label in model_order
        }
        outcomes = {
            label: records[label].get("outcome") for label in model_order
        }
        chosen_label = next(
            (label for label in model_order if correctness[label]),
            model_order[-1],
        )
        examples.append(
            {
                "sample_id": sample_id,
                "dataset_index": records[model_order[0]].get("dataset_index"),
                "question": records[model_order[0]]["question"],
                "prompt": records[model_order[0]].get("prompt"),
                "reference_answer": records[model_order[0]].get("reference_answer"),
                "label": chosen_label,
                "none_correct": not any(correctness.values()),
                "non_monotonic": any(
                    correctness[model_order[index]]
                    and not correctness[model_order[target]]
                    for index in range(len(model_order))
                    for target in range(index + 1, len(model_order))
                ),
                "correctness": correctness,
                "outcomes": outcomes,
                "records": records,
                "features": extract_request_features(records[model_order[0]]["question"]),
            }
        )
    return examples


def _score_features(
    features: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    return sum(float(features.get(name, 0.0)) * float(weight) for name, weight in weights.items())


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _macro_f1(actual: Sequence[str], predicted: Sequence[str]) -> float:
    labels = ("2B", "4B", "9B")
    values: list[float] = []
    for label in labels:
        true_positive = sum(
            actual_value == label and predicted_value == label
            for actual_value, predicted_value in zip(actual, predicted, strict=True)
        )
        false_positive = sum(
            actual_value != label and predicted_value == label
            for actual_value, predicted_value in zip(actual, predicted, strict=True)
        )
        false_negative = sum(
            actual_value == label and predicted_value != label
            for actual_value, predicted_value in zip(actual, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return fmean(values)


def _rule_candidate_metrics(
    examples: Sequence[dict[str, Any]],
    assignments: Sequence[str],
) -> dict[str, Any]:
    selected_correct = [
        bool(example["records"][assignment]["correct"])
        for example, assignment in zip(examples, assignments, strict=True)
    ]
    latencies = [
        float(example["records"][assignment]["latency_ms"])
        for example, assignment in zip(examples, assignments, strict=True)
    ]
    labels = [str(example["label"]) for example in examples]
    return {
        "accuracy": fmean(float(value) for value in selected_correct),
        "mean_latency_ms": fmean(latencies),
        "macro_f1": _macro_f1(labels, assignments),
        "call_ratios": {
            label: assignments.count(label) / len(assignments)
            for label in ("2B", "4B", "9B")
        },
    }


def fit_rule_router(
    examples: Sequence[dict[str, Any]],
    *,
    weights: Mapping[str, float],
    threshold_quantiles: Sequence[float],
) -> dict[str, Any]:
    """Fit interpretable score thresholds using development examples only."""

    if not examples:
        raise ValueError("rule router requires development examples")
    if not threshold_quantiles:
        raise ValueError("rule router requires threshold quantiles")
    scores = [
        _score_features(example["features"], weights) for example in examples
    ]
    threshold_values = sorted(
        {
            _quantile(scores, float(quantile))
            for quantile in threshold_quantiles
            if 0 < float(quantile) < 1
        }
    )
    if len(threshold_values) < 2:
        raise ValueError("rule threshold grid must yield two distinct values")

    candidate_rows: list[dict[str, Any]] = []
    for low_threshold in threshold_values:
        for high_threshold in threshold_values:
            if high_threshold <= low_threshold:
                continue
            assignments = [
                "2B"
                if score < low_threshold
                else "4B"
                if score < high_threshold
                else "9B"
                for score in scores
            ]
            metrics = _rule_candidate_metrics(examples, assignments)
            candidate_rows.append(
                {
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "assignments": assignments,
                    **metrics,
                }
            )
    if not candidate_rows:
        raise ValueError("rule threshold grid produced no candidate")

    terminal_accuracy = fmean(
        float(example["records"]["9B"]["correct"]) for example in examples
    )
    target_accuracy = terminal_accuracy - 0.01
    feasible = [
        row for row in candidate_rows if row["accuracy"] >= target_accuracy
    ]
    development_target_met = bool(feasible)
    pool = feasible or candidate_rows
    if development_target_met:
        selected = min(
            pool,
            key=lambda row: (
                row["mean_latency_ms"],
                -row["macro_f1"],
                row["call_ratios"]["9B"],
                row["low_threshold"],
                row["high_threshold"],
            ),
        )
    else:
        selected = min(
            pool,
            key=lambda row: (
                -row["accuracy"],
                row["mean_latency_ms"],
                -row["macro_f1"],
                row["call_ratios"]["9B"],
            ),
        )
    return {
        "kind": "manual_complexity_rule",
        "weights": dict(weights),
        "low_threshold": selected["low_threshold"],
        "high_threshold": selected["high_threshold"],
        "development_target_accuracy": target_accuracy,
        "development_target_met": development_target_met,
        "development_metrics": {
            key: value
            for key, value in selected.items()
            if key not in {"assignments"}
        },
        "fit_sample_ids": [example["sample_id"] for example in examples],
    }


def predict_rule(
    router: Mapping[str, Any],
    question: str,
    features: Mapping[str, float] | None = None,
) -> str:
    """Predict a model from question text and optional precomputed text features."""

    active_features = features or extract_request_features(question)
    score = _score_features(active_features, router["weights"])
    if score < float(router["low_threshold"]):
        return "2B"
    if score < float(router["high_threshold"]):
        return "4B"
    return "9B"


def fit_tfidf_router(
    examples: Sequence[dict[str, Any]],
    *,
    tfidf_config: Mapping[str, Any],
    logistic_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the fixed text classifier on development questions only."""

    if not examples:
        raise ValueError("TF-IDF router requires development examples")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for the Phase 10 TF-IDF router; "
            "install the analysis extra"
        ) from error

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=int(tfidf_config["min_df"]),
        max_features=int(tfidf_config["max_features"]),
        sublinear_tf=bool(tfidf_config["sublinear_tf"]),
        lowercase=bool(tfidf_config["lowercase"]),
    )
    classifier = LogisticRegression(
        C=float(logistic_config["C"]),
        class_weight=logistic_config["class_weight"],
        max_iter=int(logistic_config["max_iter"]),
        solver=str(logistic_config["solver"]),
        random_state=int(logistic_config["random_state"]),
    )
    questions = [str(example["question"]) for example in examples]
    labels = [str(example["label"]) for example in examples]
    matrix = vectorizer.fit_transform(questions)
    classifier.fit(matrix, labels)
    return {
        "kind": "tfidf_logistic_regression",
        "vectorizer": vectorizer,
        "classifier": classifier,
        "fit_sample_ids": [example["sample_id"] for example in examples],
        "classes": list(classifier.classes_),
    }


def predict_tfidf(router: Mapping[str, Any], question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    matrix = router["vectorizer"].transform([question])
    return str(router["classifier"].predict(matrix)[0])


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def _outcome(record: Mapping[str, Any]) -> str:
    value = record.get("outcome")
    if isinstance(value, str):
        return value
    return "correct" if record.get("correct") is True else "incorrect"


def _call_counts(assignments: Sequence[str]) -> dict[str, int]:
    return {
        label: sum(assignment == label for assignment in assignments)
        for label in ("2B", "4B", "9B")
    }


def classification_metrics(
    actual: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("classification inputs must have equal non-zero length")
    labels = ("2B", "4B", "9B")
    confusion = {
        actual_label: {
            predicted_label: sum(
                actual_value == actual_label and predicted_value == predicted_label
                for actual_value, predicted_value in zip(
                    actual,
                    predicted,
                    strict=True,
                )
            )
            for predicted_label in labels
        }
        for actual_label in labels
    }
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    return {
        "accuracy": sum(
            left == right for left, right in zip(actual, predicted, strict=True)
        )
        / len(actual),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": fmean(item["f1"] for item in per_class.values()),
    }


def evaluate_assignments(
    examples: Sequence[dict[str, Any]],
    assignments: Sequence[str],
    *,
    router_overhead_ms: Sequence[float],
) -> dict[str, Any]:
    """Evaluate one model assignment per request using historical results."""

    if len(examples) != len(assignments) or len(examples) != len(router_overhead_ms):
        raise ValueError("examples, assignments, and overhead must have equal length")
    if not examples:
        raise ValueError("cannot evaluate empty assignments")
    if any(assignment not in {"2B", "4B", "9B"} for assignment in assignments):
        raise ValueError("assignments must use 2B, 4B, or 9B")

    selected_records = [
        example["records"][assignment]
        for example, assignment in zip(examples, assignments, strict=True)
    ]
    outcomes = [_outcome(record) for record in selected_records]
    historical_latencies = [float(record["latency_ms"]) for record in selected_records]
    overhead = [float(value) for value in router_overhead_ms]
    effective_latencies = [
        latency + extra
        for latency, extra in zip(historical_latencies, overhead, strict=True)
    ]
    outcome_counts = {
        outcome: sum(value == outcome for value in outcomes)
        for outcome in (
            "correct",
            "incorrect",
            "parse_failure",
            "inference_failure",
        )
    }
    counts = _call_counts(assignments)
    return {
        "request_count": len(examples),
        "correct_count": outcome_counts["correct"],
        "accuracy": outcome_counts["correct"] / len(examples),
        "outcome_counts": outcome_counts,
        "none_correct_count": sum(
            bool(example.get("none_correct", False)) for example in examples
        ),
        "call_counts": counts,
        "call_ratios": {
            label: counts[label] / len(examples) for label in ("2B", "4B", "9B")
        },
        "latency_ms": {
            "historical": _distribution(historical_latencies),
            "with_router": _distribution(effective_latencies),
        },
        "router_overhead_ms": _distribution(overhead),
        "assignments": list(assignments),
        "sample_ids": [str(example["sample_id"]) for example in examples],
    }


def _interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "lower": percentile(values, 2.5),
        "upper": percentile(values, 97.5),
    }


def matched_random_baseline(
    predicted_models: Sequence[str],
    examples: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Shuffle a router's labels while preserving its exact model quotas."""

    if not seeds:
        raise ValueError("random baseline requires at least one seed")
    if len(predicted_models) != len(examples):
        raise ValueError("predictions and examples must have equal length")
    expected_counts = _call_counts(predicted_models)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        shuffled = list(predicted_models)
        randomizer = Random(int(seed))
        randomizer.shuffle(shuffled)
        metrics = evaluate_assignments(
            examples,
            shuffled,
            router_overhead_ms=[0.0] * len(examples),
        )
        runs.append(
            {
                "seed": int(seed),
                "accuracy": metrics["accuracy"],
                "historical_latency_mean_ms": metrics["latency_ms"]["historical"]["mean"],
                "call_counts": metrics["call_counts"],
            }
        )
    accuracies = [run["accuracy"] for run in runs]
    latencies = [run["historical_latency_mean_ms"] for run in runs]
    return {
        "seed_count": len(runs),
        "call_counts": expected_counts,
        "call_ratios": {
            label: expected_counts[label] / len(predicted_models)
            for label in ("2B", "4B", "9B")
        },
        "accuracy": _interval(accuracies),
        "historical_latency_mean_ms": _interval(latencies),
        "runs": runs,
    }


def _timed_predictions(
    router: Mapping[str, Any],
    examples: Sequence[dict[str, Any]],
    *,
    kind: str,
) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    overhead_ms: list[float] = []
    for example in examples:
        started = perf_counter_ns()
        if kind == "rule":
            prediction = predict_rule(router, str(example["question"]))
        elif kind == "tfidf":
            prediction = predict_tfidf(router, str(example["question"]))
        else:
            raise ValueError(f"unsupported router kind: {kind}")
        overhead_ms.append((perf_counter_ns() - started) / 1_000_000)
        predictions.append(prediction)
    return predictions, overhead_ms


def _feature_means_by_label(
    examples: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label in ("2B", "4B", "9B"):
        selected = [example for example in examples if example["label"] == label]
        if not selected:
            result[label] = {}
            continue
        feature_names = selected[0]["features"].keys()
        result[label] = {
            name: fmean(float(example["features"][name]) for example in selected)
            for name in feature_names
        }
    return result


def _request_metadata(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep per-request labels and independent model outcomes traceable."""

    return [
        {
            "sample_id": str(example["sample_id"]),
            "label": str(example["label"]),
            "none_correct": bool(example["none_correct"]),
            "non_monotonic": bool(example["non_monotonic"]),
            "correctness": dict(example["correctness"]),
            "outcomes": dict(example["outcomes"]),
        }
        for example in examples
    ]


def _tfidf_top_terms(router: Mapping[str, Any], limit: int = 10) -> dict[str, list[str]]:
    vectorizer = router["vectorizer"]
    classifier = router["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    result: dict[str, list[str]] = {}
    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        top_indices = sorted(
            range(len(coefficients)),
            key=lambda index: float(coefficients[index]),
            reverse=True,
        )[:limit]
        result[str(label)] = [str(feature_names[index]) for index in top_indices]
    return result


def _attach_router_metrics(
    report: dict[str, Any],
    examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    actual = [str(example["label"]) for example in examples]
    predicted = [str(value) for value in report["assignments"]]
    report["routing_classification"] = classification_metrics(actual, predicted)
    return report


def _relative_to(
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {
        "accuracy_delta_percentage_points": (
            report["accuracy"] - baseline["accuracy"]
        )
        * 100,
        "historical_mean_latency_delta_ms": (
            report["latency_ms"]["historical"]["mean"]
            - baseline["latency_ms"]["historical"]["mean"]
        ),
        "with_router_mean_latency_delta_ms": (
            report["latency_ms"]["with_router"]["mean"]
            - baseline["latency_ms"]["with_router"]["mean"]
        ),
    }


def analyze_routing(
    config: Mapping[str, Any],
    records_by_model: Mapping[str, Sequence[dict[str, Any]]],
    split_report: dict[str, Any],
) -> dict[str, Any]:
    """Fit routers on Phase 8 development data and evaluate the frozen routers once."""

    model_order = [str(model["label"]) for model in config["models"]]
    sample_ids = [
        str(record["sample_id"])
        for record in records_by_model[model_order[0]]
    ]
    development_ids, evaluation_ids = verify_phase8_split(
        sample_ids,
        split_report,
        seed=int(config["phase8_split"]["seed"]),
        development_count=int(config["phase8_split"]["development_count"]),
    )
    development_examples = build_routing_examples(
        records_by_model,
        model_order,
        development_ids,
    )

    rule_started = perf_counter_ns()
    rule_router = fit_rule_router(
        development_examples,
        weights=config["rule"]["weights"],
        threshold_quantiles=config["rule"]["threshold_quantiles"],
    )
    rule_training_ms = (perf_counter_ns() - rule_started) / 1_000_000
    tfidf_started = perf_counter_ns()
    tfidf_router = fit_tfidf_router(
        development_examples,
        tfidf_config=config["tfidf"],
        logistic_config=config["logistic_regression"],
    )
    tfidf_training_ms = (perf_counter_ns() - tfidf_started) / 1_000_000

    development_rule_predictions, development_rule_overhead = _timed_predictions(
        rule_router,
        development_examples,
        kind="rule",
    )
    development_tfidf_predictions, development_tfidf_overhead = _timed_predictions(
        tfidf_router,
        development_examples,
        kind="tfidf",
    )
    development_reports = {
        "rule": _attach_router_metrics(
            evaluate_assignments(
                development_examples,
                development_rule_predictions,
                router_overhead_ms=development_rule_overhead,
            ),
            development_examples,
        ),
        "tfidf_logistic_regression": _attach_router_metrics(
            evaluate_assignments(
                development_examples,
                development_tfidf_predictions,
                router_overhead_ms=development_tfidf_overhead,
            ),
            development_examples,
        ),
    }

    # Evaluation examples are built only after both development-only fits and checks.
    evaluation_examples = build_routing_examples(
        records_by_model,
        model_order,
        evaluation_ids,
    )
    evaluation_rule_predictions, evaluation_rule_overhead = _timed_predictions(
        rule_router,
        evaluation_examples,
        kind="rule",
    )
    evaluation_tfidf_predictions, evaluation_tfidf_overhead = _timed_predictions(
        tfidf_router,
        evaluation_examples,
        kind="tfidf",
    )
    zero_overhead = [0.0] * len(evaluation_examples)
    method_assignments = {
        "always_2B": ["2B"] * len(evaluation_examples),
        "always_4B": ["4B"] * len(evaluation_examples),
        "always_9B": ["9B"] * len(evaluation_examples),
        "manual_rule": evaluation_rule_predictions,
        "tfidf_logistic_regression": evaluation_tfidf_predictions,
        "oracle_lowest_correct": [
            str(example["label"]) for example in evaluation_examples
        ],
    }
    method_overheads = {
        "always_2B": zero_overhead,
        "always_4B": zero_overhead,
        "always_9B": zero_overhead,
        "manual_rule": evaluation_rule_overhead,
        "tfidf_logistic_regression": evaluation_tfidf_overhead,
        "oracle_lowest_correct": zero_overhead,
    }
    evaluation_reports: dict[str, dict[str, Any]] = {}
    for name, assignments in method_assignments.items():
        report = evaluate_assignments(
            evaluation_examples,
            assignments,
            router_overhead_ms=method_overheads[name],
        )
        if name in {"manual_rule", "tfidf_logistic_regression"}:
            report = _attach_router_metrics(report, evaluation_examples)
        evaluation_reports[name] = report

    random_reports = {
        name: matched_random_baseline(
            method_assignments[name],
            evaluation_examples,
            seeds=range(
                int(config["random_baseline"]["seed_start"]),
                int(config["random_baseline"]["seed_start"])
                + int(config["random_baseline"]["seed_count"]),
            ),
        )
        for name in ("manual_rule", "tfidf_logistic_regression")
    }
    baseline_9b = evaluation_reports["always_9B"]
    for name, report in evaluation_reports.items():
        report["relative_to_always_9B"] = _relative_to(report, baseline_9b)
        if name in random_reports:
            random_report = random_reports[name]
            report["relative_to_matched_random"] = {
                "accuracy_delta_percentage_points": (
                    report["accuracy"] - random_report["accuracy"]["mean"]
                )
                * 100,
                "accuracy_delta_interval_percentage_points": {
                    "lower": (
                        report["accuracy"] - random_report["accuracy"]["upper"]
                    )
                    * 100,
                    "upper": (
                        report["accuracy"] - random_report["accuracy"]["lower"]
                    )
                    * 100,
                },
                "historical_mean_latency_delta_ms": (
                    report["latency_ms"]["historical"]["mean"]
                    - random_report["historical_latency_mean_ms"]["mean"]
                ),
            }

    router_summaries = {
        "manual_rule": {
            **{
                key: value
                for key, value in rule_router.items()
                if key != "development_metrics"
            },
            "training_time_ms": rule_training_ms,
            "development_metrics": development_reports["rule"],
            "feature_means_by_label_development": _feature_means_by_label(
                development_examples
            ),
        },
        "tfidf_logistic_regression": {
            "kind": tfidf_router["kind"],
            "classes": tfidf_router["classes"],
            "fit_sample_ids": tfidf_router["fit_sample_ids"],
            "training_time_ms": tfidf_training_ms,
            "development_metrics": development_reports[
                "tfidf_logistic_regression"
            ],
            "top_positive_terms": _tfidf_top_terms(tfidf_router),
        },
    }
    return {
        "scope": "phase10_exploratory_pre_inference_text_routing",
        "exploratory": True,
        "evaluation_reused_from_phase8": True,
        "phase8_split_verified": True,
        "model_order": model_order,
        "split": {
            "method": split_report["split"]["method"],
            "seed": split_report["split"]["seed"],
            "development_count": len(development_examples),
            "evaluation_count": len(evaluation_examples),
            "development_sample_ids": development_ids,
            "evaluation_sample_ids": evaluation_ids,
        },
        "router_input_contract": {
            "fields": [
                "question",
                "char_count",
                "word_count",
                "number_count",
                "operator_count",
                "keyword_count",
            ],
            "forbidden_fields": [
                "model_output",
                "confidence",
                "reference_answer",
                "latency_ms",
                "output_token_count",
                "correct",
            ],
        },
        "routers": router_summaries,
        "development": {
            "request_metadata": _request_metadata(development_examples),
        },
        "evaluation": {
            "request_metadata": _request_metadata(evaluation_examples),
            "methods": evaluation_reports,
            "matched_random_baselines": random_reports,
        },
    }
