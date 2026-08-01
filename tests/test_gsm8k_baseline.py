from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
import json
from pathlib import Path

import pytest

from scaleflow.backends.base import Backend
from scaleflow.schemas import InferenceRequest, ModelResponse


baseline = import_module("scaleflow.baseline")


def require_function(name: str):
    function = getattr(baseline, name, None)
    assert callable(function), f"missing baseline function: {name}"
    return function


def test_gsm8k_baseline_module_exists() -> None:
    assert find_spec("scaleflow.baseline") is not None


def test_dataset_is_downloaded_once_and_verified_by_sha256(tmp_path: Path) -> None:
    ensure_dataset = require_function("ensure_dataset")
    source = tmp_path / "source.jsonl"
    target = tmp_path / "data" / "test.jsonl"
    content = b'{"question":"2 + 2?","answer":"#### 4"}\n'
    source.write_bytes(content)
    expected_sha256 = sha256(content).hexdigest()

    returned = ensure_dataset(target, source.as_uri(), expected_sha256)

    assert returned == target
    assert target.read_bytes() == content
    assert ensure_dataset(target, source.as_uri(), expected_sha256) == target

    target.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        ensure_dataset(target, source.as_uri(), expected_sha256)


def test_load_and_select_samples_preserves_explicit_index_order(
    tmp_path: Path,
) -> None:
    load_gsm8k_jsonl = require_function("load_gsm8k_jsonl")
    select_samples = require_function("select_samples")
    dataset = tmp_path / "test.jsonl"
    records = [
        {"question": f"question {index}", "answer": f"reason\n#### {index}"}
        for index in range(5)
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_gsm8k_jsonl(dataset)
    selected = select_samples(loaded, [4, 1])

    assert [item["dataset_index"] for item in selected] == [4, 1]
    assert [item["sample_id"] for item in selected] == [
        "gsm8k-test-0004",
        "gsm8k-test-0001",
    ]
    assert [item["question"] for item in selected] == ["question 4", "question 1"]
    with pytest.raises(ValueError, match="unique"):
        select_samples(loaded, [1, 1])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning\n#### 42", "42"),
        ("Reasoning\n#### 1,234.50\n", "1234.5"),
        ("#### -0.25", "-0.25"),
    ],
)
def test_parse_final_answer_accepts_strict_numeric_marker(
    text: str,
    expected: str,
) -> None:
    parse_final_answer = require_function("parse_final_answer")

    assert parse_final_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 42",
        "#### 42 apples",
        "#### $42",
        "#### 1/2",
    ],
)
def test_parse_final_answer_rejects_non_contract_output(text: str) -> None:
    parse_final_answer = require_function("parse_final_answer")

    assert parse_final_answer(text) is None


def test_score_output_separates_parse_failure_from_incorrect_answer() -> None:
    score_output = require_function("score_output")

    correct = score_output("work\n#### 72", "reference\n#### 72")
    incorrect = score_output("work\n#### 71", "reference\n#### 72")
    parse_failure = score_output("The answer is 72", "reference\n#### 72")

    assert correct == {
        "reference_answer": "72",
        "predicted_answer": "72",
        "correct": True,
        "parse_failure": False,
        "outcome": "correct",
    }
    assert incorrect["correct"] is False
    assert incorrect["parse_failure"] is False
    assert incorrect["outcome"] == "incorrect"
    assert parse_failure["correct"] is False
    assert parse_failure["parse_failure"] is True
    assert parse_failure["predicted_answer"] is None
    assert parse_failure["outcome"] == "parse_failure"


def test_percentile_uses_linear_interpolation() -> None:
    percentile = require_function("percentile")

    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_point_biserial_reports_direction_and_handles_zero_variance() -> None:
    point_biserial_correlation = require_function("point_biserial_correlation")

    correlation = point_biserial_correlation(
        [0.9, 0.8, 0.2, 0.1],
        [True, True, False, False],
    )

    assert correlation is not None
    assert correlation > 0.9
    assert point_biserial_correlation([0.5, 0.5], [True, False]) is None
    assert point_biserial_correlation([0.5, 0.6], [True, True]) is None


def test_render_prompt_requires_question_placeholder() -> None:
    render_prompt = require_function("render_prompt")

    assert render_prompt("Problem:\n{question}\n", "2 + 2?") == "Problem:\n2 + 2?\n"
    with pytest.raises(ValueError, match="question"):
        render_prompt("No placeholder", "2 + 2?")


class StaticBackend(Backend):
    def __init__(self, responses: dict[str, ModelResponse]) -> None:
        self.responses = responses

    def generate(self, request: InferenceRequest) -> ModelResponse:
        return self.responses[request.request_id]

    def get_model_info(self) -> dict[str, object]:
        return {"backend": "static", "model_id": "model-a"}

    def health_check(self) -> bool:
        return True

    def estimate_latency(self, request: InferenceRequest) -> float:
        return self.responses[request.request_id].latency_ms


def make_response(
    text: str,
    *,
    confidence: float,
    latency_ms: float,
    success: bool = True,
    error: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        model_id="model-a",
        text=text,
        confidence=confidence,
        latency_ms=latency_ms,
        success=success,
        error=error,
        token_logprobs=[-0.1, -0.2] if success else [],
        confidence_method="exp(mean(output_token_logprobs))" if success else None,
        gpu_memory_used_mb=4096.0 if success else None,
    )


def test_run_baseline_records_measurements_and_distinct_outcomes() -> None:
    run_baseline_samples = require_function("run_baseline_samples")
    samples = [
        {
            "sample_id": "gsm8k-test-0001",
            "dataset_index": 1,
            "question": "one",
            "reference_solution": "work\n#### 10",
        },
        {
            "sample_id": "gsm8k-test-0002",
            "dataset_index": 2,
            "question": "two",
            "reference_solution": "work\n#### 20",
        },
        {
            "sample_id": "gsm8k-test-0003",
            "dataset_index": 3,
            "question": "three",
            "reference_solution": "work\n#### 30",
        },
    ]
    backend = StaticBackend(
        {
            "gsm8k-test-0001": make_response(
                "reason\n#### 10", confidence=0.9, latency_ms=10.0
            ),
            "gsm8k-test-0002": make_response(
                "answer 20", confidence=0.4, latency_ms=20.0
            ),
            "gsm8k-test-0003": make_response(
                "",
                confidence=0.0,
                latency_ms=5.0,
                success=False,
                error="failed",
            ),
        }
    )

    records = run_baseline_samples(
        samples,
        backend,
        "Problem:\n{question}\n",
        max_tokens=2,
    )

    assert [record["outcome"] for record in records] == [
        "correct",
        "parse_failure",
        "inference_failure",
    ]
    assert records[0]["output_token_count"] == 2
    assert records[0]["tokens_per_second"] == pytest.approx(200.0)
    assert records[0]["hit_max_tokens"] is True
    assert records[0]["token_logprobs"] == [-0.1, -0.2]
    assert records[0]["confidence"] == 0.9
    assert records[1]["parse_failure"] is True
    assert records[1]["correct"] is False
    assert records[2]["parse_failure"] is False
    assert records[2]["error"] == "failed"


def test_warmup_is_recorded_but_fails_closed() -> None:
    warmup_backend = require_function("warmup_backend")
    successful = StaticBackend(
        {
            "warmup-01": make_response("#### 2", confidence=0.8, latency_ms=3.0),
            "warmup-02": make_response("#### 4", confidence=0.8, latency_ms=4.0),
        }
    )

    warmup = warmup_backend(successful, ["1 + 1", "2 + 2"])

    assert warmup == {
        "request_count": 2,
        "successful_count": 2,
        "total_latency_ms": 7.0,
    }

    failing = StaticBackend(
        {
            "warmup-01": make_response(
                "",
                confidence=0.0,
                latency_ms=1.0,
                success=False,
                error="warmup failed",
            )
        }
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup_backend(failing, ["1 + 1"])


def test_summary_includes_parse_failures_latency_throughput_and_correlation_note() -> None:
    summarize_baseline = require_function("summarize_baseline")
    records = [
        {
            "success": True,
            "correct": True,
            "parse_failure": False,
            "outcome": "correct",
            "latency_ms": 10.0,
            "output_token_count": 2,
            "tokens_per_second": 200.0,
            "hit_max_tokens": False,
            "confidence": 0.9,
            "gpu_memory_used_mb": 4000.0,
        },
        {
            "success": True,
            "correct": False,
            "parse_failure": False,
            "outcome": "incorrect",
            "latency_ms": 20.0,
            "output_token_count": 4,
            "tokens_per_second": 200.0,
            "hit_max_tokens": False,
            "confidence": 0.4,
            "gpu_memory_used_mb": 4100.0,
        },
        {
            "success": True,
            "correct": False,
            "parse_failure": True,
            "outcome": "parse_failure",
            "latency_ms": 30.0,
            "output_token_count": 3,
            "tokens_per_second": 100.0,
            "hit_max_tokens": True,
            "confidence": 0.2,
            "gpu_memory_used_mb": 4200.0,
        },
    ]

    summary = summarize_baseline(records)

    assert summary["request_count"] == 3
    assert summary["correct_count"] == 1
    assert summary["incorrect_count"] == 1
    assert summary["parse_failure_count"] == 1
    assert summary["max_token_hit_count"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["parse_failure_rate"] == pytest.approx(1 / 3)
    assert summary["latency_ms"] == {
        "mean": 20.0,
        "p50": 20.0,
        "p95": 29.0,
    }
    assert summary["tokens_per_second"]["mean"] == pytest.approx(500 / 3)
    assert summary["tokens_per_second"]["aggregate"] == pytest.approx(150.0)
    assert summary["gpu_memory_used_mb"]["peak"] == 4200.0
    assert summary["confidence_analysis"]["point_biserial_correlation"] is not None
    assert summary["confidence_analysis"][
        "point_biserial_correlation_parsed_only"
    ] == pytest.approx(1.0)
    assert summary["confidence_analysis"]["mean_confidence_correct"] == 0.9
    assert summary["confidence_analysis"]["mean_confidence_incorrect"] == 0.4
    assert summary["confidence_analysis"]["mean_confidence_parse_failure"] == 0.2
    assert summary["confidence_analysis"]["scope"] == "preliminary_observation"
    assert "not a formal statistical conclusion" in summary["confidence_analysis"][
        "note"
    ]


def test_baseline_json_and_jsonl_can_be_read_back(tmp_path: Path) -> None:
    write_records_jsonl = require_function("write_records_jsonl")
    write_summary_json = require_function("write_summary_json")
    records = [{"sample_id": "gsm8k-test-0001", "correct": True}]
    summary = {"request_count": 1, "accuracy": 1.0}
    records_path = tmp_path / "nested" / "records.jsonl"
    summary_path = tmp_path / "nested" / "summary.json"

    write_records_jsonl(records_path, records)
    write_summary_json(summary_path, summary)

    assert json.loads(records_path.read_text(encoding="utf-8")) == records[0]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def comparison_record(
    sample_id: str,
    model_id: str,
    outcome: str,
    *,
    reference_answer: str = "1",
    experiment_config: dict[str, object] | None = None,
) -> dict[str, object]:
    if experiment_config is None:
        experiment_config = {
            "project_seed": 42,
            "dataset": {"commit": "fixed-commit", "sample_indices": list(range(8))},
            "prompt_template": "fixed prompt",
            "warmup_prompts": ["warmup"],
            "generation_config": {"temperature": 0.0, "max_tokens": 384},
            "backend_common_config": {"dtype": "bfloat16"},
        }
    return {
        "sample_id": sample_id,
        "dataset_index": int(sample_id.rsplit("-", maxsplit=1)[-1]),
        "question": f"question for {sample_id}",
        "prompt": f"prompt for {sample_id}",
        "model_id": model_id,
        "reference_answer": reference_answer,
        "outcome": outcome,
        "correct": outcome == "correct",
        "experiment_config": experiment_config,
    }


def test_load_records_jsonl_requires_json_objects(tmp_path: Path) -> None:
    load_records_jsonl = require_function("load_records_jsonl")
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        '{"sample_id":"sample-1","correct":true}\n',
        encoding="utf-8",
    )

    assert load_records_jsonl(records_path) == [
        {"sample_id": "sample-1", "correct": True}
    ]

    records_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_records_jsonl(records_path)


def test_compare_baselines_aligns_by_id_and_reports_rescue_and_oracle() -> None:
    compare_baseline_records = require_function("compare_baseline_records")
    baseline_records = [
        comparison_record("sample-1", "model-0.8b", "correct"),
        comparison_record("sample-2", "model-0.8b", "correct"),
        comparison_record("sample-3", "model-0.8b", "incorrect"),
        comparison_record("sample-4", "model-0.8b", "incorrect"),
        comparison_record("sample-5", "model-0.8b", "parse_failure"),
        comparison_record("sample-6", "model-0.8b", "inference_failure"),
    ]
    candidate_records = [
        comparison_record("sample-6", "model-2b", "incorrect"),
        comparison_record("sample-5", "model-2b", "correct"),
        comparison_record("sample-4", "model-2b", "incorrect"),
        comparison_record("sample-3", "model-2b", "correct"),
        comparison_record("sample-2", "model-2b", "incorrect"),
        comparison_record("sample-1", "model-2b", "correct"),
    ]

    comparison = compare_baseline_records(baseline_records, candidate_records)

    assert comparison["request_count"] == 6
    assert comparison["baseline_model_id"] == "model-0.8b"
    assert comparison["candidate_model_id"] == "model-2b"
    assert comparison["categories"] == {
        "both_correct": 1,
        "only_baseline_correct": 1,
        "only_candidate_correct": 2,
        "neither_correct": 2,
    }
    assert comparison["baseline_correct_count"] == 2
    assert comparison["candidate_correct_count"] == 3
    assert comparison["baseline_accuracy"] == pytest.approx(2 / 6)
    assert comparison["candidate_accuracy"] == pytest.approx(3 / 6)
    assert comparison["baseline_not_correct_count"] == 4
    assert comparison["rescued_count"] == 2
    assert comparison["rescue_rate"] == pytest.approx(0.5)
    assert comparison["rescued_accuracy_gain"] == pytest.approx(2 / 6)
    assert comparison["rescue_by_baseline_outcome"] == {
        "incorrect": {
            "baseline_count": 2,
            "rescued_count": 1,
            "rescue_rate": 0.5,
        },
        "parse_failure": {
            "baseline_count": 1,
            "rescued_count": 1,
            "rescue_rate": 1.0,
        },
        "inference_failure": {
            "baseline_count": 1,
            "rescued_count": 0,
            "rescue_rate": 0.0,
        },
    }
    assert comparison["oracle_correct_count"] == 4
    assert comparison["oracle_accuracy"] == pytest.approx(4 / 6)
    assert [item["sample_id"] for item in comparison["per_request"]] == [
        record["sample_id"] for record in baseline_records
    ]
    assert comparison["per_request"][2] == {
        "sample_id": "sample-3",
        "reference_answer": "1",
        "baseline": {
            "model_id": "model-0.8b",
            "outcome": "incorrect",
            "correct": False,
        },
        "candidate": {
            "model_id": "model-2b",
            "outcome": "correct",
            "correct": True,
        },
        "category": "only_candidate_correct",
        "rescued": True,
        "baseline_failure_outcome": "incorrect",
    }
    assert comparison["scope"] == "offline_oracle_analysis"
    assert comparison["actual_cascade_executed"] is False


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_compare_baselines_rejects_duplicate_sample_ids(side: str) -> None:
    compare_baseline_records = require_function("compare_baseline_records")
    baseline_records = [comparison_record("sample-1", "baseline", "correct")]
    candidate_records = [comparison_record("sample-1", "candidate", "correct")]
    if side == "baseline":
        baseline_records.append(baseline_records[0])
    else:
        candidate_records.append(candidate_records[0])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        compare_baseline_records(baseline_records, candidate_records)


def test_compare_baselines_rejects_mismatched_samples_and_references() -> None:
    compare_baseline_records = require_function("compare_baseline_records")
    baseline_records = [comparison_record("sample-1", "baseline", "correct")]

    with pytest.raises(ValueError, match="sample sets"):
        compare_baseline_records(
            baseline_records,
            [comparison_record("sample-2", "candidate", "correct")],
        )

    with pytest.raises(ValueError, match="reference answer"):
        compare_baseline_records(
            baseline_records,
            [
                comparison_record(
                    "sample-1",
                    "candidate",
                    "correct",
                    reference_answer="2",
                )
            ],
        )

    with pytest.raises(ValueError, match="empty"):
        compare_baseline_records([], [])


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("dataset_index", 999),
        ("question", "different question"),
        ("prompt", "different prompt"),
    ],
)
def test_compare_baselines_rejects_mismatched_experiment_inputs(
    field: str,
    different_value: object,
) -> None:
    compare_baseline_records = require_function("compare_baseline_records")
    baseline_record = comparison_record("sample-1", "baseline", "incorrect")
    candidate_record = comparison_record("sample-1", "candidate", "correct")
    candidate_record[field] = different_value

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        compare_baseline_records([baseline_record], [candidate_record])


def test_compare_model_records_reports_all_combinations_pairs_and_oracles() -> None:
    compare_model_records = require_function("compare_model_records")
    patterns = [f"{value:03b}" for value in range(8)]
    model_ids = ["model-0.8b", "model-2b", "model-4b"]
    false_outcomes = [
        {
            "000": "incorrect",
            "001": "parse_failure",
            "010": "inference_failure",
            "011": "incorrect",
        },
        {
            "000": "incorrect",
            "001": "parse_failure",
            "100": "inference_failure",
            "101": "incorrect",
        },
        {
            "000": "inference_failure",
            "010": "incorrect",
            "100": "incorrect",
            "110": "parse_failure",
        },
    ]
    records_by_model = []
    for model_index, model_id in enumerate(model_ids):
        records_by_model.append(
            [
                comparison_record(
                    f"sample-{pattern}",
                    model_id,
                    "correct" if pattern[model_index] == "1" else false_outcomes[model_index][pattern],
                )
                for pattern in patterns
            ]
        )

    comparison = compare_model_records(records_by_model)

    assert comparison["request_count"] == 8
    assert comparison["model_order"] == model_ids
    assert [item["correct_count"] for item in comparison["models"]] == [4, 4, 4]
    assert comparison["models"][1]["outcomes"]["parse_failure"] == {
        "count": 1,
        "sample_ids": ["sample-001"],
    }
    assert list(comparison["correctness_combinations"]) == patterns
    for pattern in patterns:
        assert comparison["correctness_combinations"][pattern]["count"] == 1
        assert comparison["correctness_combinations"][pattern]["sample_ids"] == [
            f"sample-{pattern}"
        ]

    pairs = {
        (item["source_index"], item["target_index"]): item
        for item in comparison["ordered_pairs"]
    }
    assert set(pairs) == {(0, 1), (0, 2), (1, 2)}
    assert pairs[(1, 2)]["source_not_correct_count"] == 4
    assert pairs[(1, 2)]["rescued_count"] == 2
    assert pairs[(1, 2)]["rescue_rate"] == pytest.approx(0.5)
    assert pairs[(1, 2)]["rescued_sample_ids"] == ["sample-001", "sample-101"]
    assert pairs[(1, 2)]["rescue_by_source_outcome"] == {
        "incorrect": {
            "source_count": 2,
            "rescued_count": 1,
            "rescue_rate": 0.5,
            "rescued_sample_ids": ["sample-101"],
        },
        "parse_failure": {
            "source_count": 1,
            "rescued_count": 1,
            "rescue_rate": 1.0,
            "rescued_sample_ids": ["sample-001"],
        },
        "inference_failure": {
            "source_count": 1,
            "rescued_count": 0,
            "rescue_rate": 0.0,
            "rescued_sample_ids": [],
        },
    }
    assert pairs[(1, 2)]["non_monotonic_count"] == 2
    assert pairs[(1, 2)]["non_monotonic_sample_ids"] == [
        "sample-010",
        "sample-110",
    ]
    assert pairs[(1, 2)]["non_monotonic_by_target_outcome"]["parse_failure"] == {
        "target_count": 1,
        "non_monotonic_count": 1,
        "non_monotonic_rate": 1.0,
        "sample_ids": ["sample-110"],
    }

    assert comparison["oracle_progression"] == [
        {
            "model_count": 1,
            "model_ids": ["model-0.8b"],
            "oracle_correct_count": 4,
            "oracle_accuracy": 0.5,
            "incremental_correct_count": 4,
            "incremental_accuracy_gain": 0.5,
            "incremental_sample_ids": [
                "sample-100",
                "sample-101",
                "sample-110",
                "sample-111",
            ],
        },
        {
            "model_count": 2,
            "model_ids": ["model-0.8b", "model-2b"],
            "oracle_correct_count": 6,
            "oracle_accuracy": 0.75,
            "incremental_correct_count": 2,
            "incremental_accuracy_gain": 0.25,
            "incremental_sample_ids": ["sample-010", "sample-011"],
        },
        {
            "model_count": 3,
            "model_ids": model_ids,
            "oracle_correct_count": 7,
            "oracle_accuracy": 0.875,
            "incremental_correct_count": 1,
            "incremental_accuracy_gain": 0.125,
            "incremental_sample_ids": ["sample-001"],
        },
    ]
    assert comparison["per_request"][1]["correctness_pattern"] == "001"
    assert comparison["scope"] == "offline_oracle_analysis"
    assert comparison["actual_cascade_executed"] is False


def test_compare_model_records_requires_two_distinct_models() -> None:
    compare_model_records = require_function("compare_model_records")
    records = [comparison_record("sample-1", "model-a", "correct")]

    with pytest.raises(ValueError, match="at least two"):
        compare_model_records([records])
    with pytest.raises(ValueError, match="model_id.*unique"):
        compare_model_records([records, records])


def test_compare_model_records_strictly_validates_common_experiment_config() -> None:
    compare_model_records = require_function("compare_model_records")
    first = [comparison_record("sample-1", "model-a", "incorrect")]
    different_config = {
        **first[0]["experiment_config"],
        "generation_config": {"temperature": 0.0, "max_tokens": 128},
    }
    second = [
        comparison_record(
            "sample-1",
            "model-b",
            "correct",
            experiment_config=different_config,
        )
    ]

    with pytest.raises(ValueError, match="common experiment config"):
        compare_model_records([first, second])
