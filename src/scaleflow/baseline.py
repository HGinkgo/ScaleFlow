"""Single-model evaluation helpers."""

from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import re
import shutil
from statistics import fmean
from typing import Any
from urllib.request import urlopen

from scaleflow.backends.base import Backend
from scaleflow.schemas import InferenceRequest


_FINAL_ANSWER_PATTERN = re.compile(
    r"(?m)^[ \t]*####[ \t]*"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"[ \t]*$"
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(
    path: str | Path,
    source_url: str,
    expected_sha256: str,
) -> Path:
    dataset_path = Path(path)
    if dataset_path.exists():
        actual_sha256 = file_sha256(dataset_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "dataset SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".part")
    try:
        with urlopen(source_url) as response, temporary_path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "downloaded dataset SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        temporary_path.replace(dataset_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return dataset_path


def load_gsm8k_jsonl(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid GSM8K JSON on line {line_number}") from error
            if not isinstance(item, dict):
                raise ValueError(f"GSM8K line {line_number} must be an object")
            question = item.get("question")
            answer = item.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"GSM8K line {line_number} has no question")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"GSM8K line {line_number} has no answer")
            records.append({"question": question, "answer": answer})
    if not records:
        raise ValueError("GSM8K dataset is empty")
    return records


def select_samples(
    records: Sequence[dict[str, str]],
    sample_indices: Sequence[int],
) -> list[dict[str, Any]]:
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample indices must be unique")

    selected: list[dict[str, Any]] = []
    for index in sample_indices:
        if not 0 <= index < len(records):
            raise ValueError(f"sample index out of range: {index}")
        selected.append(
            {
                "sample_id": f"gsm8k-test-{index:04d}",
                "dataset_index": index,
                "question": records[index]["question"],
                "reference_solution": records[index]["answer"],
            }
        )
    return selected


def _normalize_number(value: str) -> str:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric answer: {value}") from error
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def parse_final_answer(text: str) -> str | None:
    matches = _FINAL_ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def score_output(output: str, reference_solution: str) -> dict[str, Any]:
    reference_answer = parse_final_answer(reference_solution)
    if reference_answer is None:
        raise ValueError("reference solution has no valid final answer")

    predicted_answer = parse_final_answer(output)
    if predicted_answer is None:
        return {
            "reference_answer": reference_answer,
            "predicted_answer": None,
            "correct": False,
            "parse_failure": True,
            "outcome": "parse_failure",
        }

    correct = predicted_answer == reference_answer
    return {
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "correct": correct,
        "parse_failure": False,
        "outcome": "correct" if correct else "incorrect",
    }


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile without values")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def point_biserial_correlation(
    values: Sequence[float],
    labels: Sequence[bool],
) -> float | None:
    if len(values) != len(labels):
        raise ValueError("values and labels must have the same length")
    if len(values) < 2:
        return None

    numeric_values = [float(value) for value in values]
    numeric_labels = [1.0 if label else 0.0 for label in labels]
    mean_value = fmean(numeric_values)
    mean_label = fmean(numeric_labels)
    centered_values = [value - mean_value for value in numeric_values]
    centered_labels = [value - mean_label for value in numeric_labels]
    denominator = sqrt(
        sum(value * value for value in centered_values)
        * sum(value * value for value in centered_labels)
    )
    if denominator == 0:
        return None
    numerator = sum(
        value * label
        for value, label in zip(centered_values, centered_labels, strict=True)
    )
    return numerator / denominator


def render_prompt(template: str, question: str) -> str:
    if "{question}" not in template:
        raise ValueError("prompt template must contain {question}")
    return template.replace("{question}", question)


def run_baseline_samples(
    samples: Sequence[dict[str, Any]],
    backend: Backend,
    prompt_template: str,
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        prompt = render_prompt(prompt_template, sample["question"])
        response = backend.generate(
            InferenceRequest(
                request_id=sample["sample_id"],
                prompt=prompt,
                metadata={"dataset_index": sample["dataset_index"]},
            )
        )
        reference_answer = parse_final_answer(sample["reference_solution"])
        if reference_answer is None:
            raise ValueError(
                f"sample {sample['sample_id']} has no valid reference answer"
            )

        if response.success:
            score = score_output(response.text, sample["reference_solution"])
        else:
            score = {
                "reference_answer": reference_answer,
                "predicted_answer": None,
                "correct": False,
                "parse_failure": False,
                "outcome": "inference_failure",
            }

        output_token_count = len(response.token_logprobs)
        tokens_per_second = None
        if response.success and response.latency_ms > 0:
            tokens_per_second = output_token_count * 1000 / response.latency_ms

        records.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_index": sample["dataset_index"],
                "question": sample["question"],
                "prompt": prompt,
                "reference_answer": score["reference_answer"],
                "model_id": response.model_id,
                "model_output": response.text,
                "predicted_answer": score["predicted_answer"],
                "success": response.success,
                "error": response.error,
                "outcome": score["outcome"],
                "correct": score["correct"],
                "parse_failure": score["parse_failure"],
                "latency_ms": response.latency_ms,
                "output_token_count": output_token_count,
                "tokens_per_second": tokens_per_second,
                "hit_max_tokens": response.success
                and output_token_count >= max_tokens,
                "token_logprobs": list(response.token_logprobs),
                "confidence": response.confidence,
                "confidence_method": response.confidence_method,
                "gpu_memory_used_mb": response.gpu_memory_used_mb,
            }
        )
    return records


def warmup_backend(
    backend: Backend,
    prompts: Sequence[str],
) -> dict[str, int | float]:
    total_latency_ms = 0.0
    successful_count = 0
    for index, prompt in enumerate(prompts, start=1):
        response = backend.generate(
            InferenceRequest(
                request_id=f"warmup-{index:02d}",
                prompt=prompt,
                metadata={"warmup": True},
            )
        )
        total_latency_ms += response.latency_ms
        if not response.success:
            raise RuntimeError(
                f"warmup failed for warmup-{index:02d}: {response.error}"
            )
        successful_count += 1
    return {
        "request_count": len(prompts),
        "successful_count": successful_count,
        "total_latency_ms": total_latency_ms,
    }


def _optional_mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def summarize_baseline(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty baseline run")

    request_count = len(records)
    successful_records = [record for record in records if record["success"]]
    latency_values = [float(record["latency_ms"]) for record in records]
    token_counts = [int(record["output_token_count"]) for record in records]
    tokens_per_second = [
        float(record["tokens_per_second"])
        for record in successful_records
        if record["tokens_per_second"] is not None
    ]
    successful_latency_ms = sum(
        float(record["latency_ms"]) for record in successful_records
    )
    aggregate_tokens_per_second = None
    if successful_latency_ms > 0:
        aggregate_tokens_per_second = (
            sum(int(record["output_token_count"]) for record in successful_records)
            * 1000
            / successful_latency_ms
        )

    confidence_records = [
        record
        for record in successful_records
        if isinstance(record.get("confidence"), (int, float))
    ]
    confidence_values = [float(record["confidence"]) for record in confidence_records]
    correctness_labels = [bool(record["correct"]) for record in confidence_records]
    correct_confidences = [
        float(record["confidence"])
        for record in confidence_records
        if record["correct"]
    ]
    incorrect_confidences = [
        float(record["confidence"])
        for record in confidence_records
        if record["outcome"] == "incorrect"
    ]
    parse_failure_confidences = [
        float(record["confidence"])
        for record in confidence_records
        if record["outcome"] == "parse_failure"
    ]
    parsed_confidence_records = [
        record
        for record in confidence_records
        if record["outcome"] in {"correct", "incorrect"}
    ]
    memory_values = [
        float(record["gpu_memory_used_mb"])
        for record in records
        if record.get("gpu_memory_used_mb") is not None
    ]

    correct_count = sum(record["outcome"] == "correct" for record in records)
    incorrect_count = sum(record["outcome"] == "incorrect" for record in records)
    parse_failure_count = sum(
        record["outcome"] == "parse_failure" for record in records
    )
    inference_failure_count = sum(
        record["outcome"] == "inference_failure" for record in records
    )

    return {
        "request_count": request_count,
        "successful_count": len(successful_records),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "parse_failure_count": parse_failure_count,
        "inference_failure_count": inference_failure_count,
        "max_token_hit_count": sum(
            bool(record.get("hit_max_tokens")) for record in records
        ),
        "accuracy": correct_count / request_count,
        "parse_failure_rate": parse_failure_count / request_count,
        "latency_ms": {
            "mean": fmean(latency_values),
            "p50": percentile(latency_values, 50),
            "p95": percentile(latency_values, 95),
        },
        "output_tokens": {
            "total": sum(token_counts),
            "mean": fmean(token_counts),
        },
        "tokens_per_second": {
            "mean": _optional_mean(tokens_per_second),
            "aggregate": aggregate_tokens_per_second,
        },
        "gpu_memory_used_mb": {
            "peak": max(memory_values) if memory_values else None,
        },
        "confidence_analysis": {
            "sample_count": len(confidence_records),
            "point_biserial_correlation": point_biserial_correlation(
                confidence_values,
                correctness_labels,
            ),
            "point_biserial_correlation_parsed_only": point_biserial_correlation(
                [float(record["confidence"]) for record in parsed_confidence_records],
                [bool(record["correct"]) for record in parsed_confidence_records],
            ),
            "mean_confidence_correct": _optional_mean(correct_confidences),
            "mean_confidence_incorrect": _optional_mean(incorrect_confidences),
            "mean_confidence_parse_failure": _optional_mean(
                parse_failure_confidences
            ),
            "scope": "preliminary_observation",
            "note": (
                "Exploratory result from a small fixed sample; "
                "not a formal statistical conclusion."
            ),
        },
    }


def write_records_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
