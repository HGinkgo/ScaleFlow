"""Closed-loop vLLM serving benchmark helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import random
import re
import signal
import socket
from statistics import fmean
import subprocess
import sys
import threading
from time import perf_counter
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from scaleflow.baseline import (
    parse_final_answer,
    percentile,
    render_prompt,
    score_output,
    write_records_jsonl,
    write_summary_json,
)


_EXPECTED_PROJECT_SEED = 42
_EXPECTED_DATASET = {
    "name": "openai/grade-school-math",
    "split": "test",
    "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
    "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "expected_record_count": 1319,
}
_EXPECTED_PROMPT_TEMPLATE = (
    "Solve the following grade school math problem. Show concise reasoning.\n"
    "End with exactly one final line in this format:\n"
    "#### <number>\n"
    "Do not include units or other text after the number.\n\n"
    "Problem:\n"
    "{question}\n"
)
_EXPECTED_WARMUP_PROMPTS = [
    "Compute 2 + 3. Show concise reasoning and end with exactly: #### 5",
    "Compute 7 * 8. Show concise reasoning and end with exactly: #### 56",
    "A box has 12 items and receives 4 more. Find the total and end with exactly: #### 16",
    "Sam has 20 coins and spends 7. Find the remainder and end with exactly: #### 13",
    "Three bags contain 6 apples each. Find the total and end with exactly: #### 18",
    "A 24 meter rope is cut into 4 equal pieces. Find each length and end with exactly: #### 6",
    "A shop sells 5 pens at 3 dollars each. Find the cost and end with exactly: #### 15",
    "A train travels 40 miles per hour for 3 hours. Find the distance and end with exactly: #### 120",
]
_EXPECTED_MODELS = [
    {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    },
    {
        "model_id": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    },
    {
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    },
    {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    },
]


class SSEDecoder:
    """Incrementally decode JSON data events from an SSE byte stream."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[dict[str, Any] | None]:
        self._buffer += chunk
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        events: list[dict[str, Any] | None] = []
        while b"\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split(b"\n\n", maxsplit=1)
            data_lines = [
                line[5:].lstrip()
                for line in raw_event.splitlines()
                if line.startswith(b"data:")
            ]
            if not data_lines:
                continue
            payload = b"\n".join(data_lines).decode("utf-8")
            if payload == "[DONE]":
                events.append(None)
                continue
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("SSE data payload must be a JSON object")
            events.append(value)
        return events

    def finish(self) -> list[dict[str, Any] | None]:
        if self._buffer.strip():
            raise ValueError("incomplete SSE event at end of stream")
        return []


def build_chat_payload(
    model_id: str,
    prompt: str,
    sampling: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    logprobs = int(sampling["logprobs"])
    if logprobs < 1:
        raise ValueError("logprobs must be at least 1")
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "min_p": float(sampling["min_p"]),
        "presence_penalty": float(sampling["presence_penalty"]),
        "max_tokens": int(sampling["max_tokens"]),
        "seed": int(seed),
        "logprobs": True,
        "top_logprobs": logprobs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_performance_config(config: dict[str, Any]) -> None:
    required_sections = (
        "project",
        "dataset",
        "prompt",
        "warmup",
        "server",
        "sampling",
        "performance",
        "models",
    )
    for section in required_sections:
        if not isinstance(config.get(section), dict if section != "models" else list):
            raise ValueError(f"performance config requires {section}")

    if config["project"].get("seed") != _EXPECTED_PROJECT_SEED:
        raise ValueError(f"project.seed must remain {_EXPECTED_PROJECT_SEED}")

    dataset = config["dataset"]
    for field, expected in _EXPECTED_DATASET.items():
        if dataset.get(field) != expected:
            raise ValueError(f"dataset.{field} must remain {expected}")
    indices = dataset.get("sample_indices")
    expected_count = int(dataset.get("expected_record_count", 0))
    selection_seed = dataset.get("selection_seed")
    if dataset.get("selection_method") != "python_random_sample":
        raise ValueError("dataset.selection_method must be python_random_sample")
    if not isinstance(indices, list) or len(indices) != 128:
        raise ValueError("dataset.sample_indices must contain exactly 128 values")
    if len(set(indices)) != len(indices):
        raise ValueError("dataset.sample_indices must be unique")
    if selection_seed != _EXPECTED_PROJECT_SEED:
        raise ValueError(
            f"dataset.selection_seed must remain {_EXPECTED_PROJECT_SEED}"
        )
    expected_indices = random.Random(selection_seed).sample(
        range(expected_count),
        128,
    )
    if indices != expected_indices:
        raise ValueError("dataset.sample_indices do not match the fixed selection")
    if config["prompt"].get("template") != _EXPECTED_PROMPT_TEMPLATE:
        raise ValueError("prompt.template must remain the Phase 7 template")

    sampling = config["sampling"]
    expected_sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 384,
        "logprobs": 1,
    }
    for field, expected in expected_sampling.items():
        if sampling.get(field) != expected:
            raise ValueError(f"sampling.{field} must remain {expected}")

    server = config["server"]
    expected_server = {
        "vllm_version": "0.26.0",
        "language_model_only": True,
        "enable_thinking": False,
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    for field, expected in expected_server.items():
        if server.get(field) != expected:
            raise ValueError(f"server.{field} must remain {expected}")

    performance = config["performance"]
    if performance.get("concurrency_levels") != [1, 2, 4, 8, 16]:
        raise ValueError("performance.concurrency_levels must be 1,2,4,8,16")
    if performance.get("level_warmup_mode") != "one_wave":
        raise ValueError("performance.level_warmup_mode must remain one_wave")
    preflight_count = performance.get("preflight_request_count")
    if not isinstance(preflight_count, int) or not 1 <= preflight_count <= 128:
        raise ValueError("performance.preflight_request_count is invalid")
    if config["warmup"].get("prompts") != _EXPECTED_WARMUP_PROMPTS:
        raise ValueError("warmup.prompts must remain the eight Phase 7 prompts")

    models = config["models"]
    if models != _EXPECTED_MODELS:
        raise ValueError("models must retain the fixed model IDs and revisions")


def validate_reference_contract(
    config: dict[str, Any],
    model: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    reference_records: Sequence[dict[str, Any]],
    reference_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    dataset = config["dataset"]
    if reference_summary.get("project_seed") != config["project"]["seed"]:
        raise ValueError("Phase 7 project seed does not match")
    experiment_config = reference_summary.get("experiment_config")
    if not isinstance(experiment_config, dict):
        raise ValueError("Phase 7 summary has no experiment config")
    if experiment_config.get("project_seed") != config["project"]["seed"]:
        raise ValueError("Phase 7 experiment project seed does not match")
    if experiment_config.get("warmup_prompts") != config["warmup"]["prompts"]:
        raise ValueError("Phase 7 warmup prompts do not match")
    summary_dataset = reference_summary.get("dataset")
    if not isinstance(summary_dataset, dict):
        raise ValueError("Phase 7 summary has no dataset contract")
    dataset_checks = {
        "name": dataset["name"],
        "split": dataset["split"],
        "commit": dataset["commit"],
        "sha256": dataset["sha256"],
        "record_count": dataset["expected_record_count"],
    }
    for field, expected in dataset_checks.items():
        if summary_dataset.get(field) != expected:
            raise ValueError(f"Phase 7 dataset {field} does not match")
    if reference_summary.get("prompt_template") != config["prompt"]["template"]:
        raise ValueError("Phase 7 prompt template does not match")
    if reference_summary.get("generation_config") != config["sampling"]:
        raise ValueError("Phase 7 generation config does not match")

    backend = reference_summary.get("backend_config")
    if not isinstance(backend, dict):
        raise ValueError("Phase 7 summary has no backend config")
    if backend.get("model_id") != model["model_id"]:
        raise ValueError("Phase 7 model_id does not match")
    if backend.get("revision") != model["revision"]:
        raise ValueError("Phase 7 revision does not match")
    common_backend_fields = (
        "language_model_only",
        "enable_thinking",
        "dtype",
        "max_model_len",
        "enforce_eager",
        "enable_prefix_caching",
    )
    for field in common_backend_fields:
        if backend.get(field) != config["server"].get(field):
            raise ValueError(f"Phase 7 backend {field} does not match")
    model_info = reference_summary.get("model_info")
    if not isinstance(model_info, dict) or (
        model_info.get("vllm_version") != config["server"]["vllm_version"]
    ):
        raise ValueError("Phase 7 vLLM version does not match")
    if len(reference_records) != dataset["expected_record_count"]:
        raise ValueError("Phase 7 record count does not match the dataset")

    indexed: dict[str, dict[str, Any]] = {}
    for record in reference_records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("Phase 7 record has invalid sample_id")
        if sample_id in indexed:
            raise ValueError(f"duplicate Phase 7 sample_id: {sample_id}")
        indexed[sample_id] = record

    aligned: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        record = indexed.get(sample_id)
        if record is None:
            raise ValueError(f"Phase 7 result is missing {sample_id}")
        expected_prompt = config["prompt"]["template"].replace(
            "{question}", sample["question"]
        )
        if record.get("dataset_index") != sample["dataset_index"]:
            raise ValueError(f"Phase 7 dataset_index mismatch for {sample_id}")
        if record.get("question") != sample["question"]:
            raise ValueError(f"Phase 7 question mismatch for {sample_id}")
        if record.get("prompt") != expected_prompt:
            raise ValueError(f"Phase 7 prompt mismatch for {sample_id}")
        if record.get("model_id") != model["model_id"]:
            raise ValueError(f"Phase 7 model_id mismatch for {sample_id}")
        expected_answer = parse_final_answer(sample["reference_solution"])
        if record.get("reference_answer") != expected_answer:
            raise ValueError(f"Phase 7 reference answer mismatch for {sample_id}")
        aligned[sample_id] = record
    return aligned


def _failed_stream_result(error: str, latency_ms: float) -> dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "text": "",
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "tpot_ms": None,
        "tpot_scope": "client_observed_mean",
        "input_token_count": None,
        "output_token_count": None,
    }


async def stream_chat(
    session: Any,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Send one streaming chat request and measure client-observed timings."""

    started = clock()
    first_text_at: float | None = None
    completed_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    done_received = False
    decoder = SSEDecoder()
    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=timeout_seconds,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body}")
            async for chunk in response.content.iter_any():
                for event in decoder.feed(chunk):
                    observed_at = clock()
                    if event is None:
                        done_received = True
                        completed_at = observed_at
                        continue
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    choices = event.get("choices", [])
                    if not isinstance(choices, list):
                        raise ValueError("stream choices must be a list")
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            continue
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if first_text_at is None:
                                first_text_at = observed_at
                            text_parts.append(content)
            decoder.finish()
        if not done_received or completed_at is None:
            raise ValueError("stream ended without [DONE]")
        if usage is None:
            raise ValueError("stream ended without final usage")
        if first_text_at is None:
            raise ValueError("stream returned no non-empty text token")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if not isinstance(input_tokens, int) or input_tokens < 0:
            raise ValueError("final usage has invalid prompt_tokens")
        if not isinstance(output_tokens, int) or output_tokens < 1:
            raise ValueError("final usage has invalid completion_tokens")

        latency_ms = (completed_at - started) * 1000
        ttft_ms = (first_text_at - started) * 1000
        tpot_ms = None
        if output_tokens > 1:
            tpot_ms = (latency_ms - ttft_ms) / (output_tokens - 1)
        if latency_ms < 0 or ttft_ms < 0 or (
            tpot_ms is not None and (tpot_ms < 0 or not isfinite(tpot_ms))
        ):
            raise ValueError("stream produced invalid timing values")
        return {
            "success": True,
            "error": None,
            "text": "".join(text_parts),
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "tpot_scope": "client_observed_mean",
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
        }
    except Exception as error:
        return _failed_stream_result(
            f"{type(error).__name__}: {error}",
            (clock() - started) * 1000,
        )


async def run_closed_loop(
    items: Sequence[Any],
    concurrency: int,
    sender: Callable[[Any], Awaitable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run a bounded number of workers against a FIFO request queue."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    results: list[dict[str, Any] | None] = [None] * len(items)

    async def worker() -> None:
        while True:
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[index] = await sender(item)
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(concurrency, max(1, len(items))))
    ]
    await asyncio.gather(*workers)
    if any(result is None for result in results):
        raise RuntimeError("closed-loop run did not produce every result")
    return [result for result in results if result is not None]


def build_performance_record(
    sample: dict[str, Any],
    *,
    prompt: str,
    response: dict[str, Any],
    reference_record: dict[str, Any],
    model_id: str,
    concurrency: int,
) -> dict[str, Any]:
    if reference_record.get("sample_id") != sample["sample_id"]:
        raise ValueError("Phase 7 sample_id does not match benchmark sample")
    if reference_record.get("question") != sample["question"]:
        raise ValueError("Phase 7 question does not match benchmark sample")
    if reference_record.get("prompt") != prompt:
        raise ValueError("Phase 7 prompt does not match benchmark prompt")
    if reference_record.get("model_id") != model_id:
        raise ValueError("Phase 7 model_id does not match benchmark model")

    reference_answer = parse_final_answer(sample["reference_solution"])
    if reference_answer is None:
        raise ValueError(f"sample {sample['sample_id']} has no reference answer")
    success = bool(response["success"])
    if success:
        score = score_output(response["text"], sample["reference_solution"])
        phase7_predicted = reference_record.get("predicted_answer")
        if phase7_predicted is None:
            phase7_predicted = parse_final_answer(
                str(reference_record.get("model_output", ""))
            )
        parsed_consistency: bool | None = (
            score["predicted_answer"] == phase7_predicted
        )
        text_consistency: bool | None = (
            response["text"] == reference_record.get("model_output")
        )
    else:
        score = {
            "reference_answer": reference_answer,
            "predicted_answer": None,
            "correct": False,
            "parse_failure": False,
            "outcome": "inference_failure",
        }
        phase7_predicted = reference_record.get("predicted_answer")
        parsed_consistency = None
        text_consistency = None

    output_tokens = response.get("output_token_count")
    return {
        "sample_id": sample["sample_id"],
        "dataset_index": sample["dataset_index"],
        "question": sample["question"],
        "prompt": prompt,
        "reference_answer": score["reference_answer"],
        "model_id": model_id,
        "concurrency": concurrency,
        "model_output": response["text"],
        "predicted_answer": score["predicted_answer"],
        "phase7_predicted_answer": phase7_predicted,
        "phase7_model_output": reference_record.get("model_output"),
        "parsed_answer_consistent": parsed_consistency,
        "full_text_consistent": text_consistency,
        "success": success,
        "error": response.get("error"),
        "outcome": score["outcome"],
        "correct": score["correct"],
        "parse_failure": score["parse_failure"],
        "latency_ms": response["latency_ms"],
        "ttft_ms": response.get("ttft_ms"),
        "tpot_ms": response.get("tpot_ms"),
        "tpot_scope": response.get("tpot_scope", "client_observed_mean"),
        "input_token_count": response.get("input_token_count"),
        "output_token_count": output_tokens,
        "hit_max_tokens": success
        and isinstance(output_tokens, int)
        and output_tokens >= 384,
    }


def validate_preflight(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures = [record["sample_id"] for record in records if not record["success"]]
    mismatches = [
        record["sample_id"]
        for record in records
        if record.get("parsed_answer_consistent") is not True
    ]
    if failures or mismatches:
        details = []
        if failures:
            details.append("request failures: " + ", ".join(failures))
        if mismatches:
            details.append("parsed answer mismatches: " + ", ".join(mismatches))
        raise RuntimeError("preflight failed; " + "; ".join(details))
    return {
        "request_count": len(records),
        "parsed_answer_consistent_count": len(records),
        "full_text_consistent_count": sum(
            record.get("full_text_consistent") is True for record in records
        ),
        "passed": True,
    }


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def summarize_concurrency(
    records: Sequence[dict[str, Any]],
    *,
    concurrency: int,
    duration_seconds: float,
    peak_gpu_memory_mb: float | None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty concurrency run")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    successful = [record for record in records if record["success"]]
    latency_values = [float(record["latency_ms"]) for record in successful]
    ttft_values = [
        float(record["ttft_ms"])
        for record in successful
        if record.get("ttft_ms") is not None
    ]
    tpot_values = [
        float(record["tpot_ms"])
        for record in successful
        if record.get("tpot_ms") is not None
    ]
    input_tokens = [
        int(record["input_token_count"])
        for record in successful
        if record.get("input_token_count") is not None
    ]
    output_tokens = [
        int(record["output_token_count"])
        for record in successful
        if record.get("output_token_count") is not None
    ]
    parsed_consistency = [
        bool(record["parsed_answer_consistent"])
        for record in successful
        if record.get("parsed_answer_consistent") is not None
    ]
    text_consistency = [
        bool(record["full_text_consistent"])
        for record in successful
        if record.get("full_text_consistent") is not None
    ]
    outcome_counts = {
        outcome: sum(record["outcome"] == outcome for record in records)
        for outcome in (
            "correct",
            "incorrect",
            "parse_failure",
            "inference_failure",
        )
    }
    return {
        "concurrency": concurrency,
        "request_count": len(records),
        "successful_count": len(successful),
        "failed_count": len(records) - len(successful),
        "outcome_counts": outcome_counts,
        "accuracy": outcome_counts["correct"] / len(records),
        "duration_seconds": duration_seconds,
        "request_throughput_per_second": len(successful) / duration_seconds,
        "completed_request_throughput_per_second": len(records) / duration_seconds,
        "output_throughput_tokens_per_second": sum(output_tokens)
        / duration_seconds,
        "latency_ms": _distribution(latency_values),
        "ttft_ms": _distribution(ttft_values),
        "tpot_ms": _distribution(tpot_values),
        "tpot_scope": "client_observed_mean",
        "input_tokens": {
            "total": sum(input_tokens),
            "mean": fmean(input_tokens) if input_tokens else None,
        },
        "output_tokens": {
            "total": sum(output_tokens),
            "mean": fmean(output_tokens) if output_tokens else None,
        },
        "parsed_answer_consistency_rate": (
            sum(parsed_consistency) / len(parsed_consistency)
            if parsed_consistency
            else None
        ),
        "full_text_consistency_rate": (
            sum(text_consistency) / len(text_consistency)
            if text_consistency
            else None
        ),
        "max_token_hit_count": sum(
            bool(record.get("hit_max_tokens")) for record in records
        ),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


def build_vllm_serve_command(
    server: dict[str, Any],
    model: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model["model_id"]),
        "--served-model-name",
        str(model["model_id"]),
        "--revision",
        str(model["revision"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(server["dtype"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]
    if server.get("language_model_only") is True:
        command.append("--language-model-only")
    if server.get("enforce_eager") is True:
        command.append("--enforce-eager")
    if server.get("enable_prefix_caching") is False:
        command.append("--no-enable-prefix-caching")
    if "seed" in server:
        command.extend(["--seed", str(server["seed"])])
    return command


_MODEL_MEMORY_PATTERN = re.compile(
    r"Model loading took\s+([0-9]+(?:\.[0-9]+)?)\s+GiB(?:\s+memory)?"
)
_KV_MEMORY_PATTERN = re.compile(
    r"Available KV cache memory:\s+([0-9]+(?:\.[0-9]+)?)\s+GiB"
)
_KV_TOKEN_PATTERN = re.compile(r"GPU KV cache size:\s+([0-9,]+)\s+tokens")


def parse_server_memory_log(text: str) -> dict[str, float | int | None]:
    model_match = _MODEL_MEMORY_PATTERN.search(text)
    kv_memory_match = _KV_MEMORY_PATTERN.search(text)
    kv_token_match = _KV_TOKEN_PATTERN.search(text)
    return {
        "model_weights_gib": (
            float(model_match.group(1)) if model_match is not None else None
        ),
        "available_kv_cache_gib": (
            float(kv_memory_match.group(1))
            if kv_memory_match is not None
            else None
        ),
        "gpu_kv_cache_tokens": (
            int(kv_token_match.group(1).replace(",", ""))
            if kv_token_match is not None
            else None
        ),
    }


def select_model_config(
    config: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    matches = [model for model in config["models"] if model["model_id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"model_id is not configured exactly once: {model_id}")
    return matches[0]


def _physical_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [part.strip() for part in visible.split(",") if part.strip()]
    if len(parts) != 1 or not parts[0].isdigit():
        raise RuntimeError(
            "Phase 9 requires CUDA_VISIBLE_DEVICES to contain one physical GPU index"
        )
    return int(parts[0])


class GpuMemoryMonitor:
    """Poll one physical GPU's NVML used-memory counter."""

    def __init__(self, gpu_index: int, poll_interval_seconds: float) -> None:
        self._gpu_index = gpu_index
        self._interval = poll_interval_seconds
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._peak_mb: float | None = None
        self._overall_peak_mb: float | None = None
        self._error: str | None = None
        self._pynvml: Any = None
        self._handle: Any = None

    def _read_mb(self) -> float:
        info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return float(info.used) / (1024 * 1024)

    def start(self) -> float:
        if self._thread is not None:
            raise RuntimeError("GPU memory monitor is already running")
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        initial = self._read_mb()
        self._peak_mb = initial
        self._overall_peak_mb = initial

        def poll() -> None:
            while not self._stop_event.wait(self._interval):
                try:
                    value = self._read_mb()
                except Exception as error:
                    self._error = f"{type(error).__name__}: {error}"
                    return
                with self._lock:
                    if self._peak_mb is None or value > self._peak_mb:
                        self._peak_mb = value
                    if self._overall_peak_mb is None or value > self._overall_peak_mb:
                        self._overall_peak_mb = value

        self._thread = threading.Thread(
            target=poll,
            name="scaleflow-nvml-monitor",
            daemon=True,
        )
        self._thread.start()
        return initial

    def current_mb(self) -> float:
        if self._error is not None:
            raise RuntimeError(f"NVML monitoring failed: {self._error}")
        return self._read_mb()

    def reset_peak(self) -> float:
        current = self.current_mb()
        with self._lock:
            self._peak_mb = current
        return current

    def peak_mb(self) -> float:
        if self._error is not None:
            raise RuntimeError(f"NVML monitoring failed: {self._error}")
        with self._lock:
            if self._peak_mb is None:
                raise RuntimeError("GPU memory monitor has no observation")
            return self._peak_mb

    def overall_peak_mb(self) -> float:
        if self._error is not None:
            raise RuntimeError(f"NVML monitoring failed: {self._error}")
        with self._lock:
            if self._overall_peak_mb is None:
                raise RuntimeError("GPU memory monitor has no observation")
            return self._overall_peak_mb

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 3))
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()
        self._thread = None


class VLLMServerProcess:
    """Own one local vLLM OpenAI server process and its log."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        health_url: str,
        log_path: Path,
        startup_timeout_seconds: float,
        health_poll_interval_seconds: float,
        offline: bool,
    ) -> None:
        self.command = list(command)
        self.health_url = health_url
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self.health_poll_interval_seconds = health_poll_interval_seconds
        self.offline = offline
        self.process: subprocess.Popen[Any] | None = None
        self._log_stream: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = self.log_path.open("x", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if self.offline:
            environment["HF_HUB_OFFLINE"] = "1"
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_healthy(self) -> bool:
        if not self.is_alive():
            return False
        try:
            with urlopen(self.health_url, timeout=5) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def wait_until_healthy(self) -> None:
        if self.process is None:
            raise RuntimeError("vLLM server has not been started")
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited during startup: " + self.log_tail()
                )
            if self.is_healthy():
                return
            time.sleep(self.health_poll_interval_seconds)
        raise RuntimeError(
            "vLLM server startup timed out: " + self.log_tail()
        )

    def read_log(self) -> str:
        if self._log_stream is not None:
            self._log_stream.flush()
        if not self.log_path.exists():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def log_tail(self, line_count: int = 30) -> str:
        return "\n".join(self.read_log().splitlines()[-line_count:])

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
            except ProcessLookupError:
                pass
        if self._log_stream is not None:
            self._log_stream.flush()
            self._log_stream.close()
            self._log_stream = None


def _experiment_fingerprint(
    config: dict[str, Any],
    model: dict[str, Any],
    reference_sha256: str,
) -> str:
    contract = {
        "project_seed": config["project"]["seed"],
        "dataset": {
            field: config["dataset"][field]
            for field in (
                "name",
                "split",
                "commit",
                "sha256",
                "expected_record_count",
                "selection_method",
                "selection_seed",
                "sample_indices",
            )
        },
        "prompt_template": config["prompt"]["template"],
        "sampling": config["sampling"],
        "warmup_prompts": config["warmup"]["prompts"],
        "server": config["server"],
        "performance": config["performance"],
        "model": model,
        "phase7_reference_sha256": reference_sha256,
    }
    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


async def _send_prompt_batch(
    session: Any,
    *,
    endpoint: str,
    model_id: str,
    prompts: Sequence[str],
    sampling: dict[str, Any],
    seed: int,
    concurrency: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    async def sender(prompt: str) -> dict[str, Any]:
        return await stream_chat(
            session,
            endpoint,
            build_chat_payload(model_id, prompt, sampling, seed=seed),
            timeout_seconds=timeout_seconds,
        )

    return await run_closed_loop(prompts, concurrency, sender)


def _warmup_summary(
    responses: Sequence[dict[str, Any]],
    *,
    concurrency: int,
    duration_seconds: float,
) -> dict[str, Any]:
    failed = [response["error"] for response in responses if not response["success"]]
    if failed:
        raise RuntimeError("warmup request failed: " + "; ".join(failed))
    return {
        "request_count": len(responses),
        "successful_count": len(responses),
        "concurrency": concurrency,
        "duration_seconds": duration_seconds,
    }


async def _run_async_benchmark(
    config: dict[str, Any],
    model: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    monitor: GpuMemoryMonitor,
    server_process: VLLMServerProcess,
    output_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
    preflight_only: bool,
) -> int:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("aiohttp is required for the serving benchmark") from error

    server_config = config["server"]
    sampling = config["sampling"]
    seed = int(config["project"]["seed"])
    model_id = model["model_id"]
    endpoint = (
        f"http://{server_config['host']}:{server_config['port']}"
        "/v1/chat/completions"
    )
    timeout_seconds = float(server_config["request_timeout_seconds"])
    prompt_template = config["prompt"]["template"]
    prepared = [
        {
            "sample": sample,
            "prompt": render_prompt(prompt_template, sample["question"]),
        }
        for sample in samples
    ]

    async with aiohttp.ClientSession() as session:
        warmup_started = perf_counter()
        startup_responses = await _send_prompt_batch(
            session,
            endpoint=endpoint,
            model_id=model_id,
            prompts=config["warmup"]["prompts"],
            sampling=sampling,
            seed=seed,
            concurrency=1,
            timeout_seconds=timeout_seconds,
        )
        summary["startup_warmup"] = _warmup_summary(
            startup_responses,
            concurrency=1,
            duration_seconds=perf_counter() - warmup_started,
        )
        summary["status"] = "startup_warmup_completed"
        write_summary_json(summary_path, summary)
        print(f"{model_id}: 8条启动预热完成", flush=True)

        preflight_items = prepared[: config["performance"]["preflight_request_count"]]

        async def send_preflight(item: dict[str, Any]) -> dict[str, Any]:
            response = await stream_chat(
                session,
                endpoint,
                build_chat_payload(model_id, item["prompt"], sampling, seed=seed),
                timeout_seconds=timeout_seconds,
            )
            return build_performance_record(
                item["sample"],
                prompt=item["prompt"],
                response=response,
                reference_record=references[item["sample"]["sample_id"]],
                model_id=model_id,
                concurrency=1,
            )

        preflight_records = await run_closed_loop(
            preflight_items,
            1,
            send_preflight,
        )
        summary["preflight"] = {
            "request_count": len(preflight_records),
            "sample_ids": [record["sample_id"] for record in preflight_records],
            "successful_count": sum(record["success"] for record in preflight_records),
            "parsed_answer_consistent_count": sum(
                record.get("parsed_answer_consistent") is True
                for record in preflight_records
            ),
            "full_text_consistent_count": sum(
                record.get("full_text_consistent") is True
                for record in preflight_records
            ),
            "records": [
                {
                    "sample_id": record["sample_id"],
                    "predicted_answer": record["predicted_answer"],
                    "phase7_predicted_answer": record["phase7_predicted_answer"],
                    "parsed_answer_consistent": record["parsed_answer_consistent"],
                    "full_text_consistent": record["full_text_consistent"],
                    "error": record["error"],
                }
                for record in preflight_records
            ],
        }
        validate_preflight(preflight_records)
        summary["preflight"]["passed"] = True
        summary["status"] = "preflight_passed"
        write_summary_json(summary_path, summary)
        print(f"{model_id}: 并发1预检通过", flush=True)
        if preflight_only:
            write_records_jsonl(output_path, preflight_records)
            summary["status"] = "preflight_passed"
            write_summary_json(summary_path, summary)
            return 0

        completed_records: list[dict[str, Any]] = []
        for concurrency in config["performance"]["concurrency_levels"]:
            summary["status"] = "running"
            summary["active_concurrency"] = concurrency
            write_summary_json(summary_path, summary)
            level_warmup_items = prepared[:concurrency]
            level_warmup_started = perf_counter()
            level_warmup_responses = await _send_prompt_batch(
                session,
                endpoint=endpoint,
                model_id=model_id,
                prompts=[item["prompt"] for item in level_warmup_items],
                sampling=sampling,
                seed=seed,
                concurrency=concurrency,
                timeout_seconds=timeout_seconds,
            )
            level_warmup = _warmup_summary(
                level_warmup_responses,
                concurrency=concurrency,
                duration_seconds=perf_counter() - level_warmup_started,
            )
            monitor.reset_peak()

            async def send_sample(item: dict[str, Any]) -> dict[str, Any]:
                response = await stream_chat(
                    session,
                    endpoint,
                    build_chat_payload(
                        model_id,
                        item["prompt"],
                        sampling,
                        seed=seed,
                    ),
                    timeout_seconds=timeout_seconds,
                )
                record = build_performance_record(
                    item["sample"],
                    prompt=item["prompt"],
                    response=response,
                    reference_record=references[item["sample"]["sample_id"]],
                    model_id=model_id,
                    concurrency=concurrency,
                )
                record["experiment_fingerprint"] = summary[
                    "experiment_fingerprint"
                ]
                return record

            measured_started = perf_counter()
            level_records = await run_closed_loop(
                prepared,
                concurrency,
                send_sample,
            )
            duration_seconds = perf_counter() - measured_started
            level_summary = summarize_concurrency(
                level_records,
                concurrency=concurrency,
                duration_seconds=duration_seconds,
                peak_gpu_memory_mb=monitor.peak_mb(),
            )
            level_summary["warmup"] = level_warmup
            completed_records.extend(level_records)
            summary["levels"].append(level_summary)
            summary.pop("active_concurrency", None)
            write_records_jsonl(output_path, completed_records)
            write_summary_json(summary_path, summary)
            print(
                f"{model_id}: 并发{concurrency}完成，"
                f"{level_summary['request_throughput_per_second']:.3f} requests/s，"
                f"{level_summary['output_throughput_tokens_per_second']:.3f} tokens/s",
                flush=True,
            )

            failed_records = [record for record in level_records if not record["success"]]
            if failed_records or not server_process.is_healthy():
                errors = [record["error"] for record in failed_records]
                if not server_process.is_alive():
                    errors.append("vLLM server process exited")
                elif not server_process.is_healthy():
                    errors.append("vLLM health check failed")
                summary["status"] = "stopped_early"
                summary["stop_after_concurrency"] = concurrency
                summary["stop_reason"] = "; ".join(
                    str(error) for error in errors if error
                )
                write_summary_json(summary_path, summary)
                return 1

    summary["status"] = "completed"
    write_summary_json(summary_path, summary)
    return 0


def run_performance_experiment(
    config: dict[str, Any],
    model: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    reference_path: Path,
    reference_summary_path: Path,
    output_path: Path,
    summary_path: Path,
    server_log_path: Path,
    preflight_only: bool,
    reference_sha256: str,
    reference_summary_sha256: str,
) -> int:
    artifact_paths = (output_path, summary_path, server_log_path)
    resolved_paths = [path.resolve() for path in artifact_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("output, summary, and server log paths must be distinct")
    for path in artifact_paths:
        if path.exists():
            raise ValueError(f"refusing to overwrite existing artifact: {path}")

    server_config = dict(config["server"])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((str(server_config["host"]), int(server_config["port"])))
    except OSError as error:
        raise RuntimeError(
            f"vLLM server port is unavailable: "
            f"{server_config['host']}:{server_config['port']}"
        ) from error
    server_config["seed"] = config["project"]["seed"]
    command = build_vllm_serve_command(server_config, model)
    base_url = f"http://{server_config['host']}:{server_config['port']}"
    server_process = VLLMServerProcess(
        command,
        health_url=f"{base_url}/health",
        log_path=server_log_path,
        startup_timeout_seconds=float(server_config["startup_timeout_seconds"]),
        health_poll_interval_seconds=float(
            server_config["health_poll_interval_seconds"]
        ),
        offline=bool(server_config["offline"]),
    )
    gpu_index = _physical_gpu_index()
    monitor = GpuMemoryMonitor(
        gpu_index,
        float(server_config["memory_poll_interval_seconds"]),
    )
    summary: dict[str, Any] = {
        "scope": "gsm8k_closed_loop_concurrency_benchmark",
        "status": "starting",
        "model_id": model["model_id"],
        "revision": model["revision"],
        "vllm_version": server_config["vllm_version"],
        "experiment_fingerprint": _experiment_fingerprint(
            config,
            model,
            reference_sha256,
        ),
        "dataset": {
            **{
                field: config["dataset"][field]
                for field in (
                    "name",
                    "split",
                    "commit",
                    "sha256",
                    "expected_record_count",
                    "selection_method",
                    "selection_seed",
                    "sample_indices",
                )
            },
            "request_count": len(samples),
            "sample_ids": [sample["sample_id"] for sample in samples],
        },
        "prompt_template": config["prompt"]["template"],
        "sampling": config["sampling"],
        "server": config["server"],
        "performance": config["performance"],
        "reference": {
            "records": str(reference_path),
            "records_sha256": reference_sha256,
            "summary": str(reference_summary_path),
            "summary_sha256": reference_summary_sha256,
        },
        "preflight_only": preflight_only,
        "server_command": command,
        "server_log": str(server_log_path),
        "output": str(output_path),
        "levels": [],
        "memory": {
            "gpu_index": gpu_index,
            "gpu_memory_utilization": server_config["gpu_memory_utilization"],
            "nvml_scope": "total_device_used_memory",
        },
        "timing_scope": {
            "ttft": "client_first_non_empty_text_token",
            "tpot": "client_observed_mean_including_local_http_and_event_processing",
            "excludes": ["model_loading", "server_startup", "warmup"],
        },
    }
    exit_code = 1
    monitor_started = False
    server_started = False
    try:
        initial_mb = monitor.start()
        monitor_started = True
        summary["memory"]["initial_gpu_memory_mb"] = initial_mb
        if initial_mb > float(server_config["gpu_idle_max_used_mb"]):
            raise RuntimeError(
                "target GPU is not idle: "
                f"{initial_mb:.2f} MiB used exceeds "
                f"{server_config['gpu_idle_max_used_mb']} MiB"
            )
        print(
            f"启动 {model['model_id']}，GPU {gpu_index} 初始显存 {initial_mb:.2f} MiB",
            flush=True,
        )
        server_process.start()
        server_started = True
        server_process.wait_until_healthy()
        summary["memory"]["server_ready_gpu_memory_mb"] = monitor.current_mb()
        summary["memory"]["server_startup_peak_gpu_memory_mb"] = monitor.peak_mb()
        summary["status"] = "server_ready"
        write_summary_json(summary_path, summary)
        exit_code = asyncio.run(
            _run_async_benchmark(
                config,
                model,
                samples,
                references,
                monitor=monitor,
                server_process=server_process,
                output_path=output_path,
                summary_path=summary_path,
                summary=summary,
                preflight_only=preflight_only,
            )
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        if server_started:
            try:
                summary["server_log_tail"] = server_process.log_tail()
            except Exception as log_error:
                summary["server_log_tail_error"] = (
                    f"{type(log_error).__name__}: {log_error}"
                )
        print(summary["error"], flush=True)
        exit_code = 1
    finally:
        cleanup_errors: dict[str, str] = {}
        if server_started:
            try:
                server_process.stop()
            except Exception as error:
                cleanup_errors["server_shutdown"] = (
                    f"{type(error).__name__}: {error}"
                )
        if monitor_started:
            time.sleep(1)
            try:
                summary["memory"]["after_shutdown_gpu_memory_mb"] = (
                    monitor.current_mb()
                )
            except Exception as error:
                cleanup_errors["memory_after_shutdown"] = (
                    f"{type(error).__name__}: {error}"
                )
            try:
                summary["memory"]["observed_peak_gpu_memory_mb"] = (
                    monitor.overall_peak_mb()
                )
            except Exception as error:
                cleanup_errors["memory_observed_peak"] = (
                    f"{type(error).__name__}: {error}"
                )
            try:
                monitor.stop()
            except Exception as error:
                cleanup_errors["memory_monitor_shutdown"] = (
                    f"{type(error).__name__}: {error}"
                )
        try:
            log_text = server_process.read_log() if server_log_path.exists() else ""
        except Exception as error:
            log_text = ""
            cleanup_errors["server_log_read"] = (
                f"{type(error).__name__}: {error}"
            )
        summary["memory"].update(parse_server_memory_log(log_text))
        if cleanup_errors:
            summary["cleanup_errors"] = cleanup_errors
            if exit_code == 0:
                exit_code = 1
                summary["status"] = "completed_with_cleanup_errors"
        write_summary_json(summary_path, summary)
    return exit_code
