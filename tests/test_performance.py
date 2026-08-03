import asyncio
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from scaleflow.performance import (
    SSEDecoder,
    build_chat_payload,
    build_performance_record,
    build_vllm_serve_command,
    parse_server_memory_log,
    run_closed_loop,
    run_performance_experiment,
    stream_chat,
    summarize_concurrency,
    validate_performance_config,
    validate_preflight,
    validate_reference_contract,
)

from scaleflow.config import load_config


MODEL_ID = "Qwen/Qwen3.5-0.8B"


def test_chat_payload_matches_quality_baseline_contract() -> None:
    payload = build_chat_payload(
        MODEL_ID,
        "Solve this problem",
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 384,
            "logprobs": 1,
        },
        seed=42,
    )

    assert payload == {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Solve this problem"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 384,
        "seed": 42,
        "logprobs": True,
        "top_logprobs": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert "ignore_eos" not in payload


def test_sse_decoder_handles_arbitrary_chunk_boundaries() -> None:
    decoder = SSEDecoder()
    first = decoder.feed(b'data: {"choices":[{"delta":{"content":"A"}}]}\n')
    second = decoder.feed(b'\ndata: {"usage":{"completion_tokens":1}}\n\n')
    third = decoder.feed(b"data: [DONE]\n\n")

    assert first == []
    assert second == [
        {"choices": [{"delta": {"content": "A"}}]},
        {"usage": {"completion_tokens": 1}},
    ]
    assert third == [None]
    assert decoder.finish() == []


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    status = 200

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def text(self) -> str:
        return ""


class _FakeSession:
    def __init__(self, chunks: list[bytes]) -> None:
        self.response = _FakeResponse(chunks)
        self.calls: list[tuple[str, dict, float]] = []

    def post(self, url: str, *, json: dict, timeout: float):
        self.calls.append((url, json, timeout))
        return self.response


def test_stream_chat_observes_real_first_text_and_final_usage() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"reason"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"\\n#### 4"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n\n',
        b"data: [DONE]\n\n",
    ]
    session = _FakeSession(chunks)
    times = iter([10.0, 10.05, 10.2, 10.4, 10.5, 10.6])

    result = asyncio.run(
        stream_chat(
            session,
            "http://127.0.0.1:8000/v1/chat/completions",
            {"model": MODEL_ID},
            timeout_seconds=30.0,
            clock=lambda: next(times),
        )
    )

    assert result["success"] is True
    assert result["text"] == "reason\n#### 4"
    assert result["input_token_count"] == 12
    assert result["output_token_count"] == 3
    assert result["ttft_ms"] == pytest.approx(200.0)
    assert result["latency_ms"] == pytest.approx(600.0)
    assert result["tpot_ms"] == pytest.approx(200.0)
    assert result["tpot_scope"] == "client_observed_mean"
    assert session.calls[0][2] == 30.0


def test_stream_chat_rejects_missing_usage() -> None:
    session = _FakeSession(
        [
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    result = asyncio.run(
        stream_chat(
            session,
            "http://localhost/v1/chat/completions",
            {},
            timeout_seconds=10.0,
        )
    )

    assert result["success"] is False
    assert "usage" in result["error"]


def test_closed_loop_limits_real_inflight_requests_and_preserves_order() -> None:
    active = 0
    maximum_active = 0
    started: list[int] = []

    async def sender(item: int) -> dict[str, int]:
        nonlocal active, maximum_active
        started.append(item)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        active -= 1
        return {"item": item}

    results = asyncio.run(run_closed_loop(list(range(9)), 4, sender))

    assert maximum_active == 4
    assert started == list(range(9))
    assert [result["item"] for result in results] == list(range(9))
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(run_closed_loop([1], 0, sender))


def test_performance_record_scores_and_compares_phase7_answer() -> None:
    sample = {
        "sample_id": "gsm8k-test-0001",
        "dataset_index": 1,
        "question": "2 + 2?",
        "reference_solution": "work\n#### 4",
    }
    phase7 = {
        "sample_id": "gsm8k-test-0001",
        "question": "2 + 2?",
        "prompt": "Problem: 2 + 2?",
        "model_id": MODEL_ID,
        "model_output": "old reasoning\n#### 4",
        "predicted_answer": "4",
    }
    response = {
        "success": True,
        "error": None,
        "text": "new reasoning\n#### 4",
        "latency_ms": 100.0,
        "ttft_ms": 20.0,
        "tpot_ms": 10.0,
        "tpot_scope": "client_observed_mean",
        "input_token_count": 10,
        "output_token_count": 9,
    }

    record = build_performance_record(
        sample,
        prompt="Problem: 2 + 2?",
        response=response,
        reference_record=phase7,
        model_id=MODEL_ID,
        concurrency=4,
    )

    assert record["outcome"] == "correct"
    assert record["predicted_answer"] == "4"
    assert record["phase7_predicted_answer"] == "4"
    assert record["parsed_answer_consistent"] is True
    assert record["full_text_consistent"] is False
    assert record["concurrency"] == 4


def test_preflight_fails_closed_on_parsed_answer_mismatch() -> None:
    records = [
        {
            "sample_id": "a",
            "success": True,
            "parsed_answer_consistent": True,
            "full_text_consistent": False,
        },
        {
            "sample_id": "b",
            "success": True,
            "parsed_answer_consistent": False,
            "full_text_consistent": False,
        },
    ]

    with pytest.raises(RuntimeError, match="b"):
        validate_preflight(records)


def test_concurrency_summary_uses_successful_usage_and_wall_time() -> None:
    records = [
        {
            "success": True,
            "outcome": "correct",
            "latency_ms": 100.0,
            "ttft_ms": 20.0,
            "tpot_ms": 10.0,
            "input_token_count": 10,
            "output_token_count": 9,
            "parsed_answer_consistent": True,
            "full_text_consistent": True,
        },
        {
            "success": True,
            "outcome": "incorrect",
            "latency_ms": 300.0,
            "ttft_ms": 60.0,
            "tpot_ms": 30.0,
            "input_token_count": 14,
            "output_token_count": 1,
            "parsed_answer_consistent": False,
            "full_text_consistent": False,
        },
        {
            "success": False,
            "outcome": "inference_failure",
            "latency_ms": 50.0,
            "ttft_ms": None,
            "tpot_ms": None,
            "input_token_count": None,
            "output_token_count": None,
            "parsed_answer_consistent": None,
            "full_text_consistent": None,
        },
    ]

    summary = summarize_concurrency(
        records,
        concurrency=2,
        duration_seconds=2.0,
        peak_gpu_memory_mb=1234.0,
    )

    assert summary["request_count"] == 3
    assert summary["successful_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["request_throughput_per_second"] == 1.0
    assert summary["output_throughput_tokens_per_second"] == 5.0
    assert summary["latency_ms"] == {"mean": 200.0, "p50": 200.0, "p95": 290.0}
    assert summary["ttft_ms"] == {"mean": 40.0, "p50": 40.0, "p95": 58.0}
    assert summary["tpot_ms"] == {"mean": 20.0, "p50": 20.0, "p95": 29.0}
    assert summary["input_tokens"]["mean"] == 12.0
    assert summary["output_tokens"]["mean"] == 5.0
    assert summary["parsed_answer_consistency_rate"] == 0.5
    assert summary["full_text_consistency_rate"] == 0.5
    assert summary["peak_gpu_memory_mb"] == 1234.0


def test_server_command_pins_memory_revision_and_chat_mode(tmp_path: Path) -> None:
    command = build_vllm_serve_command(
        {
            "host": "127.0.0.1",
            "port": 8123,
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "language_model_only": True,
        },
        {"model_id": MODEL_ID, "revision": "fixed-revision"},
    )

    rendered = " ".join(command)
    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    assert "serve" in command
    assert "--revision fixed-revision" in rendered
    assert "--gpu-memory-utilization 0.9" in rendered
    assert "--language-model-only" in command
    assert "--enforce-eager" in command
    assert "--no-enable-prefix-caching" in command
    assert "--generation-config vllm" in rendered
    kwargs = command[command.index("--default-chat-template-kwargs") + 1]
    assert json.loads(kwargs) == {"enable_thinking": False}


def test_parse_server_memory_log_extracts_reported_budgets() -> None:
    text = """
    Model loading took 16.8 GiB memory and 3.72 seconds
    Available KV cache memory: 4.72 GiB
    GPU KV cache size: 155,648 tokens
    """

    assert parse_server_memory_log(text) == {
        "model_weights_gib": 16.8,
        "available_kv_cache_gib": 4.72,
        "gpu_kv_cache_tokens": 155648,
    }


def test_phase9_config_freezes_same_128_requests_and_memory_budget() -> None:
    config = load_config("configs/qwen35_gsm8k_concurrency.yaml")

    validate_performance_config(config)
    assert config["dataset"]["sample_indices"] == random.Random(42).sample(
        range(1319), 128
    )
    assert config["performance"]["concurrency_levels"] == [1, 2, 4, 8, 16]
    assert config["server"]["gpu_memory_utilization"] == 0.90
    assert config["sampling"]["max_tokens"] == 384
    assert len(config["warmup"]["prompts"]) == 8
    assert [model["model_id"] for model in config["models"]] == [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ]

    changed = copy.deepcopy(config)
    changed["sampling"]["max_tokens"] = 32
    with pytest.raises(ValueError, match="max_tokens"):
        validate_performance_config(changed)

    changed = copy.deepcopy(config)
    changed["project"]["seed"] = 7
    with pytest.raises(ValueError, match="project.seed"):
        validate_performance_config(changed)

    changed = copy.deepcopy(config)
    changed["dataset"]["selection_seed"] = 7
    changed["dataset"]["sample_indices"] = random.Random(7).sample(
        range(1319), 128
    )
    with pytest.raises(ValueError, match="selection_seed"):
        validate_performance_config(changed)

    changed = copy.deepcopy(config)
    changed["models"][0]["revision"] = "different-revision"
    with pytest.raises(ValueError, match="models"):
        validate_performance_config(changed)

    changed = copy.deepcopy(config)
    changed["performance"]["level_warmup_mode"] = "none"
    with pytest.raises(ValueError, match="level_warmup_mode"):
        validate_performance_config(changed)


def test_reference_contract_rejects_generation_or_revision_drift() -> None:
    config = {
        "project": {"seed": 42},
        "dataset": {
            "name": "gsm8k",
            "split": "test",
            "commit": "dataset-commit",
            "sha256": "dataset-sha",
            "expected_record_count": 1,
            "sample_indices": [0],
        },
        "prompt": {"template": "Problem: {question}"},
        "warmup": {"prompts": ["warmup"]},
        "server": {
            "vllm_version": "0.26.0",
            "language_model_only": True,
            "enable_thinking": False,
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 384,
            "logprobs": 1,
        },
    }
    model = {"model_id": MODEL_ID, "revision": "fixed-revision"}
    samples = [
        {
            "sample_id": "gsm8k-test-0000",
            "dataset_index": 0,
            "question": "2 + 2?",
            "reference_solution": "#### 4",
        }
    ]
    records = [
        {
            "sample_id": "gsm8k-test-0000",
            "dataset_index": 0,
            "question": "2 + 2?",
            "prompt": "Problem: 2 + 2?",
            "reference_answer": "4",
            "model_id": MODEL_ID,
            "model_output": "#### 4",
            "predicted_answer": "4",
        }
    ]
    summary = {
        "project_seed": 42,
        "dataset": {
            "name": "gsm8k",
            "split": "test",
            "commit": "dataset-commit",
            "sha256": "dataset-sha",
            "record_count": 1,
        },
        "prompt_template": "Problem: {question}",
        "generation_config": dict(config["sampling"]),
        "backend_config": {
            **config["server"],
            "model_id": MODEL_ID,
            "revision": "fixed-revision",
            "gpu_memory_utilization": 0.25,
        },
        "model_info": {"vllm_version": "0.26.0"},
        "experiment_config": {
            "project_seed": 42,
            "warmup_prompts": ["warmup"],
        },
    }

    aligned = validate_reference_contract(
        config,
        model,
        samples,
        records,
        summary,
    )

    assert list(aligned) == ["gsm8k-test-0000"]
    bad_summary = copy.deepcopy(summary)
    bad_summary["generation_config"]["max_tokens"] = 32
    with pytest.raises(ValueError, match="generation"):
        validate_reference_contract(config, model, samples, records, bad_summary)
    bad_summary = copy.deepcopy(summary)
    bad_summary["backend_config"]["revision"] = "other"
    with pytest.raises(ValueError, match="revision"):
        validate_reference_contract(config, model, samples, records, bad_summary)

    bad_summary = copy.deepcopy(summary)
    bad_summary["project_seed"] = 7
    with pytest.raises(ValueError, match="project seed"):
        validate_reference_contract(config, model, samples, records, bad_summary)

    bad_summary = copy.deepcopy(summary)
    bad_summary["experiment_config"]["warmup_prompts"] = ["different"]
    with pytest.raises(ValueError, match="warmup"):
        validate_reference_contract(config, model, samples, records, bad_summary)

    with pytest.raises(ValueError, match="record count"):
        validate_reference_contract(config, model, samples, [], summary)


def _small_benchmark_fixture() -> tuple[
    dict, dict, list[dict], dict[str, dict]
]:
    config = {
        "project": {"seed": 42},
        "dataset": {
            "name": "openai/grade-school-math",
            "split": "test",
            "commit": "dataset-commit",
            "sha256": "dataset-sha",
            "expected_record_count": 2,
            "selection_method": "python_random_sample",
            "selection_seed": 42,
            "sample_indices": [0, 1],
        },
        "prompt": {"template": "Problem: {question}"},
        "warmup": {"prompts": ["warmup-a", "warmup-b"]},
        "server": {
            "vllm_version": "0.26.0",
            "host": "127.0.0.1",
            "port": 0,
            "language_model_only": True,
            "enable_thinking": False,
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "startup_timeout_seconds": 1,
            "request_timeout_seconds": 1,
            "health_poll_interval_seconds": 0.01,
            "memory_poll_interval_seconds": 0.01,
            "gpu_idle_max_used_mb": 512,
            "offline": True,
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 384,
            "logprobs": 1,
        },
        "performance": {
            "concurrency_levels": [1, 2],
            "preflight_request_count": 1,
            "level_warmup_mode": "one_wave",
        },
    }
    model = {"model_id": MODEL_ID, "revision": "fixed-revision"}
    samples = [
        {
            "sample_id": f"gsm8k-test-000{index}",
            "dataset_index": index,
            "question": f"q{index + 1}",
            "reference_solution": "#### 4",
        }
        for index in range(2)
    ]
    references = {
        sample["sample_id"]: {
            "sample_id": sample["sample_id"],
            "dataset_index": sample["dataset_index"],
            "question": sample["question"],
            "prompt": f"Problem: {sample['question']}",
            "reference_answer": "4",
            "model_id": MODEL_ID,
            "model_output": "#### 4",
            "predicted_answer": "4",
        }
        for sample in samples
    }
    return config, model, samples, references


class _AsyncSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _BenchmarkMonitor:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_peak(self) -> float:
        self.reset_count += 1
        return 100.0

    def peak_mb(self) -> float:
        return 200.0


class _HealthyServer:
    def is_healthy(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True


def _successful_stream_result() -> dict:
    return {
        "success": True,
        "error": None,
        "text": "#### 4",
        "latency_ms": 10.0,
        "ttft_ms": 2.0,
        "tpot_ms": 4.0,
        "tpot_scope": "client_observed_mean",
        "input_token_count": 5,
        "output_token_count": 3,
    }


def test_async_benchmark_runs_each_level_and_persists_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scaleflow.performance as performance

    config, model, samples, references = _small_benchmark_fixture()
    monitor = _BenchmarkMonitor()
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientSession=_AsyncSession),
    )

    async def fake_stream(*args, **kwargs) -> dict:
        return _successful_stream_result()

    monkeypatch.setattr(performance, "stream_chat", fake_stream)
    output_path = tmp_path / "records.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = {"experiment_fingerprint": "fingerprint", "levels": []}

    exit_code = asyncio.run(
        performance._run_async_benchmark(
            config,
            model,
            samples,
            references,
            monitor=monitor,
            server_process=_HealthyServer(),
            output_path=output_path,
            summary_path=summary_path,
            summary=summary,
            preflight_only=False,
        )
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert exit_code == 0
    assert summary["status"] == "completed"
    assert [level["concurrency"] for level in summary["levels"]] == [1, 2]
    assert [record["concurrency"] for record in records] == [1, 1, 2, 2]
    assert monitor.reset_count == 2


def test_async_benchmark_stops_higher_levels_after_request_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scaleflow.performance as performance

    config, model, samples, references = _small_benchmark_fixture()
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientSession=_AsyncSession),
    )

    async def fake_stream(session, endpoint, payload, **kwargs) -> dict:
        prompt = payload["messages"][0]["content"]
        if prompt == "Problem: q2":
            failed = _successful_stream_result()
            failed.update(
                success=False,
                error="RuntimeError: request failed",
                text="",
                ttft_ms=None,
                tpot_ms=None,
                input_token_count=None,
                output_token_count=None,
            )
            return failed
        return _successful_stream_result()

    monkeypatch.setattr(performance, "stream_chat", fake_stream)
    output_path = tmp_path / "records.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = {"experiment_fingerprint": "fingerprint", "levels": []}

    exit_code = asyncio.run(
        performance._run_async_benchmark(
            config,
            model,
            samples,
            references,
            monitor=_BenchmarkMonitor(),
            server_process=_HealthyServer(),
            output_path=output_path,
            summary_path=summary_path,
            summary=summary,
            preflight_only=False,
        )
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert exit_code == 1
    assert summary["status"] == "stopped_early"
    assert summary["stop_after_concurrency"] == 1
    assert len(summary["levels"]) == 1
    assert len(records) == 2


def test_experiment_rejects_colliding_artifact_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "same.json"

    with pytest.raises(ValueError, match="distinct"):
        run_performance_experiment(
            {},
            {},
            [],
            {},
            reference_path=tmp_path / "reference.jsonl",
            reference_summary_path=tmp_path / "reference-summary.json",
            output_path=artifact,
            summary_path=artifact,
            server_log_path=artifact,
            preflight_only=False,
            reference_sha256="records-sha",
            reference_summary_sha256="summary-sha",
        )


def test_experiment_records_cleanup_errors_and_still_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scaleflow.performance as performance

    config, model, samples, references = _small_benchmark_fixture()

    class FailingMonitor:
        current_calls = 0

        def start(self) -> float:
            return 10.0

        def current_mb(self) -> float:
            self.current_calls += 1
            if self.current_calls > 1:
                raise RuntimeError("NVML final read failed")
            return 200.0

        def peak_mb(self) -> float:
            return 220.0

        def overall_peak_mb(self) -> float:
            return 230.0

        def stop(self) -> None:
            return None

    class FailingServer:
        def __init__(self, *, log_path: Path) -> None:
            self.log_path = log_path

        def start(self) -> None:
            self.log_path.write_text(
                "Model loading took 1.5 GiB memory\n"
                "Available KV cache memory: 2.5 GiB\n"
                "GPU KV cache size: 10,000 tokens\n"
            )

        def wait_until_healthy(self) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError("shutdown failed")

        def read_log(self) -> str:
            return self.log_path.read_text()

        def log_tail(self) -> str:
            return ""

    monkeypatch.setattr(performance, "_physical_gpu_index", lambda: 0)
    monkeypatch.setattr(
        performance,
        "GpuMemoryMonitor",
        lambda *args, **kwargs: FailingMonitor(),
    )
    monkeypatch.setattr(
        performance,
        "VLLMServerProcess",
        lambda *args, log_path, **kwargs: FailingServer(log_path=log_path),
    )
    monkeypatch.setattr(performance.time, "sleep", lambda seconds: None)

    async def fake_benchmark(*args, summary: dict, **kwargs) -> int:
        summary["status"] = "completed"
        return 0

    monkeypatch.setattr(performance, "_run_async_benchmark", fake_benchmark)
    summary_path = tmp_path / "summary.json"

    exit_code = run_performance_experiment(
        config,
        model,
        samples,
        references,
        reference_path=tmp_path / "reference.jsonl",
        reference_summary_path=tmp_path / "reference-summary.json",
        output_path=tmp_path / "records.jsonl",
        summary_path=summary_path,
        server_log_path=tmp_path / "server.log",
        preflight_only=False,
        reference_sha256="records-sha",
        reference_summary_sha256="summary-sha",
    )

    summary = json.loads(summary_path.read_text())
    assert exit_code == 1
    assert summary["status"] == "completed_with_cleanup_errors"
    assert "server_shutdown" in summary["cleanup_errors"]
    assert "memory_after_shutdown" in summary["cleanup_errors"]
    assert summary["memory"]["model_weights_gib"] == 1.5
