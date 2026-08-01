from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scaleflow.backends import MockBackend
from scaleflow import cli
from scaleflow.config import load_config


MODEL_08 = "Qwen/Qwen3.5-0.8B"
MODEL_2 = "Qwen/Qwen3.5-2B"
MODEL_4 = "Qwen/Qwen3.5-4B"
MODEL_9 = "Qwen/Qwen3.5-9B"


def test_all_records_selection_uses_original_order_and_ignores_seed() -> None:
    indices = cli._resolve_sample_indices(
        {
            "selection_method": "all_records",
            "selection_seed": 999,
            "expected_record_count": 4,
        },
        record_count=4,
    )

    assert indices == [0, 1, 2, 3]


def test_missing_selection_method_keeps_explicit_index_selection() -> None:
    indices = cli._resolve_sample_indices(
        {"sample_indices": [2, 0], "expected_record_count": 3},
        record_count=3,
    )

    assert indices == [2, 0]


def test_all_records_rejects_unexpected_dataset_count() -> None:
    with pytest.raises(cli.ConfigError, match="all_records"):
        cli._resolve_sample_indices(
            {
                "selection_method": "all_records",
                "expected_record_count": 4,
            },
            record_count=3,
        )


def test_module_cli_help_starts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""

    completed = subprocess.run(
        [sys.executable, "-m", "scaleflow", "--help"],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ScaleFlow" in completed.stdout
    assert "run-mock" in completed.stdout
    assert "run-vllm" in completed.stdout
    assert "run-gsm8k" in completed.stdout
    assert "compare-gsm8k" in completed.stdout
    assert "compare-gsm8k-multi" in completed.stdout


def test_run_mock_cli_writes_deterministic_routes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    def run(output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "scaleflow",
                "run-mock",
                "--config",
                "configs/mock_qwen35.yaml",
                "--output",
                str(output),
            ],
            cwd=project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    first = run(first_output)
    second = run(second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_output.read_text(encoding="utf-8") == second_output.read_text(
        encoding="utf-8"
    )

    records = [
        json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["request_id"] for record in records] == ["easy", "medium", "hard"]
    assert [[step["model_id"] for step in record["decision_trace"]] for record in records] == [
        [MODEL_08],
        [MODEL_08, MODEL_2],
        [MODEL_08, MODEL_2, MODEL_4, MODEL_9],
    ]
    assert [record["escalation_count"] for record in records] == [0, 1, 3]
    assert [record["total_latency_ms"] for record in records] == [12.0, 40.0, 185.0]
    assert [record["final_answer"] for record in records] == [
        "4",
        "The final price is 60 yuan.",
        "8",
    ]
    assert [step["reason"] for step in records[2]["decision_trace"]] == [
        "confidence_below_threshold",
        "confidence_below_threshold",
        "confidence_below_threshold",
        "maximum_model_reached",
    ]
    assert all(record["success"] is True for record in records)
    assert all(record["error"] is None for record in records)


def test_run_gsm8k_cli_writes_records_and_summary_without_real_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        '{"question":"What is 2 + 2?","answer":"reason\\n#### 4"}\n',
        encoding="utf-8",
    )
    dataset_sha256 = sha256(dataset.read_bytes()).hexdigest()
    config = {
        "project": {"name": "ScaleFlow", "seed": 42},
        "dataset": {
            "name": "openai/grade-school-math",
            "split": "test",
            "commit": "test-commit",
            "source_url": dataset.as_uri(),
            "local_path": str(dataset),
            "sha256": dataset_sha256,
            "expected_record_count": 1,
            "selection_method": "explicit_indices",
            "selection_seed": 42,
            "sample_indices": [0],
        },
        "prompt": {"template": "Problem:\n{question}\n"},
        "warmup": {"prompts": ["1 + 1"]},
        "backend": {"model_id": MODEL_08},
        "sampling": {"max_tokens": 8},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    backend = MockBackend(
        model_id=MODEL_08,
        responses={
            "warmup-01": {
                "text": "#### 2",
                "confidence": 0.8,
                "latency_ms": 1.0,
            },
            "gsm8k-test-0000": {
                "text": "reason\n#### 4",
                "confidence": 0.9,
                "latency_ms": 2.0,
            },
        },
    )
    monkeypatch.setattr(cli, "_build_vllm_backend", lambda loaded: backend)
    output_path = tmp_path / "results" / "records.jsonl"
    summary_path = tmp_path / "results" / "summary.json"

    exit_code = cli.run_gsm8k(config_path, output_path, summary_path)

    assert exit_code == 0
    record = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert record["sample_id"] == "gsm8k-test-0000"
    assert record["outcome"] == "correct"
    assert record["experiment_config"] == {
        "project_seed": 42,
        "dataset": {
            "name": "openai/grade-school-math",
            "split": "test",
            "commit": "test-commit",
            "sha256": dataset_sha256,
            "expected_record_count": 1,
            "selection_method": "explicit_indices",
            "selection_seed": 42,
            "sample_indices": [0],
        },
        "prompt_template": "Problem:\n{question}\n",
        "warmup_prompts": ["1 + 1"],
        "generation_config": {"max_tokens": 8},
        "backend_common_config": {},
    }
    assert len(record["experiment_fingerprint"]) == 64
    assert summary["request_count"] == 1
    assert summary["experiment_config"] == record["experiment_config"]
    assert summary["experiment_fingerprint"] == record["experiment_fingerprint"]
    assert summary["dataset"]["sample_ids"] == ["gsm8k-test-0000"]
    assert summary["warmup"]["request_count"] == 1
    assert summary["model_info"]["backend"] == "mock"


def test_gsm8k_backend_disables_prefix_caching(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture_backend(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "VLLMBackend", capture_backend)
    config = load_config("configs/qwen35_0_8b_gsm8k.yaml")

    cli._build_vllm_backend(config)

    assert captured["enable_prefix_caching"] is False
    assert captured["gpu_memory_utilization"] == 0.25
    assert captured["max_model_len"] == 2048


def test_compare_gsm8k_cli_aligns_jsonl_and_writes_comparison(
    tmp_path: Path,
    capsys,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output_path = tmp_path / "comparison.json"
    baseline_records = [
        {
            "sample_id": "sample-1",
            "dataset_index": 1,
            "question": "question 1",
            "prompt": "prompt 1",
            "model_id": MODEL_08,
            "reference_answer": "1",
            "outcome": "correct",
            "correct": True,
        },
        {
            "sample_id": "sample-2",
            "dataset_index": 2,
            "question": "question 2",
            "prompt": "prompt 2",
            "model_id": MODEL_08,
            "reference_answer": "2",
            "outcome": "parse_failure",
            "correct": False,
        },
    ]
    candidate_records = [
        {
            "sample_id": "sample-2",
            "dataset_index": 2,
            "question": "question 2",
            "prompt": "prompt 2",
            "model_id": MODEL_2,
            "reference_answer": "2",
            "outcome": "correct",
            "correct": True,
        },
        {
            "sample_id": "sample-1",
            "dataset_index": 1,
            "question": "question 1",
            "prompt": "prompt 1",
            "model_id": MODEL_2,
            "reference_answer": "1",
            "outcome": "incorrect",
            "correct": False,
        },
    ]
    baseline_path.write_text(
        "".join(json.dumps(record) + "\n" for record in baseline_records),
        encoding="utf-8",
    )
    candidate_path.write_text(
        "".join(json.dumps(record) + "\n" for record in candidate_records),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "compare-gsm8k",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert comparison["categories"] == {
        "both_correct": 0,
        "only_baseline_correct": 1,
        "only_candidate_correct": 1,
        "neither_correct": 0,
    }
    assert comparison["rescued_count"] == 1
    assert comparison["rescue_by_baseline_outcome"]["parse_failure"] == {
        "baseline_count": 1,
        "rescued_count": 1,
        "rescue_rate": 1.0,
    }
    assert comparison["oracle_accuracy"] == 1.0
    assert json.loads(capsys.readouterr().out)["output"] == str(output_path)


def test_compare_gsm8k_multi_cli_preserves_model_order_and_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "multi-comparison.json"
    experiment_config = {
        "project_seed": 42,
        "dataset": {"commit": "fixed", "sample_indices": [1, 2]},
        "prompt_template": "fixed prompt",
        "warmup_prompts": ["warmup"],
        "generation_config": {"temperature": 0.0},
        "backend_common_config": {"dtype": "bfloat16"},
    }
    model_paths = [tmp_path / f"model-{index}.jsonl" for index in range(3)]
    model_ids = [MODEL_08, MODEL_2, MODEL_4]
    patterns = ["001", "110"]
    for model_index, (model_path, model_id) in enumerate(
        zip(model_paths, model_ids, strict=True)
    ):
        records = []
        for sample_index, pattern in enumerate(patterns, start=1):
            correct = pattern[model_index] == "1"
            records.append(
                {
                    "sample_id": f"sample-{sample_index}",
                    "dataset_index": sample_index,
                    "question": f"question {sample_index}",
                    "prompt": f"prompt {sample_index}",
                    "model_id": model_id,
                    "reference_answer": str(sample_index),
                    "outcome": "correct" if correct else "incorrect",
                    "correct": correct,
                    "experiment_config": experiment_config,
                }
            )
        model_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    exit_code = cli.main(
        [
            "compare-gsm8k-multi",
            "--inputs",
            *(str(path) for path in model_paths),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert comparison["model_order"] == model_ids
    assert comparison["inputs"] == [str(path) for path in model_paths]
    assert len(comparison["ordered_pairs"]) == 3
    assert comparison["correctness_combinations"]["001"]["sample_ids"] == [
        "sample-1"
    ]
    assert comparison["oracle_progression"][-1]["oracle_correct_count"] == 2
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["output"] == str(output_path)
    assert stdout["model_order"] == model_ids
