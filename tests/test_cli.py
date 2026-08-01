import os
from pathlib import Path
import subprocess
import sys


MODEL_08 = "Qwen/Qwen3.5-0.8B"
MODEL_2 = "Qwen/Qwen3.5-2B"
MODEL_4 = "Qwen/Qwen3.5-4B"
MODEL_9 = "Qwen/Qwen3.5-9B"


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

    import json

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
