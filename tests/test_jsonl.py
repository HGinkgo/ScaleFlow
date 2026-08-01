import json

from scaleflow.scheduler.runner import write_results_jsonl
from scaleflow.schemas import DecisionRecord, InferenceResult


def test_jsonl_results_can_be_written_and_read(tmp_path) -> None:
    result = InferenceResult(
        request_id="request-1",
        final_answer="answer",
        final_model="model-a",
        total_latency_ms=10.0,
        escalation_count=0,
        decision_trace=[
            DecisionRecord(
                model_id="model-a",
                confidence=0.9,
                action="return",
                reason="confidence_threshold_met",
            )
        ],
        success=True,
        error=None,
        token_logprobs=[-0.1, -0.2],
        confidence_method="exp(mean(output_token_logprobs))",
        gpu_memory_used_mb=2048.0,
    )
    output = tmp_path / "result.jsonl"

    write_results_jsonl(output, [result])

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "request_id": "request-1",
            "final_answer": "answer",
            "final_model": "model-a",
            "total_latency_ms": 10.0,
            "escalation_count": 0,
            "decision_trace": [
                {
                    "model_id": "model-a",
                    "confidence": 0.9,
                    "action": "return",
                    "reason": "confidence_threshold_met",
                }
            ],
            "success": True,
            "error": None,
            "token_logprobs": [-0.1, -0.2],
            "confidence_method": "exp(mean(output_token_logprobs))",
            "gpu_memory_used_mb": 2048.0,
        }
    ]
