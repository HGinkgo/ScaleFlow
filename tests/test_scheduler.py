from scaleflow.backends import MockBackend
from scaleflow.scheduler.policies import AlwaysModelPolicy, ConfidenceCascadePolicy
from scaleflow.scheduler.runner import run_request
from scaleflow.schemas import InferenceRequest


def make_backend(
    model_id: str,
    confidence: float,
    latency_ms: float = 10.0,
    *,
    success: bool = True,
) -> MockBackend:
    return MockBackend(
        model_id=model_id,
        responses={
            "request-1": {
                "text": f"answer from {model_id}" if success else "",
                "confidence": confidence,
                "latency_ms": latency_ms,
                "success": success,
                "error": None if success else "simulated failure",
            }
        },
    )


def test_always_model_policy_only_calls_configured_model() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {"model-b": make_backend("model-b", 0.5)},
        AlwaysModelPolicy("model-b"),
    )

    assert result.final_model == "model-b"
    assert [step.model_id for step in result.decision_trace] == ["model-b"]
    assert result.decision_trace[0].action == "return"
    assert result.escalation_count == 0


def test_runner_preserves_backend_measurements() -> None:
    backend = make_backend("model-a", 0.9)
    original_generate = backend.generate

    def generate_with_measurements(request):
        response = original_generate(request)
        return response.__class__(
            model_id=response.model_id,
            text=response.text,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            success=response.success,
            error=response.error,
            token_logprobs=[-0.1, -0.2],
            confidence_method="exp(mean(output_token_logprobs))",
            gpu_memory_used_mb=2048.0,
        )

    backend.generate = generate_with_measurements

    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {"model-a": backend},
        AlwaysModelPolicy("model-a"),
    )

    assert result.token_logprobs == [-0.1, -0.2]
    assert result.confidence_method == "exp(mean(output_token_logprobs))"
    assert result.gpu_memory_used_mb == 2048.0


def test_confidence_cascade_stops_on_high_confidence() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {
            "small": make_backend("small", 0.9),
            "large": make_backend("large", 0.95),
        },
        ConfidenceCascadePolicy(["small", "large"], confidence_threshold=0.8),
    )

    assert result.final_model == "small"
    assert result.decision_trace[0].reason == "confidence_threshold_met"
    assert result.escalation_count == 0


def test_confidence_cascade_escalates_on_low_confidence() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {
            "small": make_backend("small", 0.4, latency_ms=10.0),
            "large": make_backend("large", 0.9, latency_ms=20.0),
        },
        ConfidenceCascadePolicy(["small", "large"], confidence_threshold=0.8),
    )

    assert result.final_model == "large"
    assert result.final_answer == "answer from large"
    assert result.total_latency_ms == 30.0
    assert [step.action for step in result.decision_trace] == ["escalate", "return"]
    assert result.escalation_count == 1


def test_confidence_cascade_stops_at_largest_model() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {
            "small": make_backend("small", 0.2),
            "large": make_backend("large", 0.7),
        },
        ConfidenceCascadePolicy(["small", "large"], confidence_threshold=0.8),
    )

    assert result.final_model == "large"
    assert result.decision_trace[-1].action == "return"
    assert result.decision_trace[-1].reason == "maximum_model_reached"
    assert result.escalation_count == 1


def test_confidence_cascade_escalates_after_failure() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {
            "small": make_backend("small", 0.0, success=False),
            "large": make_backend("large", 0.9),
        },
        ConfidenceCascadePolicy(["small", "large"], confidence_threshold=0.8),
    )

    assert result.success is True
    assert result.decision_trace[0].reason == "model_failed"
    assert result.escalation_count == 1


def test_confidence_cascade_returns_failure_at_largest_model() -> None:
    result = run_request(
        InferenceRequest("request-1", "prompt"),
        {
            "small": make_backend("small", 0.0, success=False),
            "large": make_backend("large", 0.0, success=False),
        },
        ConfidenceCascadePolicy(["small", "large"], confidence_threshold=0.8),
    )

    assert result.success is False
    assert result.error == "simulated failure"
    assert result.final_model == "large"
    assert result.decision_trace[-1].reason == "maximum_model_failed"
    assert result.escalation_count == 1
