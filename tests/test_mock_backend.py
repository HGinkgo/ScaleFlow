from importlib import import_module

from scaleflow.schemas import InferenceRequest, ModelResponse


def test_backend_exports_only_supported_backends() -> None:
    backends = import_module("scaleflow.backends")

    assert backends.__all__ == [
        "Backend",
        "BackendUnavailableError",
        "MockBackend",
        "VLLMBackend",
    ]


def test_mock_backend_returns_configured_result() -> None:
    backends = import_module("scaleflow.backends")
    backend_type = getattr(backends, "MockBackend")
    backend = backend_type(
        model_id="mock-qwen",
        responses={
            "easy": {
                "text": "controlled answer",
                "confidence": 0.95,
                "latency_ms": 12.5,
                "success": True,
                "error": None,
            }
        },
    )

    request = InferenceRequest("easy", "What is 1 + 1?", {"difficulty": "easy"})
    result = backend.generate(request)

    assert result == ModelResponse(
        model_id="mock-qwen",
        text="controlled answer",
        confidence=0.95,
        latency_ms=12.5,
        success=True,
        error=None,
    )
    assert backend.estimate_latency(request) == 12.5
    assert backend.get_model_info()["backend"] == "mock"
    assert backend.health_check() is True


def test_mock_backend_can_simulate_failure() -> None:
    backends = import_module("scaleflow.backends")
    backend = backends.MockBackend(
        model_id="mock-qwen",
        responses={
            "hard": {
                "text": "",
                "confidence": 0.0,
                "latency_ms": 9.0,
                "success": False,
                "error": "simulated failure",
            }
        },
    )

    response = backend.generate(InferenceRequest("hard", "hard prompt"))

    assert response.success is False
    assert response.error == "simulated failure"
