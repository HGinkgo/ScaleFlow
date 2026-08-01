from typing import Any

from scaleflow.backends.base import Backend
from scaleflow.schemas import InferenceRequest, ModelResponse


class MockBackend(Backend):
    """Deterministic CPU-only backend for framework and policy tests."""

    def __init__(
        self,
        model_id: str = "mock-model",
        responses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id cannot be empty")
        self._model_id = model_id
        self._responses = responses or {}

    def generate(self, request: InferenceRequest) -> ModelResponse:
        profile = self._responses.get(request.request_id)
        if profile is None:
            return ModelResponse(
                model_id=self._model_id,
                text="",
                confidence=0.0,
                latency_ms=0.0,
                success=False,
                error=f"no mock response configured for request {request.request_id}",
            )
        return ModelResponse(
            model_id=self._model_id,
            text=str(profile.get("text", "")),
            confidence=float(profile.get("confidence", 0.0)),
            latency_ms=float(profile.get("latency_ms", 0.0)),
            success=bool(profile.get("success", True)),
            error=profile.get("error"),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {"backend": "mock", "model_id": self._model_id, "available": True}

    def health_check(self) -> bool:
        return True

    def estimate_latency(self, request: InferenceRequest) -> float:
        profile = self._responses.get(request.request_id, {})
        return float(profile.get("latency_ms", 0.0))
