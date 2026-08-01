from scaleflow.backends.base import (
    Backend,
    BackendUnavailableError,
)
from scaleflow.schemas import InferenceRequest, ModelResponse


class SGLangBackend(Backend):
    """Phase 0 interface placeholder; it does not import or start SGLang."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def _unavailable(self) -> BackendUnavailableError:
        return BackendUnavailableError("SGLang integration is deferred beyond Phase 0")

    def generate(self, request: InferenceRequest) -> ModelResponse:
        del request
        raise self._unavailable()

    def get_model_info(self) -> dict[str, object]:
        return {"backend": "sglang", "model_id": self._model_id, "available": False}

    def health_check(self) -> bool:
        return False

    def estimate_latency(self, request: InferenceRequest) -> float:
        del request
        raise self._unavailable()
