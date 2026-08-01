from abc import ABC, abstractmethod
from typing import Any

from scaleflow.schemas import InferenceRequest, ModelResponse


class BackendUnavailableError(RuntimeError):
    """Raised when a configured inference backend is not available."""


class Backend(ABC):
    @abstractmethod
    def generate(self, request: InferenceRequest) -> ModelResponse:
        """Generate one result for a request."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return backend and model metadata without running inference."""

    @abstractmethod
    def health_check(self) -> bool:
        """Report whether the backend can currently serve requests."""

    @abstractmethod
    def estimate_latency(self, request: InferenceRequest) -> float:
        """Estimate request service time in milliseconds."""
