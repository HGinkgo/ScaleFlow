from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model_id: str
    text: str
    confidence: float
    latency_ms: float
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    model_id: str
    confidence: float
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class InferenceResult:
    request_id: str
    final_answer: str
    final_model: str
    total_latency_ms: float
    escalation_count: int
    decision_trace: list[DecisionRecord]
    success: bool
    error: str | None
