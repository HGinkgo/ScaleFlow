from collections.abc import Sequence

from scaleflow.schemas import DecisionRecord, ModelResponse


class AlwaysModelPolicy:
    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id cannot be empty")
        self.model_order = (model_id,)

    def decide(self, response: ModelResponse, model_index: int) -> DecisionRecord:
        del model_index
        return DecisionRecord(
            model_id=response.model_id,
            confidence=response.confidence,
            action="return",
            reason="fixed_model_policy" if response.success else "fixed_model_failed",
        )


class ConfidenceCascadePolicy:
    def __init__(
        self,
        model_order: Sequence[str],
        confidence_threshold: float,
    ) -> None:
        if not model_order:
            raise ValueError("model_order cannot be empty")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model_order = tuple(model_order)
        self.confidence_threshold = float(confidence_threshold)

    def decide(self, response: ModelResponse, model_index: int) -> DecisionRecord:
        is_last_model = model_index == len(self.model_order) - 1
        if not response.success:
            action = "return" if is_last_model else "escalate"
            reason = "maximum_model_failed" if is_last_model else "model_failed"
        elif response.confidence >= self.confidence_threshold:
            action = "return"
            reason = "confidence_threshold_met"
        elif is_last_model:
            action = "return"
            reason = "maximum_model_reached"
        else:
            action = "escalate"
            reason = "confidence_below_threshold"

        return DecisionRecord(
            model_id=response.model_id,
            confidence=response.confidence,
            action=action,
            reason=reason,
        )
