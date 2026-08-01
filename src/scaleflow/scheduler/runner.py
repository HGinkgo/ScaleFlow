from collections.abc import Iterable, Mapping
from dataclasses import asdict
import json
from pathlib import Path

from scaleflow.backends.base import Backend
from scaleflow.scheduler.policies import AlwaysModelPolicy, ConfidenceCascadePolicy
from scaleflow.schemas import DecisionRecord, InferenceRequest, InferenceResult


Policy = AlwaysModelPolicy | ConfidenceCascadePolicy


def run_request(
    request: InferenceRequest,
    backends: Mapping[str, Backend],
    policy: Policy,
) -> InferenceResult:
    decision_trace: list[DecisionRecord] = []
    total_latency_ms = 0.0

    for model_index, model_id in enumerate(policy.model_order):
        try:
            backend = backends[model_id]
        except KeyError as error:
            raise ValueError(f"backend not configured for model: {model_id}") from error

        response = backend.generate(request)
        total_latency_ms += response.latency_ms
        decision = policy.decide(response, model_index)
        decision_trace.append(decision)

        if decision.action == "return":
            return InferenceResult(
                request_id=request.request_id,
                final_answer=response.text,
                final_model=response.model_id,
                total_latency_ms=total_latency_ms,
                escalation_count=sum(
                    step.action == "escalate" for step in decision_trace
                ),
                decision_trace=decision_trace,
                success=response.success,
                error=response.error,
                token_logprobs=response.token_logprobs,
                confidence_method=response.confidence_method,
                gpu_memory_used_mb=response.gpu_memory_used_mb,
            )

    raise RuntimeError("policy exhausted its model order without returning a result")


def run_requests(
    requests: Iterable[InferenceRequest],
    backends: Mapping[str, Backend],
    policy: Policy,
) -> list[InferenceResult]:
    return [run_request(request, backends, policy) for request in requests]


def write_results_jsonl(
    output_path: str | Path,
    results: Iterable[InferenceResult],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
