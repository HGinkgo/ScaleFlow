from collections.abc import Sequence
from math import exp, fsum, isfinite
import os
from time import perf_counter
from typing import Any

from scaleflow.backends.base import Backend, BackendUnavailableError
from scaleflow.schemas import InferenceRequest, ModelResponse


CONFIDENCE_METHOD = "exp(mean(output_token_logprobs))"


def extract_chosen_token_logprobs(
    token_ids: Sequence[int],
    positions: Any,
) -> list[float]:
    if not token_ids:
        raise ValueError("vLLM returned no output tokens")
    if positions is None:
        raise ValueError("vLLM did not return output token logprobs")
    if len(token_ids) != len(positions):
        raise ValueError("output token and logprob counts do not match")

    chosen: list[float] = []
    for token_id, position in zip(token_ids, positions, strict=True):
        try:
            value = float(position[token_id].logprob)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                f"logprob for generated token {token_id} is missing"
            ) from error
        if not isfinite(value):
            raise ValueError(f"logprob for generated token {token_id} is not finite")
        chosen.append(value)
    return chosen


def confidence_from_token_logprobs(token_logprobs: Sequence[float]) -> float:
    if not token_logprobs:
        raise ValueError("cannot compute confidence without token logprobs")
    if not all(isfinite(value) for value in token_logprobs):
        raise ValueError("cannot compute confidence from non-finite logprobs")
    return exp(fsum(token_logprobs) / len(token_logprobs))


def _gpu_memory_used_mb() -> float | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first_device = visible_devices.split(",", maxsplit=1)[0].strip()
    if not first_device.isdigit():
        return None

    pynvml_module = None
    try:
        import pynvml as pynvml_module

        pynvml_module.nvmlInit()
        handle = pynvml_module.nvmlDeviceGetHandleByIndex(int(first_device))
        used_bytes = pynvml_module.nvmlDeviceGetMemoryInfo(handle).used
        return used_bytes / (1024 * 1024)
    except Exception:
        return None
    finally:
        if pynvml_module is not None:
            try:
                pynvml_module.nvmlShutdown()
            except Exception:
                pass


class VLLMBackend(Backend):
    """Synchronous text-only vLLM backend for one local model."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        language_model_only: bool = True,
        enable_thinking: bool = False,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.4,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = True,
        seed: int = 42,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 64,
        logprobs: int = 1,
    ) -> None:
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if not language_model_only:
            raise ValueError("VLLMBackend currently supports text-only execution")
        if enable_thinking:
            raise ValueError("VLLMBackend currently supports non-thinking mode only")
        if logprobs < 1:
            raise ValueError("logprobs must be at least 1 for measured confidence")

        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise BackendUnavailableError(
                "vLLM is not installed; install the project with the vllm extra"
            ) from error

        self._model_id = model_id
        self._revision = revision
        self._vllm_version = vllm.__version__
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enforce_eager = enforce_eager
        self._enable_prefix_caching = enable_prefix_caching
        self._last_latency_ms: float | None = None
        self._gpu_memory_before_mb = _gpu_memory_used_mb()

        load_started = perf_counter()
        self._llm = LLM(
            model=model_id,
            revision=revision,
            language_model_only=True,
            trust_remote_code=False,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
        )
        self._model_load_latency_ms = (perf_counter() - load_started) * 1000
        self._gpu_memory_loaded_mb = _gpu_memory_used_mb()
        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            logprobs=logprobs,
            seed=seed,
        )

    def generate(self, request: InferenceRequest) -> ModelResponse:
        started = perf_counter()
        try:
            outputs = self._llm.chat(
                [{"role": "user", "content": request.prompt}],
                self._sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise ValueError("vLLM returned an unexpected number of completions")
            completion = outputs[0].outputs[0]
            token_logprobs = extract_chosen_token_logprobs(
                completion.token_ids,
                completion.logprobs,
            )
            confidence = confidence_from_token_logprobs(token_logprobs)
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000
            self._last_latency_ms = latency_ms
            return ModelResponse(
                model_id=self._model_id,
                text="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{type(error).__name__}: {error}",
                gpu_memory_used_mb=_gpu_memory_used_mb(),
            )

        latency_ms = (perf_counter() - started) * 1000
        self._last_latency_ms = latency_ms
        return ModelResponse(
            model_id=self._model_id,
            text=completion.text,
            confidence=confidence,
            latency_ms=latency_ms,
            success=True,
            error=None,
            token_logprobs=token_logprobs,
            confidence_method=CONFIDENCE_METHOD,
            gpu_memory_used_mb=_gpu_memory_used_mb(),
        )

    def get_model_info(self) -> dict[str, object]:
        gpu_memory_delta_mb = None
        if (
            self._gpu_memory_before_mb is not None
            and self._gpu_memory_loaded_mb is not None
        ):
            gpu_memory_delta_mb = (
                self._gpu_memory_loaded_mb - self._gpu_memory_before_mb
            )
        return {
            "backend": "vllm",
            "vllm_version": self._vllm_version,
            "model_id": self._model_id,
            "revision": self._revision,
            "available": True,
            "language_model_only": True,
            "enable_thinking": False,
            "dtype": self._dtype,
            "max_model_len": self._max_model_len,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "enforce_eager": self._enforce_eager,
            "enable_prefix_caching": self._enable_prefix_caching,
            "model_load_latency_ms": self._model_load_latency_ms,
            "gpu_memory_before_mb": self._gpu_memory_before_mb,
            "gpu_memory_loaded_mb": self._gpu_memory_loaded_mb,
            "gpu_memory_delta_mb": gpu_memory_delta_mb,
        }

    def health_check(self) -> bool:
        return self._llm is not None

    def estimate_latency(self, request: InferenceRequest) -> float:
        del request
        if self._last_latency_ms is None:
            raise BackendUnavailableError("no real latency observation is available yet")
        return self._last_latency_ms
