# ScaleFlow

中文版本：[README.md](README.md)

## Overview

ScaleFlow is a research framework for collaborative inference and scheduling across language models of different sizes in resource-constrained edge deployments. It studies how to route, validate, and escalate heterogeneous requests among local models while reserving cloud inference for cases where local quality is insufficient.

ScaleFlow does not implement a new low-level inference runtime and does not train or modify language-model parameters. Real-model execution uses established runtimes. `Qwen/Qwen3.5-0.8B` is now integrated through vLLM, while a deterministic MockBackend remains available for scheduler and data-flow validation.

## Thesis Background

The project supports a master's thesis on collaborative multi-scale language-model inference and scheduling optimization for resource-constrained edge environments. The central question is how to select a local model, decide when to escalate, decide when to return the current answer, and decide when cloud fallback is justified while balancing task quality, end-to-end latency, throughput, GPU resource consumption, and cloud-call ratio.

The research focuses on edge computing and cloud-edge deployment constraints. It does not cover wireless physical-layer modeling, model-parameter training, complex GPU memory management, or reinforcement learning introduced without experimental justification.

## Edge-Constrained Scenario

The target setting includes:

- requests with different difficulty, quality requirements, and latency deadlines;
- limited edge GPU memory and concurrent serving capacity;
- network round-trip latency, transfer overhead, and API cost between edge and cloud;
- low-cost local handling for simple requests and progressive escalation for harder ones;
- a cloud model used as fallback and a quality reference, not as the primary research subject.

## Multi-Scale Collaboration

The selected local Qwen3.5 model family is:

1. `Qwen/Qwen3.5-0.8B`
2. `Qwen/Qwen3.5-2B`
3. `Qwen/Qwen3.5-4B`
4. `Qwen/Qwen3.5-9B`

The intended quality-aware cascade is `0.8B -> 2B -> 4B -> 9B`. Model order and confidence thresholds are read from YAML rather than hard-coded in scheduler logic. DeepSeek-V4-Flash is planned as the cloud fallback. Its API integration is not implemented and is never called by the current Mock experiments.

## Implemented Features

Phase 2 currently includes:

- four shared data structures: `InferenceRequest`, `ModelResponse`, `DecisionRecord`, and `InferenceResult`;
- `AlwaysModelPolicy` for fixed-model execution;
- `ConfidenceCascadePolicy` for configuration-driven confidence escalation;
- a MockBackend with configurable text, confidence, simulated latency, success, and failure;
- deterministic easy, medium, and hard requests;
- complete decision traces, escalation counts, and accumulated simulated latency;
- direct JSONL result writing;
- CPU-only unit and CLI integration tests;
- synchronous single-model text inference with `Qwen/Qwen3.5-0.8B` on stable vLLM;
- explicit text-only, non-thinking execution with a pinned model revision;
- real selected-token logprobs, length-normalized confidence, end-to-end inference latency, and NVML GPU-memory readings;
- five fixed real-model smoke requests defined in YAML;
- an unimplemented SGLang placeholder.

The 2B, 4B, and 9B models, SGLang, DeepSeek API access, real multi-model cascades, concurrent queues, resource-aware scheduling, dataset evaluation, network simulation, formal benchmarks, plotting, and learning-based policies are not implemented yet.

## Repository Layout

```text
ScaleFlow/
├── configs/
│   ├── mock_qwen35.yaml
│   └── qwen35_0_8b_vllm.yaml
├── src/scaleflow/
│   ├── backends/
│   │   ├── base.py
│   │   ├── mock.py
│   │   ├── sglang.py
│   │   └── vllm.py
│   ├── scheduler/
│   │   ├── policies.py
│   │   └── runner.py
│   ├── schemas.py
│   ├── config.py
│   ├── cli.py
│   └── __main__.py
├── tests/
├── environment.yml
└── pyproject.toml
```

The generated `results/` directory and experiment artifacts are excluded from Git.

## Environment and Installation

Linux, conda, and Python 3.12 are required. Mock-only execution does not need a GPU. Real-model execution requires an NVIDIA GPU, a compatible driver, and the optional vLLM dependencies.

```bash
conda create -n scaleflow python=3.12 pip -y
conda run -n scaleflow python -m pip install -e '.[dev]'
```

Install the pinned stable vLLM version for real-model execution. It resolves the matching PyTorch and CUDA user-space runtime dependencies:

```bash
conda run -n scaleflow python -m pip install -e '.[dev,vllm]'
```

Alternatively, use the repository environment file:

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev]'
```

## Run the Mock Example

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

The command runs three fixed requests and writes JSONL output. The JSONL file is a generated artifact and is not versioned. Mock confidence and latency values are configuration inputs, not real-model measurements.

## Run Tests

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

Tests do not use a GPU, download models, or call remote APIs.

## Run Qwen3.5-0.8B

The following command exposes one GPU and runs five fixed text requests. The first run downloads the model into the Hugging Face cache. Model weights, caches, and generated results are excluded from Git.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

On networks that cannot reach Hugging Face directly, select a verified compatible endpoint through environment variables. The restricted network used for Phase 2 validation required:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

The backend accepts text only and fixes `enable_thinking` to `false`. At completion, the command prints vLLM version, model-load latency, and GPU memory before and after loading. Per-request JSONL contains generated text, inference latency, GPU memory, the decision record, and raw logprobs.

### Logprobs and confidence

The configuration requests `logprobs: 1`. For each actually selected output token, vLLM returns:

```text
l_i = log p(token_i | prompt, token_1, ..., token_{i-1})
```

All `l_i` values are preserved in `token_logprobs`. ScaleFlow currently uses the geometric mean output-token probability:

```text
confidence = exp((1 / T) * sum(l_i))
```

The JSONL `confidence_method` field records `exp(mean(output_token_logprobs))`. This normalization reduces the direct effect of output length, but it has not yet been calibrated against task correctness and must not be interpreted as the probability that an answer is correct.

`total_latency_ms` starts immediately before `LLM.chat` and ends after text, logprob, and confidence parsing; one-time model loading is excluded. The first request can include Triton kernel JIT, so the five-request smoke run is not a formal performance benchmark. `gpu_memory_used_mb` is the total NVML used-memory reading for the selected visible GPU; the CLI summary also reports the before/after loading delta.

## Configuration

[`configs/mock_qwen35.yaml`](configs/mock_qwen35.yaml) defines:

- the fixed random seed;
- scheduler policy;
- confidence threshold;
- model order;
- easy, medium, and hard requests;
- text, confidence, simulated latency, success, and error values for every model/request pair.

No real model path, GPU identifier, or API key is stored in the configuration. Future secrets must be provided through environment variables or an untracked local `.env` file.

[`configs/qwen35_0_8b_vllm.yaml`](configs/qwen35_0_8b_vllm.yaml) pins the model ID and revision, BF16 dtype, 4096-token context, GPU-memory utilization, non-thinking mode, deterministic sampling, logprob count, and five requests. GPU selection uses `CUDA_VISIBLE_DEVICES`; download endpoint selection uses `HF_ENDPOINT`. Neither is hard-coded in the project configuration.

## Development Status and Roadmap

Phase 0 (independent environment and skeleton), Phase 1 (minimal MockBackend scheduling flow), and Phase 2 (real Qwen3.5-0.8B single-model inference) are complete. Planned work will proceed incrementally:

1. integrate and validate Qwen3.5-2B, 4B, and 9B individually;
2. calibrate the relationship between real token logprobs and quality on labeled tasks;
3. add real fixed routing, dynamic cascade, Local-Only, Cloud-Only, and Oracle baselines;
4. incorporate queue state, GPU load, deadlines, and communication cost;
5. run reproducible experiments under varying load and network conditions and produce Pareto analyses;
6. evaluate simple learning-based methods only if rule-based policies show clear limitations.

These items are plans, not claims about currently implemented functionality or experimental results.

## Reproducibility

The Mock scenario is fully defined in YAML, with fixed request order, responses, thresholds, and simulated latencies. The real-model configuration pins the model revision, seed, inference parameters, and request order. Experiments should also record GPU model, driver, CUDA, PyTorch, vLLM, code commit, load latency, and per-request results. Model caches and generated raw results are excluded from Git by default and will use a separate documented release process if research artifacts are published.

## License Status

This project is currently under private research. An open-source license will be selected before public release.
