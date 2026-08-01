# ScaleFlow

中文版本：[README.md](README.md)

## Overview

ScaleFlow is a research framework for collaborative inference and scheduling across language models of different sizes in resource-constrained edge deployments. It studies how to route, validate, and escalate heterogeneous requests among local models while reserving cloud inference for cases where local quality is insufficient.

ScaleFlow does not implement a new low-level inference runtime and does not train or modify language-model parameters. Future real-model execution will use established runtimes such as vLLM or SGLang. The current phase uses a deterministic MockBackend to validate scheduling behavior and the experiment data flow.

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

Phase 1 currently includes:

- four shared data structures: `InferenceRequest`, `ModelResponse`, `DecisionRecord`, and `InferenceResult`;
- `AlwaysModelPolicy` for fixed-model execution;
- `ConfidenceCascadePolicy` for configuration-driven confidence escalation;
- a MockBackend with configurable text, confidence, simulated latency, success, and failure;
- deterministic easy, medium, and hard requests;
- complete decision traces, escalation counts, and accumulated simulated latency;
- direct JSONL result writing;
- CPU-only unit and CLI integration tests;
- dependency-free placeholders for future vLLM and SGLang integrations.

Real model loading, measured confidence, GPU scheduling, dataset evaluation, resource monitoring, network simulation, DeepSeek API access, formal benchmarks, plotting, and learning-based policies are not implemented yet.

## Repository Layout

```text
ScaleFlow/
├── configs/
│   └── mock_qwen35.yaml
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

Linux, conda, and Python 3.12 are required. Phase 1 does not require CUDA, PyTorch, vLLM, or SGLang.

```bash
conda create -n scaleflow python=3.12 pip -y
conda run -n scaleflow python -m pip install -e '.[dev]'
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

## Configuration

[`configs/mock_qwen35.yaml`](configs/mock_qwen35.yaml) defines:

- the fixed random seed;
- scheduler policy;
- confidence threshold;
- model order;
- easy, medium, and hard requests;
- text, confidence, simulated latency, success, and error values for every model/request pair.

No real model path, GPU identifier, or API key is stored in the configuration. Future secrets must be provided through environment variables or an untracked local `.env` file.

## Development Status and Roadmap

Phase 0 (independent environment and project skeleton) and Phase 1 (minimal MockBackend scheduling flow) are complete. Planned work will proceed incrementally:

1. integrate a real inference backend and profile individual models;
2. obtain real token log probabilities and calibrate confidence thresholds;
3. add fixed routing, dynamic cascade, Local-Only, Cloud-Only, and Oracle baselines;
4. incorporate queue state, GPU load, deadlines, and communication cost;
5. run reproducible experiments under varying load and network conditions and produce Pareto analyses;
6. evaluate simple learning-based methods only if rule-based policies show clear limitations.

These items are plans, not claims about currently implemented functionality or experimental results.

## Reproducibility

The current Mock scenario is fully defined in YAML. Request order, model responses, thresholds, and simulated latencies are fixed, so repeated runs with the same configuration produce identical results. Formal experiments will record the code commit, environment versions, configuration, random seed, dataset version, model version, and per-request outputs. Generated raw results are excluded from Git by default and will use a separate documented release process if research artifacts are published.

## License Status

This project is currently under private research. An open-source license will be selected before public release.
