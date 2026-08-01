<div align="center">
  <h1>ScaleFlow</h1>
  <p>Collaborative multi-scale language-model inference for resource-constrained edge environments</p>
  <p>
    <a href="README.md">简体中文</a> |
    <strong>English</strong>
  </p>
  <p>
    <a href="https://www.python.org/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"></a>
    <a href="https://pypi.org/project/vllm/0.26.0/"><img alt="vLLM 0.26.0" src="https://img.shields.io/badge/vLLM-0.26.0-4B8BBE"></a>
    <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B"><img alt="Qwen3.5" src="https://img.shields.io/badge/Model-Qwen3.5-7C3AED"></a>
    <a href="LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache"></a>
    <a href="https://github.com/HGinkgo/ScaleFlow/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/HGinkgo/ScaleFlow?style=flat&logo=github"></a>
    <a href="https://github.com/HGinkgo/ScaleFlow/commits/main"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/HGinkgo/ScaleFlow?style=flat"></a>
  </p>
</div>

## Overview

ScaleFlow is a lightweight, configuration-driven framework for studying request routing, result validation, and dynamic escalation across local language models under limited GPU resources.

The project does not implement a new low-level inference runtime and does not train or modify model parameters. Real-model execution uses vLLM, while MockBackend provides deterministic validation of scheduling and result-recording logic.

## Features

- YAML-based model, sampling, and scheduling configuration;
- `AlwaysModelPolicy` and confidence-driven `ConfidenceCascadePolicy`;
- deterministic MockBackend outputs, latency, confidence, and failure simulation;
- text-only, non-thinking Qwen3.5-0.8B inference through vLLM;
- real output-token logprobs, length-normalized confidence, latency, and GPU-memory readings;
- complete decision traces with direct JSONL output;
- CPU-only unit and CLI integration tests.

The current local model path is:

```text
Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B
```

Only Qwen3.5-0.8B is integrated for real inference. The remaining models and cloud fallback are future work.

## Repository Layout

```text
ScaleFlow/
├── configs/                 # Reproducible experiment configurations
├── src/scaleflow/
│   ├── backends/            # MockBackend and VLLMBackend
│   ├── scheduler/           # Policies and synchronous execution
│   ├── schemas.py           # Shared data structures
│   ├── config.py            # YAML loading
│   └── cli.py               # Command-line entry point
├── tests/                   # CPU-only tests
├── environment.yml
└── pyproject.toml
```

Model caches, datasets, and generated results are stored in Git-ignored directories.

## Installation

ScaleFlow uses Python 3.12 in an isolated conda environment:

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev]'
```

Install the pinned vLLM version for real-model execution:

```bash
conda run -n scaleflow python -m pip install -e '.[dev,vllm]'
```

## Quickstart

Run the deterministic Mock scenario:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

Run real Qwen3.5-0.8B inference:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

Select the GPU with `CUDA_VISIBLE_DEVICES`. Model ID, revision, and generation parameters come from YAML; a download endpoint can be selected with the `HF_ENDPOINT` environment variable.

Run all tests:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

## Configuration and Measurements

- `configs/mock_qwen35.yaml` defines the deterministic four-scale Qwen3.5 Mock cascade;
- `configs/qwen35_0_8b_vllm.yaml` pins the Qwen3.5-0.8B revision, BF16 dtype, non-thinking mode, and deterministic sampling parameters.

For the conditional logprob `l_i` of each generated token, confidence is currently defined as:

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL records preserve full token logprobs, the confidence method, latency, GPU-memory readings, and decision records. This confidence has not been calibrated as the probability of answer correctness.

## Status

The project skeleton, deterministic Mock scheduling flow, and real Qwen3.5-0.8B single-model inference path are implemented. Real multi-model cascades, labeled-dataset evaluation, cloud fallback, concurrent queues, and resource-aware scheduling are not yet implemented.

Published results should record the configuration, random seed, model revision, runtime environment, and code version. Smoke-test observations are not presented as benchmark conclusions.

## License

Copyright 2026 Pengfei_He. Licensed under the [Apache License 2.0](LICENSE).
