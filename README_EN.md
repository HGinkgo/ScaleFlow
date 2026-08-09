<div align="center">
  <h1>ScaleFlow</h1>
  <p>A lightweight framework for collaborative multi-scale language-model inference on resource-constrained edge devices</p>
  <p>
    <a href="README.md">中文版本</a> |
    <strong>English</strong>
  </p>
  <p>
    <a href="https://www.python.org/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"></a>
    <a href="https://pypi.org/project/vllm/0.26.0/"><img alt="vLLM 0.26.0" src="https://img.shields.io/badge/vLLM-0.26.0-4B8BBE"></a>
    <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B"><img alt="Qwen3.5" src="https://img.shields.io/badge/Model-Qwen3.5-7C3AED"></a>
    <a href="LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache"></a>
  </p>
</div>

## Overview

ScaleFlow is a lightweight, configuration-driven framework for studying request selection, result validation, and collaborative inference among local language models under limited GPU resources.

The project does not implement a new inference runtime or train model parameters. Real inference uses vLLM; MockBackend validates scheduling and result-recording logic deterministically.

## Features

- YAML-driven model, sampling, and experiment configuration;
- text-only, non-thinking inference with Qwen3.5-0.8B, 2B, 4B, and 9B;
- MockBackend, fixed-model, and confidence-cascade policies;
- JSONL result recording, GSM8K scoring, and offline multi-model alignment;
- CPU-only unit and CLI integration tests.

## Repository Layout

```text
ScaleFlow/
├── configs/                 # Experiment configurations
├── src/scaleflow/
│   ├── backends/            # MockBackend and VLLMBackend
│   ├── scheduler/           # Policies and execution flow
│   ├── baseline.py          # Scoring and result output
│   ├── offline.py           # Offline analysis
│   ├── routing.py           # Text-routing analysis
│   ├── performance.py       # Performance tests
│   ├── schemas.py           # Shared data structures
│   ├── config.py            # YAML loading
│   └── cli.py               # CLI entry point
├── tests/                   # Tests
├── environment.yml
└── pyproject.toml
```

Model caches, datasets, logs, and experiment results remain in Git-ignored directories.

## Installation

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev,analysis]'
```

Real-model execution additionally requires the pinned `.[vllm]` extra.

## Quickstart

Run the deterministic Mock scenario:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

Run the tests:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

Other experiments are exposed as `python -m scaleflow` subcommands. See `configs/` and the corresponding source modules for experiment protocols and parameters; experiment steps and results are intentionally not duplicated here.

The three-model offline cascade template is `configs/qwen35_gsm8k_three_cascade.yaml`.

## License

Copyright 2026 Pengfei_He. Licensed under the [Apache License 2.0](LICENSE).
