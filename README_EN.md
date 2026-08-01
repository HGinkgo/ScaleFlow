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
  </p>
</div>

## Overview

ScaleFlow is a lightweight, configuration-driven framework for studying request routing, result validation, and dynamic escalation across local language models under limited GPU resources.

The project does not implement a new low-level inference runtime and does not train or modify model parameters. Real-model execution uses vLLM, while MockBackend provides deterministic validation of scheduling and result-recording logic.

## Features

- YAML-based model, sampling, and scheduling configuration;
- `AlwaysModelPolicy` and confidence-driven `ConfidenceCascadePolicy`;
- deterministic MockBackend outputs, latency, confidence, and failure simulation;
- text-only, non-thinking Qwen3.5-0.8B, 2B, and 4B inference through vLLM;
- real output-token logprobs, length-normalized confidence, latency, and GPU-memory readings;
- complete decision traces with direct JSONL output;
- a fixed GSM8K test set, 64 sample IDs, warmup procedure, and automatic scoring baseline;
- strict offline comparison of two or more ordered model results, including pairwise rescue and progressive-oracle analysis;
- CPU-only unit and CLI integration tests.

The current local model path is:

```text
Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B
```

Qwen3.5-0.8B, 2B, and 4B are integrated for real inference. The 9B and cloud fallback remain future work.

## Repository Layout

```text
ScaleFlow/
├── configs/                 # Reproducible experiment configurations
├── src/scaleflow/
│   ├── backends/            # MockBackend and VLLMBackend
│   ├── scheduler/           # Policies and synchronous execution
│   ├── baseline.py          # GSM8K scoring, statistics, and output
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

Run the fixed GSM8K-64 single-model baseline:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_0_8b_gsm8k.yaml \
  --output results/qwen35_0_8b_gsm8k_64.jsonl \
  --summary results/qwen35_0_8b_gsm8k_64_summary.json
```

The first run downloads and verifies the official GSM8K `test` data. Raw data, per-sample JSONL, and the aggregate JSON are Git-ignored.

Run 2B and 4B sequentially with the same experiment contract, then align all three result sets offline:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_2b_gsm8k.yaml \
  --output results/qwen35_2b_gsm8k_64.jsonl \
  --summary results/qwen35_2b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_4b_gsm8k.yaml \
  --output results/qwen35_4b_gsm8k_64.jsonl \
  --summary results/qwen35_4b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow compare-gsm8k-multi \
  --inputs results/qwen35_0_8b_gsm8k_64.jsonl \
           results/qwen35_2b_gsm8k_64.jsonl \
           results/qwen35_4b_gsm8k_64.jsonl \
  --output results/qwen35_multi_comparison.json
```

Run each model in a separate process so they never reside in GPU memory together. The `--inputs` order defines ascending model capability. The comparator validates samples, prompts, reference answers, and shared experiment settings, and retains every correctness combination with sample IDs. `compare-gsm8k` remains available as the two-model compatibility entry point. Comparison is offline only and does not execute a cascade.

Run all tests:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

## Configuration and Measurements

- `configs/mock_qwen35.yaml` defines the deterministic four-scale Qwen3.5 Mock cascade;
- `configs/qwen35_0_8b_vllm.yaml` pins the Qwen3.5-0.8B revision, BF16 dtype, non-thinking mode, and deterministic sampling parameters.
- `configs/qwen35_0_8b_gsm8k.yaml` pins the GSM8K commit, SHA256, 64 sample indices, prompt, eight warmup requests, and generation parameters.
- `configs/qwen35_2b_gsm8k.yaml` preserves the same experiment contract while pinning the 2B revision and runtime memory settings.
- `configs/qwen35_4b_gsm8k.yaml` preserves the same experiment contract while pinning the 4B revision and its runtime memory settings.

The GSM8K baseline uses the OpenAI test-set commit `3101c7d5072418e28b9008a6636bde82a006892c` and verifies SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`. Future model-size runs should reuse the same sample indices and prompt.

For the conditional logprob `l_i` of each generated token, confidence is currently defined as:

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL records preserve full token logprobs, the confidence method, latency, GPU-memory readings, and decision records. This confidence has not been calibrated as the probability of answer correctness.

GSM8K records keep `correct`, `incorrect`, `parse_failure`, and `inference_failure` separate; parse failures are not hidden inside answer errors. The confidence-correctness relationship on 64 samples is exploratory only and is not a formal statistical conclusion.

## Status

The project skeleton, deterministic Mock scheduling flow, fixed Qwen3.5-0.8B, 2B, and 4B GSM8K-64 baselines, and generic multi-model offline alignment are implemented. Real multi-model cascades, cloud fallback, concurrent queues, and resource-aware scheduling are not yet implemented.

| Model | Correct | incorrect / parse / inference | Mean / P50 / P95 latency (ms) | Aggregate tokens/s | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 33/64 | 28 / 3 / 0 | 3761.83 / 3389.28 / 7101.28 | 41.52 | 6882.06 |
| Qwen3.5-2B | 40/64 | 20 / 4 / 0 | 2723.27 / 2486.30 / 4174.76 | 40.52 | 6992.06 |
| Qwen3.5-4B | 59/64 | 4 / 1 / 0 | 3878.75 / 3457.85 / 7103.72 | 30.87 | 11664.06 |

The 4B model rescued 20 of the 24 requests that 2B did not answer correctly (83.33%): 16/20 `incorrect`, 4/4 `parse_failure`, and 0/0 `inference_failure`. The two-model oracle was 49/64 (76.56%); adding 4B raised it to 60/64 (93.75%), an increment of 11 samples or 17.19 percentage points. Non-monotonic cases included nine where 2B was not correct but 0.8B was correct, and one where 4B was not correct but 2B was correct; the latter was also the only 4B failure rescued by either smaller model. These are offline upper-bound results, not an executed cascade.

The 0.8B/2B runs used `gpu_memory_utilization=0.25`, while 4B used `0.45`. vLLM reported approximately 1.53/3.36, 3.63/1.24, and 7.99/1.58 GiB of weights/KV cache; 4B peak activation was about 0.96 GiB. The 4B confidence-correctness point-biserial correlation was 0.2007, or 0.0598 after excluding parse failures. All accuracy, latency, and confidence observations on these fixed 64 samples are exploratory, not formal performance or statistical conclusions.

## License

Copyright 2026 Pengfei_He. Licensed under the [Apache License 2.0](LICENSE).
