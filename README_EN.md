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
- text-only, non-thinking Qwen3.5-0.8B, 2B, 4B, and 9B inference through vLLM;
- real output-token logprobs, length-normalized confidence, latency, and GPU-memory readings;
- complete decision traces with direct JSONL output;
- fixed GSM8K configurations for both 64 samples and all 1,319 test records, with warmup and automatic scoring;
- strict offline comparison of two or more ordered model results, including pairwise rescue and progressive-oracle analysis;
- CPU-only unit and CLI integration tests.

The current local model path is:

```text
Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B
```

All four Qwen3.5 local models are integrated for real inference. Cloud fallback remains future work.

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

Run the complete GSM8K test split (1,319 records) with the following four configurations, one model process at a time:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_0_8b_gsm8k_full.yaml \
  --output results/qwen35_0_8b_gsm8k_full.jsonl \
  --summary results/qwen35_0_8b_gsm8k_full_summary.json
```

Replace the configuration and output names with `qwen35_2b_gsm8k_full.yaml`, `qwen35_4b_gsm8k_full.yaml`, and `qwen35_9b_gsm8k_full.yaml`. Do not keep the models resident concurrently. After all four runs, compare them offline:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scalaflow compare-gsm8k-multi \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl \
           results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl \
           results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_full_comparison.json
```

Run 2B, 4B, and 9B sequentially with the same experiment contract, then align all four result sets offline:

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

CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_9b_gsm8k.yaml \
  --output results/qwen35_9b_gsm8k_64.jsonl \
  --summary results/qwen35_9b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow compare-gsm8k-multi \
  --inputs results/qwen35_0_8b_gsm8k_64.jsonl \
           results/qwen35_2b_gsm8k_64.jsonl \
           results/qwen35_4b_gsm8k_64.jsonl \
           results/qwen35_9b_gsm8k_64.jsonl \
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
- `configs/qwen35_9b_gsm8k.yaml` preserves the same experiment contract, pins the 9B BF16 revision, and uses `gpu_memory_utilization=0.90` for one GPU.
- `configs/qwen35_*_gsm8k_full.yaml` uses the same dataset commit, prompt, generation settings, and warmup procedure, selecting all 1,319 records in original order.

The GSM8K baseline uses the OpenAI test-set commit `3101c7d5072418e28b9008a6636bde82a006892c` and verifies SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`. Future model-size runs should reuse the same sample indices and prompt.

For the conditional logprob `l_i` of each generated token, confidence is currently defined as:

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL records preserve full token logprobs, the confidence method, latency, GPU-memory readings, and decision records. This confidence has not been calibrated as the probability of answer correctness.

GSM8K records keep `correct`, `incorrect`, `parse_failure`, and `inference_failure` separate; parse failures are not hidden inside answer errors. The confidence-correctness relationship on 64 samples is exploratory only and is not a formal statistical conclusion.

## Full GSM8K Evaluation

The full evaluation uses all 1,319 test records in their original order. The prompt, model revisions, BF16, non-thinking mode, generation parameters, and eight warmup requests are unchanged from the 64-sample baseline. Each model ran in a separate process on one RTX 3090; data and generated results remain Git-ignored.

| Model | Accuracy | incorrect | parse_failure | inference_failure | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 52.24% | 540 | 90 | 0 | 3962.16 | 3376.18 | 8968.62 |
| Qwen3.5-2B | 64.59% | 389 | 78 | 0 | 2954.64 | 2611.49 | 6165.59 |
| Qwen3.5-4B | 89.61% | 120 | 17 | 0 | 3956.88 | 3363.12 | 8033.22 |
| Qwen3.5-9B | 92.95% | 60 | 33 | 0 | 4698.13 | 4150.91 | 9594.98 |

| Model | Mean output tokens | Aggregate tokens/s | Peak GPU MiB | Point-biserial confidence correlation |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 165.87 | 41.86 | 6882.06 | 0.403 |
| Qwen3.5-2B | 121.09 | 40.98 | 6992.06 | 0.413 |
| Qwen3.5-4B | 122.30 | 30.91 | 11662.06 | 0.397 |
| Qwen3.5-9B | 148.36 | 31.58 | 22678.06 | 0.431 |

The aligned 4B/9B outcomes are: both correct 1,143, only 4B correct 39, only 9B correct 83, and both not correct 54. Of 137 not-correct 4B requests, 9B rescues 83 (60.58%); this includes 74/120 `incorrect`, 9/17 `parse_failure`, and 0/0 `inference_failure`. There are 50 samples where 9B is not correct but at least one smaller model is correct. All sample IDs, all 16 four-model correctness combinations, and their IDs are retained in the `per_request` and `correctness_combinations` fields of `results/qwen35_gsm8k_full_comparison.json`.

The bit order for correctness combinations is `0.8B, 2B, 4B, 9B`:

```text
0000:43  0001:51  0010:24  0011:210
0100:4   0101:20  0110:9   0111:269
1000:3   1001:4   1010:3   1011:129
1100:4   1101:8   1110:3   1111:535
```

Progressive post-hoc oracle accuracy is 52.24% for 0.8B (689 correct), 75.13% after adding 2B (991, +302), 92.87% after adding 4B (1,225, +234), and 96.74% after adding 9B (1,276, +51). The full GSM8K split separates all four models reliably. 4B and 9B remain close, but their full-set gap is 3.33 percentage points, larger than the gap seen on 64 samples. GSM8K is relatively easy for the larger models and should not be treated as fully saturated; a harder, separately specified evaluation set is recommended for finer scheduling analysis. The confidence correlations above are exploratory observations for this fixed run, not calibration or formal statistical conclusions.

## License

Copyright 2026 Pengfei_He. Licensed under the [Apache License 2.0](LICENSE).
