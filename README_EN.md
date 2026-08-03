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
- deterministic holdout confidence validation, Pareto threshold search, and offline cascade replay;
- CPU-only offline routing analysis using a manual rule and TF-IDF/logistic regression from request text;
- real closed-loop concurrency benchmarking with a vLLM OpenAI server and an asynchronous streaming client;
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
│   ├── offline.py           # Confidence analysis and offline cascade replay
│   ├── routing.py           # CPU-only pre-inference text routing analysis
│   ├── performance.py       # Closed-loop load and streaming timing
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

Install the lightweight analysis dependency for Phase 10:

```bash
conda run -n scaleflow python -m pip install -e '.[analysis,dev]'
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
  python -m scaleflow compare-gsm8k-multi \
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

Validate confidence, search thresholds on the development split, and evaluate the frozen policy once on the holdout split:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  analyze-gsm8k-confidence --config configs/qwen35_gsm8k_offline.yaml \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_confidence_development.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  search-gsm8k-cascade --config configs/qwen35_gsm8k_offline.yaml \
  --confidence-report results/qwen35_gsm8k_confidence_development.json \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_cascade_policy.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  evaluate-gsm8k-cascade --config configs/qwen35_gsm8k_offline.yaml \
  --policy results/qwen35_gsm8k_cascade_policy.json \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_cascade_evaluation.json
```

Thresholds are selected only on 660 development records. The 659-record holdout is evaluated once after freezing the policy; a `.evaluated` marker next to the policy rejects repeated execution. These commands do not load models or use a GPU.

Run one closed-loop serving benchmark (run the four models sequentially):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow python -m scaleflow \
  run-gsm8k-concurrency --config configs/qwen35_gsm8k_concurrency.yaml \
  --model-id Qwen/Qwen3.5-0.8B \
  --reference results/qwen35_0_8b_gsm8k_full.jsonl \
  --reference-summary results/qwen35_0_8b_gsm8k_full_summary.json \
  --output results/phase9_qwen35_0_8b_concurrency.jsonl \
  --summary results/phase9_qwen35_0_8b_concurrency_summary.json \
  --server-log results/phase9_qwen35_0_8b_concurrency_server.log
```

Run the one-shot pre-inference text-routing analysis using the Phase 8 split and the Phase 7 full results:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  analyze-gsm8k-routing \
  --config configs/qwen35_gsm8k_routing.yaml \
  --split-report results/phase8_confidence_development.json \
  --inputs results/phase7_persistent_qwen35_2b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_4b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_9b_gsm8k_full.jsonl \
  --output results/phase10_text_routing.json
```

The command fits both lightweight routers on 660 development records, freezes them, and evaluates once on the 659-record exploratory split already used in Phase 8. Results stay Git-ignored; do not rerun or tune against the evaluation split.

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
- `configs/qwen35_gsm8k_offline.yaml` fixes the holdout split, confidence bootstrap, threshold grid, and random-acceptance seed set.
- `configs/qwen35_gsm8k_concurrency.yaml` fixes 128 requests, all four revisions, a shared `gpu_memory_utilization=0.90`, warmups, and five concurrency levels.
- `configs/qwen35_gsm8k_routing.yaml` fixes the Phase 8 SHA256 split, text features, rule threshold grid, TF-IDF/logistic-regression parameters, and matched-random seeds.

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

Progressive post-hoc oracle accuracy is 52.24% for 0.8B (689 correct), 75.13% after adding 2B (991, +302), 92.87% after adding 4B (1,225, +234), and 96.74% after adding 9B (1,276, +51). The full GSM8K split separates all four models reliably. 4B and 9B remain close, but their full-set gap is 3.34 percentage points, larger than the gap seen on 64 samples. GSM8K is relatively easy for the larger models and should not be treated as fully saturated; a harder, separately specified evaluation set is recommended for finer scheduling analysis. The confidence correlations above are exploratory observations for this fixed run, not calibration or formal statistical conclusions.

## Offline Confidence Cascade

The 1,319 records are deterministically split into 660 development and 659 holdout records with `SHA256(seed=42, sample_id)`. Development AUROC is 0.701, 0.743, and 0.779 for 0.8B, 2B, and 4B. Every 95% bootstrap lower bound exceeds 0.5; the lower bounds for the bottom-20% error lift and AURC improvement are also positive, so all three intermediate models remain in the chain.

Each stage has 51 candidates, including explicit always-accept and always-escalate boundaries. Of 132,651 threshold combinations, 120 match or exceed development-set 9B accuracy. The minimum-mean-latency feasible thresholds are 0.9256, 0.9414, and 0.9421, giving the same 607/660 development accuracy as 9B.

The frozen policy scores 611/659 (92.72%) on the holdout, below standalone 9B at 619/659 (93.93%). Final failures are 39 `incorrect`, 9 `parse_failure`, and 0 `inference_failure`. It invokes 9B for 65.86% of requests, reducing 9B calls by 34.14%, but estimated sequential latency is 13,744/12,189/27,033 ms (mean/P50/P95), versus 4,611/4,154/8,912 ms for standalone 9B.

A confidence-independent baseline over 1,000 fixed seeds matches the per-stage acceptance and 9B invocation rates. Its mean accuracy is 90.74%, with a randomization interval of 89.38% to 92.11%. Confidence improves accuracy by 1.97 percentage points on average, with a randomization-difference interval of 0.61 to 3.34 points. Confidence ranking is therefore informative, but this threshold policy neither preserves 9B holdout accuracy nor reduces sequential end-to-end latency.

Cascade latency is an offline sum of recorded single-model latencies. It excludes model loading, switching, queueing, and concurrency effects. Randomization intervals describe fixed-holdout policy variation, not population-level statistical confidence intervals.

## Concurrent Serving Evaluation

The benchmark deterministically selects the same 128 GSM8K requests with `seed=42` and runs each model separately on one RTX 3090. It performs eight startup warmups and one unmeasured wave at each concurrency before closed-loop levels 1, 2, 4, 8, and 16. Generation stops naturally at EOS; `max_tokens=384` is only a safety cap. TTFT starts at the first non-empty text token observed by the client. TPOT is a client-observed mean that includes local HTTP streaming and event-processing overhead.

| Model | C1 req/s | C2 | C4 | C8 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.259 | 0.483 | 0.878 | 1.725 | 3.063 |
| Qwen3.5-2B | 0.364 | 0.700 | 1.340 | 2.568 | 4.502 |
| Qwen3.5-4B | 0.251 | 0.474 | 0.948 | 1.783 | 3.037 |
| Qwen3.5-9B | 0.215 | 0.413 | 0.780 | 1.472 | 2.635 |

| Model at C16 | Output tok/s | Mean latency ms | P95 ms | TTFT P95 ms | Mean TPOT ms | Parsed-answer consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 489.9 | 4403 | 8344 | 132 | 27.08 | 87.50% |
| Qwen3.5-2B | 521.4 | 3279 | 5964 | 182 | 27.60 | 88.28% |
| Qwen3.5-4B | 372.5 | 4534 | 10058 | 326 | 35.98 | 97.66% |
| Qwen3.5-9B | 377.7 | 5173 | 9964 | 541 | 35.12 | 97.66% |

| Model | Weights GiB | KV Cache GiB | KV Cache tokens | Peak NVML MiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.53 | 18.68 | 877714 | 22807 |
| Qwen3.5-2B | 3.63 | 16.57 | 778532 | 22819 |
| Qwen3.5-4B | 7.99 | 12.21 | 221476 | 22883 |
| Qwen3.5-9B | 16.80 | 3.39 | 61440 | 22747 |

All 2,560 measured requests succeeded without OOM, KV-cache exhaustion, or queued requests. Maximum logged KV-cache use was 2.1%/2.4%/8.5%/30.6% for 0.8B/2B/4B/9B. With the same memory utilization target, vLLM allocates memory left after weights to the KV cache and runtime workspace, so the roughly 22.2 GiB NVML peaks are not model-weight sizes.

2B beats 0.8B in request throughput and mean latency at every tested concurrency. Combined with full-GSM8K quality, 0.8B has no practical serving advantage within C1-C16. 4B delivers about 15%-21% more request throughput than 9B while retaining close quality, making it the primary edge quality tier; 9B remains the local high-quality fallback. At C16, 9B TTFT P95 reaches 541 ms, but there is no waiting queue or KV-cache pressure, which points to compute contention under batching. The recommended core chain is `2B -> 4B -> 9B`, with 0.8B retained only as a control baseline.

At C1, parsed answers and full text both match the single-model baseline exactly. At higher concurrency, GPU batching changes numerical execution paths enough to reduce exact-text consistency even with greedy parameters and a fixed seed. Parsed final-answer consistency is therefore the primary reproducibility metric; full-text consistency remains the strict metric.

## Pre-inference Text Routing (Phase 10, Exploratory)

This analysis does not run models. It reads the Phase 7 full 2B/4B/9B results and reuses the exact Phase 8 `SHA256(seed=42, sample_id)` split: 660 development records and 659 exploratory evaluation records. The evaluation records were already used in Phase 8, so these are exploratory offline results, not an independent test. Router inputs contain only question text and character, word, number, operator, and keyword counts; outputs, confidence, correctness, and measured latency are excluded.

| Method | Accuracy | incorrect / parse / inference | none_correct | 2B / 4B / 9B calls | Historical mean / P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always 2B | 62.67% | 207 / 39 / 0 | 24 | 100 / 0 / 0% | 3053 / 2679 / 6864 |
| Always 4B | 89.83% | 58 / 9 / 0 | 24 | 0 / 100 / 0% | 3987 / 3361 / 8115 |
| Always 9B | 93.93% | 29 / 11 / 0 | 24 | 0 / 0 / 100% | 4611 / 4154 / 8912 |
| Manual rule | 81.34% | 101 / 22 / 0 | 24 | 50.53 / 4.40 / 45.07% | 3915 / 3367 / 8580 |
| TF-IDF + logistic regression | 70.71% | 164 / 29 / 0 | 24 | 73.14 / 23.07 / 3.79% | 3361 / 2933 / 7356 |
| Post-hoc lowest correct model | 96.36% | 17 / 7 / 0 | 24 | 62.67 / 29.44 / 7.89% | 3545 / 2954 / 7868 |

Historical latency replays only the selected model's natural-completion latency. Adding measured router prediction cost gives 3915.41 ms mean for the manual rule (0.032 ms prediction overhead) and 3361.47 ms for TF-IDF (0.547 ms overhead). Training took 43.9 ms and 10.42 s respectively. Neither router meets the pre-set quality target of no more than a one-point drop from 9B on evaluation: manual routing is -12.59 points and TF-IDF is -23.22 points.

For 1,000 fixed-seed matched-quota random baselines, the manual rule's random accuracy mean is 77.93% (76.02%–79.82% interval); the rule is +3.40 points with a +1.52 to +5.31 point difference interval. TF-IDF's matched random mean is 70.11% (68.44%–71.78%), with a +0.60 point difference and an interval of -1.06 to +2.28. The manual rule has a weak useful ranking signal but insufficient quality; TF-IDF falls from 96.52% on development to 70.71% on evaluation, showing unstable generalization. No evaluation retuning was performed. Requests where all three models were incorrect are reported separately as `none_correct`. The conclusion is that lightweight routing from request text alone is insufficient to replace the quality baseline.

## License

Copyright 2026 Pengfei_He. Licensed under the [Apache License 2.0](LICENSE).
