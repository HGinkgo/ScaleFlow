# Phase 10 Pre-Inference Text Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reproducible CPU-only CLI that trains two text-only routers on the frozen Phase 8 development split and evaluates them once on the previously used exploratory split.

**Architecture:** Keep the implementation in a focused `routing.py` module. The CLI loads and validates the three Phase 7 model result files, verifies the Phase 8 split contract, fits the rule and TF-IDF/logistic-regression routers only on the 660 development records, then computes direct-routing, matched-random, constant-model, and oracle summaries for the 659 evaluation records. No Backend or online scheduler changes are needed.

**Tech Stack:** Python standard library, existing GSM8K comparison helpers, scikit-learn `TfidfVectorizer` and `LogisticRegression`, YAML configuration, pytest.

---

### Task 1: Add the analysis dependency and frozen configuration

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/qwen35_gsm8k_routing.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test**

Add a test that loads the new YAML and asserts the three fixed model IDs, Phase 8 split method/seed/count, rule weights, fixed TF-IDF/logistic-regression settings, and the 1,000-seed random baseline.

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest tests/test_config.py -q`

Expected: FAIL because the routing configuration does not yet exist.

- [ ] **Step 3: Add the minimal dependency and YAML contract**

Add an `analysis` optional dependency containing the tested pinned scikit-learn release. The YAML must contain only public dataset identifiers, Phase 8 split path, model IDs, fixed rule weights/candidate thresholds, fixed TF-IDF and logistic-regression parameters, and random seeds. It must not contain model paths or secrets.

- [ ] **Step 4: Run the focused test**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Install and verify the analysis dependency**

Run: `conda run -n scaleflow python -m pip install -e '.[analysis,dev]' && conda run -n scaleflow python -c 'import sklearn; print(sklearn.__version__)'`

Expected: the pinned scikit-learn version imports successfully without installing model or CUDA packages.

### Task 2: Build the frozen input contract and text-only feature extraction

**Files:**
- Create: `src/scaleflow/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write failing tests for Phase 8 split verification and feature isolation**

Cover: recomputed SHA256 ordering exactly matches the saved development/evaluation ID lists; a changed saved ID is rejected; feature extraction accepts only the question text and returns deterministic character, word, number, operator, and keyword counts; model outputs and latency fields cannot be passed as features.

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest tests/test_routing.py -q`

Expected: FAIL because the routing module and functions do not yet exist.

- [ ] **Step 3: Implement the minimal contract and feature functions**

Implement stable functions `verify_phase8_split(sample_ids: Sequence[str], split_report: dict[str, Any], seed: int, development_count: int) -> tuple[list[str], list[str]]`, `extract_request_features(question: str) -> dict[str, float]`, and `build_routing_examples(records_by_model: Sequence[Sequence[dict[str, Any]]], model_order: Sequence[str], development_ids: Sequence[str]) -> list[dict[str, Any]]`.

Use `compare_model_records` for common sample/prompt/reference/config validation, enforce exactly the three ordered models, preserve all per-model outcomes, and construct labels `2B`, `4B`, or `9B` plus `none_correct`. `extract_request_features` must have no access to result records.

- [ ] **Step 4: Run the focused tests**

Expected: PASS, including the mismatch rejection test.

### Task 3: Implement the development-only rule and TF-IDF routers

**Files:**
- Modify: `src/scaleflow/routing.py`
- Modify: `tests/test_routing.py`

- [ ] **Step 1: Write failing router tests**

Cover deterministic rule predictions, development-only threshold selection, fixed TF-IDF/logistic-regression fitting, class labels limited to `2B/4B/9B`, deterministic repeated predictions, and non-monotonic/`none_correct` labels retained.

- [ ] **Step 2: Run the focused tests to verify failure**

Expected: FAIL on missing router implementations.

- [ ] **Step 3: Implement the rule router**

Use fixed YAML weights to calculate a complexity score. Search only the configured small threshold grid on development examples. Select candidates by the approved development objective: accuracy at least development 9B accuracy minus one percentage point, then minimum historical direct-routing latency, then label macro-F1 and lower 9B call rate. If no candidate is feasible, select the highest development accuracy and set an explicit `development_target_met=false` flag.

- [ ] **Step 4: Implement the learning router**

Fit `TfidfVectorizer` and `LogisticRegression` only on development questions and labels, using the fixed YAML settings and `random_state=42`. Do not tune on evaluation records and do not expose `predict_proba` as calibrated correctness.

- [ ] **Step 5: Run the focused tests**

Expected: PASS and no test imports GPU or calls a model backend.

### Task 4: Implement routing metrics and matched random baselines

**Files:**
- Modify: `src/scaleflow/routing.py`
- Modify: `tests/test_routing.py`

- [ ] **Step 1: Write failing metric tests**

Cover constant-model baselines, lowest-correct oracle labels, outcome and `none_correct` counts, call ratios, mean/P50/P95 historical latency, latency with per-request router overhead, confusion matrices, class metrics, and exact-call-count random shuffles across fixed seeds.

- [ ] **Step 2: Run the focused tests to verify failure**

Expected: FAIL on missing metric functions.

- [ ] **Step 3: Implement metric functions**

Add `evaluate_assignments(assignments: Sequence[str], records_by_model: dict[str, dict[str, dict[str, Any]]], router_overhead_ms: Sequence[float]) -> dict[str, Any]` and `matched_random_baseline(predicted_models: Sequence[str], records_by_model: dict[str, dict[str, dict[str, Any]]], seeds: Sequence[int]) -> dict[str, Any]`.

Compute two latency views: selected model history only, and selected model history plus measured single-request prediction overhead. For random baselines, shuffle the already selected labels, preserving exact 2B/4B/9B counts for each seed. Report 2.5/97.5 percentiles over seeds.

- [ ] **Step 4: Run the focused tests**

Expected: PASS.

### Task 5: Add the single command and CLI tests

**Files:**
- Modify: `src/scaleflow/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_routing.py`

- [ ] **Step 1: Write failing CLI tests**

Assert `analyze-gsm8k-routing` appears in help and parses config, Phase 8 split report, three input JSONL paths, and output path.

- [ ] **Step 2: Run the focused tests to verify failure**

Expected: FAIL because the parser and command handler do not exist.

- [ ] **Step 3: Implement the command**

Add:

```text
python -m scaleflow analyze-gsm8k-routing \
  --config configs/qwen35_gsm8k_routing.yaml \
  --split-report results/phase8_confidence_development.json \
  --inputs results/phase7_persistent_qwen35_2b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_4b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_9b_gsm8k_full.jsonl \
  --output results/phase10_text_routing.json
```

The handler must validate all inputs, fit both routers on development records, evaluate the exploratory split exactly once in that invocation, record input/config fingerprints, and refuse to overwrite an existing output. It must not load vLLM, import CUDA code, or read any model output as a feature.

- [ ] **Step 4: Run focused CLI tests**

Expected: PASS.

### Task 6: Run short synthetic and real offline validation

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`

- [ ] **Step 1: Run all CPU tests before real analysis**

Run: `CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q`

Expected: all existing tests and new routing tests pass.

- [ ] **Step 2: Run the real Phase 10 command once**

Run the exact command from Task 5 with the three ignored Phase 7 result files. Confirm the output records `phase8_split_verified=true`, development count 660, evaluation count 659, and both router reports.

- [ ] **Step 3: Validate output invariants without changing parameters**

Check that every evaluation sample appears once, all three model call counts match the reported ratios, each random baseline preserves its router’s call counts, and the output is marked exploratory. Do not rerun with modified thresholds or features.

- [ ] **Step 4: Update both READMEs**

Add the command, dependency installation, feature restrictions, direct-routing latency semantics, exploratory status, and a concise table of the measured results. Do not claim independent generalization or disclose ignored result files as tracked assets.

- [ ] **Step 5: Run final verification and commit/push**

Run:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
conda run -n scaleflow python -m pip check
git diff --check
git status --short --branch
```

Stage only Phase 10 code, configuration, tests, plans, and README changes. Confirm `results/` remains ignored, commit with `feat: add exploratory text routing analysis`, push `main`, and verify local `HEAD` equals `origin/main`.
