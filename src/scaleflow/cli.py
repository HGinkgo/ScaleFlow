import argparse
from collections.abc import Sequence
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from typing import Any

from scaleflow import __version__
from scaleflow.baseline import (
    compare_baseline_records,
    compare_model_records,
    ensure_dataset,
    file_sha256,
    load_gsm8k_jsonl,
    load_records_jsonl,
    percentile,
    run_baseline_samples,
    select_samples,
    summarize_baseline,
    warmup_backend,
    write_records_jsonl,
    write_summary_json,
)
from scaleflow.backends import MockBackend, VLLMBackend
from scaleflow.config import ConfigError, load_config
from scaleflow.offline import (
    analyze_confidence,
    analyze_confidence_models,
    random_acceptance_baseline,
    replay_cascade,
    search_pareto_thresholds,
    split_sample_ids,
)
from scaleflow.performance import (
    run_performance_experiment,
    select_model_config,
    validate_performance_config,
    validate_reference_contract,
)
from scaleflow.routing import analyze_routing
from scaleflow.scheduler.policies import AlwaysModelPolicy, ConfidenceCascadePolicy
from scaleflow.scheduler.runner import run_requests, write_results_jsonl
from scaleflow.schemas import InferenceRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scaleflow",
        description="ScaleFlow multi-scale language model research framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_mock_parser = subparsers.add_parser(
        "run-mock",
        help="run the deterministic MockBackend scheduling scenario",
    )
    run_mock_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the mock YAML configuration",
    )
    run_mock_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the output JSONL file",
    )
    run_vllm_parser = subparsers.add_parser(
        "run-vllm",
        help="run fixed text requests with a local vLLM model",
    )
    run_vllm_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the vLLM YAML configuration",
    )
    run_vllm_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the output JSONL file",
    )
    run_gsm8k_parser = subparsers.add_parser(
        "run-gsm8k",
        help="run a fixed GSM8K single-model baseline",
    )
    run_gsm8k_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the GSM8K baseline YAML configuration",
    )
    run_gsm8k_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the per-sample JSONL output",
    )
    run_gsm8k_parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="path to the aggregate JSON summary",
    )
    compare_gsm8k_parser = subparsers.add_parser(
        "compare-gsm8k",
        help="compare two aligned GSM8K baseline JSONL files",
    )
    compare_gsm8k_parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="path to the baseline model JSONL results",
    )
    compare_gsm8k_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="path to the candidate model JSONL results",
    )
    compare_gsm8k_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the comparison JSON output",
    )
    compare_multi_parser = subparsers.add_parser(
        "compare-gsm8k-multi",
        help="compare ordered GSM8K baseline JSONL files",
    )
    compare_multi_parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL result paths in ascending model-capability order",
    )
    compare_multi_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the multi-model comparison JSON output",
    )
    confidence_parser = subparsers.add_parser(
        "analyze-gsm8k-confidence",
        help="validate confidence on a deterministic development split",
    )
    confidence_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the offline analysis YAML configuration",
    )
    confidence_parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="GSM8K JSONL paths in ascending model-capability order",
    )
    confidence_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to the development confidence report",
    )
    search_parser = subparsers.add_parser(
        "search-gsm8k-cascade",
        help="freeze offline cascade thresholds on the development split",
    )
    search_parser.add_argument("--config", type=Path, required=True)
    search_parser.add_argument("--confidence-report", type=Path, required=True)
    search_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    search_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate-gsm8k-cascade",
        help="evaluate one frozen offline cascade on the holdout split",
    )
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    concurrency_parser = subparsers.add_parser(
        "run-gsm8k-concurrency",
        help="benchmark one vLLM model with closed-loop streaming concurrency",
    )
    concurrency_parser.add_argument("--config", type=Path, required=True)
    concurrency_parser.add_argument("--model-id", required=True)
    concurrency_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Phase 7 full GSM8K JSONL for the selected model",
    )
    concurrency_parser.add_argument(
        "--reference-summary",
        type=Path,
        required=True,
        help="Phase 7 full GSM8K summary JSON for the selected model",
    )
    concurrency_parser.add_argument("--output", type=Path, required=True)
    concurrency_parser.add_argument("--summary", type=Path, required=True)
    concurrency_parser.add_argument("--server-log", type=Path, required=True)
    concurrency_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="stop after startup warmup and the concurrency-1 contract check",
    )
    routing_parser = subparsers.add_parser(
        "analyze-gsm8k-routing",
        help="analyze exploratory pre-inference text routing",
    )
    routing_parser.add_argument("--config", type=Path, required=True)
    routing_parser.add_argument("--split-report", type=Path, required=True)
    routing_parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    routing_parser.add_argument("--output", type=Path, required=True)
    return parser


def _build_requests(config: dict[str, Any]) -> list[InferenceRequest]:
    return [
        InferenceRequest(
            request_id=item["request_id"],
            prompt=item["prompt"],
            metadata=dict(item.get("metadata", {})),
        )
        for item in config["requests"]
    ]


def _build_mock_backends(config: dict[str, Any]) -> dict[str, MockBackend]:
    return {
        item["model_id"]: MockBackend(
            model_id=item["model_id"],
            responses=item["responses"],
        )
        for item in config["models"]
    }


def _build_policy(
    config: dict[str, Any],
) -> AlwaysModelPolicy | ConfidenceCascadePolicy:
    scheduler = config["scheduler"]
    policy_name = scheduler["policy"]
    if policy_name == "always_model":
        return AlwaysModelPolicy(scheduler["model_id"])
    if policy_name == "confidence_cascade":
        return ConfidenceCascadePolicy(
            scheduler["model_order"],
            scheduler["confidence_threshold"],
        )
    raise ConfigError(f"unsupported scheduler policy: {policy_name}")


def run_mock(config_path: Path, output_path: Path) -> int:
    config = load_config(config_path)
    results = run_requests(
        _build_requests(config),
        _build_mock_backends(config),
        _build_policy(config),
    )
    write_results_jsonl(output_path, results)
    print(f"wrote {len(results)} mock results to {output_path}")
    return 0


def _build_vllm_backend(config: dict[str, Any]) -> VLLMBackend:
    backend = config["backend"]
    sampling = config["sampling"]
    return VLLMBackend(
        model_id=backend["model_id"],
        revision=backend.get("revision"),
        language_model_only=backend["language_model_only"],
        enable_thinking=backend["enable_thinking"],
        dtype=backend["dtype"],
        max_model_len=backend["max_model_len"],
        gpu_memory_utilization=backend["gpu_memory_utilization"],
        enforce_eager=backend["enforce_eager"],
        enable_prefix_caching=backend.get("enable_prefix_caching", True),
        seed=config["project"]["seed"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        presence_penalty=sampling["presence_penalty"],
        max_tokens=sampling["max_tokens"],
        logprobs=sampling["logprobs"],
    )


def _common_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    backend = config["backend"]
    dataset_fields = (
        "name",
        "split",
        "commit",
        "sha256",
        "expected_record_count",
        "selection_method",
        "selection_seed",
        "sample_indices",
    )
    backend_fields = (
        "language_model_only",
        "enable_thinking",
        "dtype",
        "max_model_len",
        "enforce_eager",
        "enable_prefix_caching",
    )
    return {
        "project_seed": config["project"]["seed"],
        "dataset": {
            field: dataset[field] for field in dataset_fields if field in dataset
        },
        "prompt_template": config["prompt"]["template"],
        "warmup_prompts": list(config["warmup"]["prompts"]),
        "generation_config": dict(config["sampling"]),
        "backend_common_config": {
            field: backend[field] for field in backend_fields if field in backend
        },
    }


def _experiment_fingerprint(experiment_config: dict[str, Any]) -> str:
    canonical = json.dumps(
        experiment_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_sample_indices(
    dataset_config: dict[str, Any],
    *,
    record_count: int,
) -> list[int]:
    selection_method = dataset_config.get("selection_method")
    if selection_method == "all_records":
        expected_record_count = dataset_config["expected_record_count"]
        if record_count != expected_record_count:
            raise ConfigError(
                "all_records requires the verified dataset size: "
                f"expected {expected_record_count}, got {record_count}"
            )
        return list(range(record_count))

    sample_indices = dataset_config.get("sample_indices")
    if not isinstance(sample_indices, list):
        raise ConfigError(
            "dataset.sample_indices is required unless "
            "selection_method is all_records"
        )
    return list(sample_indices)


def run_vllm(config_path: Path, output_path: Path) -> int:
    config = load_config(config_path)
    backend = _build_vllm_backend(config)
    model_id = config["backend"]["model_id"]
    results = run_requests(
        _build_requests(config),
        {model_id: backend},
        _build_policy(config),
    )
    write_results_jsonl(output_path, results)
    summary = {
        "request_count": len(results),
        "successful_requests": sum(result.success for result in results),
        "output": str(output_path),
        "model_info": backend.get_model_info(),
    }
    print(json.dumps(summary, ensure_ascii=False))
    failures = [result for result in results if not result.success]
    if failures:
        print(
            "vLLM run failed: "
            + "; ".join(f"{item.request_id}: {item.error}" for item in failures),
            file=sys.stderr,
        )
        return 1
    return 0


def run_gsm8k(
    config_path: Path,
    output_path: Path,
    summary_path: Path,
) -> int:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    dataset_path = ensure_dataset(
        dataset_config["local_path"],
        dataset_config["source_url"],
        dataset_config["sha256"],
    )
    dataset_records = load_gsm8k_jsonl(dataset_path)
    expected_record_count = dataset_config["expected_record_count"]
    if len(dataset_records) != expected_record_count:
        raise ConfigError(
            "GSM8K record count mismatch: "
            f"expected {expected_record_count}, got {len(dataset_records)}"
        )
    sample_indices = _resolve_sample_indices(
        dataset_config,
        record_count=len(dataset_records),
    )
    samples = select_samples(dataset_records, sample_indices)

    backend = _build_vllm_backend(config)
    warmup = warmup_backend(backend, config["warmup"]["prompts"])
    records = run_baseline_samples(
        samples,
        backend,
        config["prompt"]["template"],
        max_tokens=config["sampling"]["max_tokens"],
    )
    experiment_config = _common_experiment_config(config)
    experiment_fingerprint = _experiment_fingerprint(experiment_config)
    for record in records:
        record["experiment_config"] = experiment_config
        record["experiment_fingerprint"] = experiment_fingerprint
    write_records_jsonl(output_path, records)

    summary = summarize_baseline(records)
    summary.update(
        {
            "project_seed": config["project"]["seed"],
            "experiment_config": experiment_config,
            "experiment_fingerprint": experiment_fingerprint,
            "dataset": {
                "name": dataset_config["name"],
                "split": dataset_config["split"],
                "commit": dataset_config["commit"],
                "sha256": dataset_config["sha256"],
                "record_count": len(dataset_records),
                "selection_method": dataset_config.get(
                    "selection_method", "explicit_indices"
                ),
                "selection_seed": dataset_config.get("selection_seed"),
                "sample_indices": sample_indices,
                "sample_ids": [sample["sample_id"] for sample in samples],
            },
            "prompt_template": config["prompt"]["template"],
            "generation_config": dict(config["sampling"]),
            "backend_config": dict(config["backend"]),
            "warmup": warmup,
            "model_info": backend.get_model_info(),
            "output": str(output_path),
        }
    )
    write_summary_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["inference_failure_count"] else 0


def compare_gsm8k(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> int:
    comparison = compare_baseline_records(
        load_records_jsonl(baseline_path),
        load_records_jsonl(candidate_path),
    )
    comparison["inputs"] = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
    }
    write_summary_json(output_path, comparison)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "request_count": comparison["request_count"],
                "categories": comparison["categories"],
                "rescued_count": comparison["rescued_count"],
                "rescue_rate": comparison["rescue_rate"],
                "oracle_accuracy": comparison["oracle_accuracy"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def compare_gsm8k_multi(input_paths: Sequence[Path], output_path: Path) -> int:
    comparison = compare_model_records(
        [load_records_jsonl(path) for path in input_paths]
    )
    comparison["inputs"] = [str(path) for path in input_paths]
    write_summary_json(output_path, comparison)
    final_oracle = comparison["oracle_progression"][-1]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "request_count": comparison["request_count"],
                "model_order": comparison["model_order"],
                "ordered_pair_count": len(comparison["ordered_pairs"]),
                "oracle_correct_count": final_oracle["oracle_correct_count"],
                "oracle_accuracy": final_oracle["oracle_accuracy"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def analyze_gsm8k_confidence(
    config_path: Path,
    input_paths: Sequence[Path],
    output_path: Path,
) -> int:
    config = load_config(config_path)
    split = config.get("split")
    validation = config.get("confidence_validation")
    if not isinstance(split, dict) or not isinstance(validation, dict):
        raise ConfigError("offline config requires split and confidence_validation")
    if split.get("method") != "sha256_seed_sample_id":
        raise ConfigError("split.method must be sha256_seed_sample_id")

    report = analyze_confidence_models(
        [load_records_jsonl(path) for path in input_paths],
        development_count=int(split["development_count"]),
        split_seed=int(split["seed"]),
        low_confidence_fraction=float(validation["low_confidence_fraction"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
        bootstrap_seed=int(validation["bootstrap_seed"]),
    )
    report["input_files"] = [
        {
            "model_id": model["model_id"],
            "sha256": file_sha256(path),
        }
        for model, path in zip(report["models"], input_paths, strict=True)
    ]
    report["inputs"] = [str(path) for path in input_paths]
    write_summary_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "passed_intermediate_model_ids": report[
                    "passed_intermediate_model_ids"
                ],
                "failed_intermediate_model_ids": report[
                    "failed_intermediate_model_ids"
                ],
                "development_count": report["split"]["development_count"],
                "evaluation_outcomes_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must contain a JSON object")
    return value


def _offline_split_config(config: dict[str, Any]) -> dict[str, Any]:
    split = config.get("split")
    if not isinstance(split, dict) or split.get("method") != "sha256_seed_sample_id":
        raise ConfigError("split.method must be sha256_seed_sample_id")
    return split


def _ordered_model_ids(
    records_by_model: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    model_ids: list[str] = []
    for records in records_by_model:
        if not records:
            raise ConfigError("offline analysis cannot use an empty input")
        ids = {record.get("model_id") for record in records}
        if len(ids) != 1 or not isinstance(next(iter(ids)), str):
            raise ConfigError("each input must contain exactly one model_id")
        model_ids.append(next(iter(ids)))
    if len(set(model_ids)) != len(model_ids):
        raise ConfigError("input model_id values must be unique")
    return model_ids


def _input_fingerprints(
    model_ids: Sequence[str],
    input_paths: Sequence[Path],
) -> list[dict[str, str]]:
    return [
        {"model_id": model_id, "sha256": file_sha256(path)}
        for model_id, path in zip(model_ids, input_paths, strict=True)
    ]


def _select_passed_chain(
    model_ids: Sequence[str],
    passed_intermediate_model_ids: Sequence[str],
    terminal_model_id: str,
) -> tuple[list[int], list[str]]:
    if not model_ids or model_ids[-1] != terminal_model_id:
        raise ConfigError("terminal model does not match the input order")
    chain_indices = [
        index
        for index, model_id in enumerate(model_ids[:-1])
        if model_id in passed_intermediate_model_ids
    ] + [len(model_ids) - 1]
    chain_ids = [model_ids[index] for index in chain_indices]
    if chain_ids[:-1] != list(passed_intermediate_model_ids):
        raise ConfigError("passed model order does not match input order")
    return chain_indices, chain_ids


def search_gsm8k_cascade(
    config_path: Path,
    confidence_report_path: Path,
    input_paths: Sequence[Path],
    output_path: Path,
) -> int:
    config = load_config(config_path)
    split_config = _offline_split_config(config)
    search_config = config.get("threshold_search")
    if not isinstance(search_config, dict):
        raise ConfigError("offline config requires threshold_search")
    confidence_report = _load_json_object(
        confidence_report_path,
        "confidence report",
    )
    if confidence_report.get("scope") != "development_confidence_gate":
        raise ConfigError("confidence report has the wrong scope")
    if confidence_report.get("all_intermediate_models_failed") is True:
        raise ConfigError("all intermediate models failed the confidence gate")

    records_by_model = [load_records_jsonl(path) for path in input_paths]
    model_ids = _ordered_model_ids(records_by_model)
    input_fingerprints = _input_fingerprints(model_ids, input_paths)
    if confidence_report.get("input_files") != input_fingerprints:
        raise ConfigError("confidence report input fingerprints do not match")
    passed_ids = confidence_report.get("passed_intermediate_model_ids")
    if not isinstance(passed_ids, list) or not all(
        isinstance(model_id, str) for model_id in passed_ids
    ):
        raise ConfigError("confidence report has invalid passed model IDs")
    chain_indices, chain_ids = _select_passed_chain(
        model_ids,
        passed_ids,
        str(confidence_report.get("terminal_model_id")),
    )

    sample_ids = [record["sample_id"] for record in records_by_model[0]]
    development_ids, evaluation_ids = split_sample_ids(
        sample_ids,
        development_count=int(split_config["development_count"]),
        seed=int(split_config["seed"]),
    )
    report_split = confidence_report.get("split")
    if not isinstance(report_split, dict) or (
        report_split.get("development_sample_ids") != development_ids
        or report_split.get("evaluation_sample_ids") != evaluation_ids
    ):
        raise ConfigError("confidence report split does not match current inputs")

    chain_records = [records_by_model[index] for index in chain_indices]
    search = search_pareto_thresholds(
        chain_records,
        development_ids,
        quantile_step=float(search_config["quantile_step"]),
    )
    selected = search["selected"]
    development_replay = replay_cascade(
        chain_records,
        development_ids,
        selected["thresholds"],
    )
    report = {
        "scope": "development_threshold_search",
        "evaluation_outcomes_read": False,
        "selection_rule": (
            "minimize mean cumulative latency subject to cascade correct count "
            "being at least the terminal-model correct count on development data"
        ),
        "model_chain": chain_ids,
        "excluded_intermediate_model_ids": confidence_report.get(
            "failed_intermediate_model_ids",
            [],
        ),
        "split": report_split,
        "input_files": input_fingerprints,
        "target": {
            "model_id": search["target_model_id"],
            "correct_count": search["target_correct_count"],
            "accuracy": search["target_accuracy"],
        },
        "selected_policy": selected,
        "search": {
            key: value
            for key, value in search.items()
            if key not in {"selected", "target_model_id", "target_correct_count", "target_accuracy"}
        },
        "development_replay": development_replay,
    }
    write_summary_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "model_chain": chain_ids,
                "thresholds": selected["thresholds"],
                "development_accuracy": selected["accuracy"],
                "target_accuracy": search["target_accuracy"],
                "evaluation_outcomes_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _direct_model_summary(
    records: Sequence[dict[str, Any]],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    indexed = {record["sample_id"]: record for record in records}
    selected = [indexed[sample_id] for sample_id in sample_ids]
    latency = [float(record["latency_ms"]) for record in selected]
    correct_count = sum(bool(record["correct"]) for record in selected)
    return {
        "model_id": selected[0]["model_id"],
        "correct_count": correct_count,
        "accuracy": correct_count / len(selected),
        "latency_ms": {
            "mean": sum(latency) / len(latency),
            "p50": percentile(latency, 50),
            "p95": percentile(latency, 95),
        },
    }


def evaluate_gsm8k_cascade(
    config_path: Path,
    policy_path: Path,
    input_paths: Sequence[Path],
    output_path: Path,
) -> int:
    config = load_config(config_path)
    split_config = _offline_split_config(config)
    random_config = config.get("random_baseline")
    if not isinstance(random_config, dict):
        raise ConfigError("offline config requires random_baseline")
    policy = _load_json_object(policy_path, "frozen cascade policy")
    if policy.get("scope") != "development_threshold_search":
        raise ConfigError("cascade policy has the wrong scope")
    marker_path = policy_path.with_name(policy_path.name + ".evaluated")
    if marker_path.exists():
        raise ConfigError("this frozen cascade policy has already been evaluated")
    if output_path.exists():
        raise ConfigError("holdout evaluation output already exists")

    records_by_model = [load_records_jsonl(path) for path in input_paths]
    model_ids = _ordered_model_ids(records_by_model)
    expected_files = policy.get("input_files")
    actual_files = _input_fingerprints(model_ids, input_paths)
    if expected_files != actual_files:
        raise ConfigError("frozen policy input fingerprints do not match")

    sample_ids = [record["sample_id"] for record in records_by_model[0]]
    development_ids, evaluation_ids = split_sample_ids(
        sample_ids,
        development_count=int(split_config["development_count"]),
        seed=int(split_config["seed"]),
    )
    policy_split = policy.get("split")
    if not isinstance(policy_split, dict) or (
        policy_split.get("development_sample_ids") != development_ids
        or policy_split.get("evaluation_sample_ids") != evaluation_ids
    ):
        raise ConfigError("frozen policy split does not match current inputs")

    model_chain = policy.get("model_chain")
    if not isinstance(model_chain, list):
        raise ConfigError("frozen policy has no model chain")
    chain_indices = [model_ids.index(model_id) for model_id in model_chain]
    chain_records = [records_by_model[index] for index in chain_indices]
    selected_policy = policy.get("selected_policy")
    if not isinstance(selected_policy, dict) or not isinstance(
        selected_policy.get("thresholds"),
        list,
    ):
        raise ConfigError("frozen policy has no thresholds")

    try:
        with marker_path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"status": "evaluation_reserved", "output": output_path.name},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except FileExistsError as error:
        raise ConfigError(
            "this frozen cascade policy has already been evaluated"
        ) from error

    cascade = replay_cascade(
        chain_records,
        evaluation_ids,
        selected_policy["thresholds"],
    )
    seed_start = int(random_config["seed_start"])
    seed_count = int(random_config["seed_count"])
    random_baseline = random_acceptance_baseline(
        chain_records,
        evaluation_ids,
        acceptance_rates=cascade["stage_acceptance_rates"],
        seeds=range(seed_start, seed_start + seed_count),
    )
    accuracy_gain = cascade["accuracy"] - random_baseline["accuracy"]["mean"]
    accuracy_gain_interval = {
        "lower": cascade["accuracy"] - random_baseline["accuracy"]["upper"],
        "upper": cascade["accuracy"] - random_baseline["accuracy"]["lower"],
    }
    validation_config = config.get("confidence_validation")
    if not isinstance(validation_config, dict):
        raise ConfigError("offline config requires confidence_validation")
    full_confidence = {
        "scope": "full_dataset_descriptive_only",
        "used_for_method_selection": False,
        "models": [],
    }
    for model_id, records in zip(model_ids, records_by_model, strict=True):
        analysis = analyze_confidence(
            records,
            low_confidence_fraction=float(
                validation_config["low_confidence_fraction"]
            ),
            bootstrap_iterations=int(validation_config["bootstrap_iterations"]),
            bootstrap_seed=int(validation_config["bootstrap_seed"]),
        )
        analysis["model_id"] = model_id
        full_confidence["models"].append(analysis)
    report = {
        "scope": "single_use_holdout_evaluation",
        "policy_frozen_before_evaluation": True,
        "evaluation_sample_ids": evaluation_ids,
        "model_chain": model_chain,
        "selected_thresholds": selected_policy["thresholds"],
        "cascade": cascade,
        "random_baseline": random_baseline,
        "confidence_vs_random": {
            "accuracy_gain_over_random_mean": accuracy_gain,
            "accuracy_gain_interval_95": accuracy_gain_interval,
            "gain_ci_lower_above_zero": accuracy_gain_interval["lower"] > 0,
        },
        "standalone_models": [
            _direct_model_summary(records, evaluation_ids)
            for records in records_by_model
        ],
        "full_dataset_confidence": full_confidence,
        "latency_scope": "offline_replay_estimate",
        "latency_exclusions": [
            "model_loading",
            "model_switching",
            "queueing",
            "concurrency_interference",
        ],
    }
    write_summary_json(output_path, report)
    marker_path.write_text(
        json.dumps(
            {
                "status": "evaluation_completed",
                "output": output_path.name,
                "output_sha256": file_sha256(output_path),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "evaluation_count": len(evaluation_ids),
                "cascade_accuracy": cascade["accuracy"],
                "random_accuracy_mean": random_baseline["accuracy"]["mean"],
                "terminal_invocation_rate": cascade["terminal_invocation_rate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_gsm8k_concurrency(
    config_path: Path,
    model_id: str,
    reference_path: Path,
    reference_summary_path: Path,
    output_path: Path,
    summary_path: Path,
    server_log_path: Path,
    *,
    preflight_only: bool,
) -> int:
    config = load_config(config_path)
    validate_performance_config(config)
    model = select_model_config(config, model_id)
    try:
        installed_vllm = version("vllm")
    except PackageNotFoundError as error:
        raise ConfigError("vLLM is not installed in the active environment") from error
    expected_vllm = config["server"]["vllm_version"]
    if installed_vllm != expected_vllm:
        raise ConfigError(
            f"vLLM version mismatch: expected {expected_vllm}, got {installed_vllm}"
        )

    dataset_config = config["dataset"]
    dataset_path = ensure_dataset(
        dataset_config["local_path"],
        dataset_config["source_url"],
        dataset_config["sha256"],
    )
    dataset_records = load_gsm8k_jsonl(dataset_path)
    if len(dataset_records) != dataset_config["expected_record_count"]:
        raise ConfigError(
            "GSM8K record count mismatch: "
            f"expected {dataset_config['expected_record_count']}, "
            f"got {len(dataset_records)}"
        )
    samples = select_samples(dataset_records, dataset_config["sample_indices"])
    reference_records = load_records_jsonl(reference_path)
    reference_summary = _load_json_object(
        reference_summary_path,
        "Phase 7 reference summary",
    )
    aligned_references = validate_reference_contract(
        config,
        model,
        samples,
        reference_records,
        reference_summary,
    )
    return run_performance_experiment(
        config,
        model,
        samples,
        aligned_references,
        reference_path=reference_path,
        reference_summary_path=reference_summary_path,
        output_path=output_path,
        summary_path=summary_path,
        server_log_path=server_log_path,
        preflight_only=preflight_only,
        reference_sha256=file_sha256(reference_path),
        reference_summary_sha256=file_sha256(reference_summary_path),
    )


def analyze_gsm8k_routing(
    config_path: Path,
    split_report_path: Path,
    input_paths: Sequence[Path],
    output_path: Path,
) -> int:
    if output_path.exists():
        raise ConfigError(f"routing output already exists: {output_path}")
    config = load_config(config_path)
    expected_models = config.get("models")
    if not isinstance(expected_models, list) or len(expected_models) != 3:
        raise ConfigError("routing config must contain exactly three models")
    records_by_model_sequence = [load_records_jsonl(path) for path in input_paths]
    if any(len(records) != 1319 for records in records_by_model_sequence):
        raise ConfigError("Phase 10 inputs must each contain exactly 1319 records")
    comparison = compare_model_records(records_by_model_sequence)
    expected_model_ids = [model["model_id"] for model in expected_models]
    if comparison["model_order"] != expected_model_ids:
        raise ConfigError(
            "Phase 10 input order/model IDs do not match the routing configuration"
        )
    label_by_model_id = {
        model["model_id"]: model["label"] for model in expected_models
    }
    records_by_label = {
        label_by_model_id[model_id]: records
        for model_id, records in zip(
            comparison["model_order"],
            records_by_model_sequence,
            strict=True,
        )
    }
    split_report = _load_json_object(split_report_path, "Phase 8 split report")
    report = analyze_routing(config, records_by_label, split_report)
    report["config_fingerprint"] = sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report["input_contract"] = {
        "model_result_paths": [str(path) for path in input_paths],
        "model_result_sha256": [file_sha256(path) for path in input_paths],
        "phase8_split_report_path": str(split_report_path),
        "phase8_split_report_sha256": file_sha256(split_report_path),
        "phase7_experiment_config": comparison["experiment_config"],
    }
    write_summary_json(output_path, report)
    print(f"wrote exploratory routing report to {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-mock":
        return run_mock(args.config, args.output)
    if args.command == "run-vllm":
        return run_vllm(args.config, args.output)
    if args.command == "run-gsm8k":
        return run_gsm8k(args.config, args.output, args.summary)
    if args.command == "compare-gsm8k":
        return compare_gsm8k(args.baseline, args.candidate, args.output)
    if args.command == "compare-gsm8k-multi":
        return compare_gsm8k_multi(args.inputs, args.output)
    if args.command == "analyze-gsm8k-confidence":
        return analyze_gsm8k_confidence(args.config, args.inputs, args.output)
    if args.command == "search-gsm8k-cascade":
        return search_gsm8k_cascade(
            args.config,
            args.confidence_report,
            args.inputs,
            args.output,
        )
    if args.command == "evaluate-gsm8k-cascade":
        return evaluate_gsm8k_cascade(
            args.config,
            args.policy,
            args.inputs,
            args.output,
        )
    if args.command == "run-gsm8k-concurrency":
        return run_gsm8k_concurrency(
            args.config,
            args.model_id,
            args.reference,
            args.reference_summary,
            args.output,
            args.summary,
            args.server_log,
            preflight_only=args.preflight_only,
        )
    if args.command == "analyze-gsm8k-routing":
        return analyze_gsm8k_routing(
            args.config,
            args.split_report,
            args.inputs,
            args.output,
        )
    parser.print_help()
    return 0
