import argparse
from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from scaleflow import __version__
from scaleflow.baseline import (
    compare_baseline_records,
    compare_model_records,
    ensure_dataset,
    load_gsm8k_jsonl,
    load_records_jsonl,
    run_baseline_samples,
    select_samples,
    summarize_baseline,
    warmup_backend,
    write_records_jsonl,
    write_summary_json,
)
from scaleflow.backends import MockBackend, VLLMBackend
from scaleflow.config import ConfigError, load_config
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
    parser.print_help()
    return 0
