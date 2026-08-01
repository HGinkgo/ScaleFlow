import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from scaleflow import __version__
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
        seed=config["project"]["seed"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        presence_penalty=sampling["presence_penalty"],
        max_tokens=sampling["max_tokens"],
        logprobs=sampling["logprobs"],
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-mock":
        return run_mock(args.config, args.output)
    if args.command == "run-vllm":
        return run_vllm(args.config, args.output)
    parser.print_help()
    return 0
