"""Isolated Phase 15 vLLM worker; all 48 answer calls use one loaded 4B model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

from scaleflow.backends import VLLMBackend
from scaleflow.baseline import file_sha256
from scaleflow.rag import load_jsonl
from scaleflow.phase15_diagnostics import chat_token_ids, exclusive_json, output_flags, validate_budget


def generate_one(backend, item: dict) -> dict:
    """Observe vLLM token IDs/finish metadata without changing legacy backends."""
    started = perf_counter()
    try:
        outputs = backend._llm.chat([{"role": "user", "content": item["prompt"]}],
            backend._sampling_params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        latency_ms = (perf_counter() - started) * 1000
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("vLLM returned an unexpected number of completions")
        completion = outputs[0].outputs[0]
        actual_ids = list(outputs[0].prompt_token_ids)
        if actual_ids != item["preflight_token_ids"]:
            raise ValueError("actual vLLM prompt tokens differ from preflight; possible truncation/template drift")
        raw = completion.text
        output_tokens = len(completion.token_ids)
        finish_reason = getattr(completion, "finish_reason", None)
        row = {"request_id": item["request_id"], "sample_id": item["sample_id"], "method": item["method"],
               "input_tokens": len(actual_ids), "output_tokens": output_tokens,
               "raw_output": raw, "finish_reason": finish_reason,
               "stop_reason": getattr(completion, "stop_reason", None),
               "latency_ms": latency_ms, "success": True, "error": None,
               "measurement_type": "real_vllm_answer_only", "prompt_tokens_match_preflight": True,
               "output_token_ids": list(completion.token_ids)}
        row.update(output_flags(raw, output_tokens, finish_reason))
        return row
    except Exception as exc:
        return {"request_id": item["request_id"], "sample_id": item["sample_id"], "method": item["method"],
                "input_tokens": item.get("preflight_input_tokens"), "output_tokens": None,
                "raw_output": "", "finish_reason": None, "stop_reason": None,
                "latency_ms": (perf_counter() - started) * 1000,
                "success": False, "error": f"{type(exc).__name__}: {exc}", "has_final": False,
                "final_answer": None, "output_budget_reached": False,
                "measurement_type": "real_vllm_answer_only", "prompt_tokens_match_preflight": False}


def main() -> None:
    job_path, output_path = map(Path, sys.argv[1:])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    meta_path = output_path.with_suffix(".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError("Phase 15 generation output already exists")
    if file_sha256(Path(job["requests_path"])) != job["inputs_sha256"]:
        raise ValueError("paired input hash differs from frozen job")
    backend_args = dict(job["backend"])
    backend_args["model_id"] = job["model_path"]
    # The local snapshot is deliberate: no Phase 15 network/model download.
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(backend_args["model_id"], local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(f"tokenizer preflight failed: {type(exc).__name__}: {exc}") from exc
    requests = load_jsonl(Path(job["requests_path"]))
    if len(requests) != 48 or len({r["request_id"] for r in requests}) != 48:
        raise ValueError("expected exactly 48 distinct answer requests")
    for item in requests:
        ids = chat_token_ids(tokenizer, item["prompt"])
        validate_budget(len(ids), int(job["max_model_len"]), int(job["max_tokens"]))
        if len(ids) != item["preflight_input_tokens"]:
            raise ValueError("tokenizer preflight drift")
        item["preflight_token_ids"] = ids
    backend = VLLMBackend(**backend_args)
    # Warmup is excluded from request latency and output files.
    warmups = []
    for prompt in job["warmup_prompts"]:
        started = perf_counter()
        outputs = backend._llm.chat([{"role": "user", "content": prompt}], backend._sampling_params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        warmups.append({"latency_ms_excluded": (perf_counter() - started) * 1000,
                        "input_tokens": len(outputs[0].prompt_token_ids), "raw_output": outputs[0].outputs[0].text})
    exclusive_json(meta_path, {"model_id": job["model_id"], "revision": job["revision"],
        "backend_info": backend.get_model_info(), "backend_settings": job["backend"],
        "job_sha256": file_sha256(job_path), "inputs_sha256": job["inputs_sha256"],
        "warmups_excluded": warmups, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": job["measurement_scope"], "sample_count": 24, "answer_call_count": 48})
    with output_path.open("x", encoding="utf-8") as stream:
        for item in requests:
            row = generate_one(backend, item)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{item['request_id']}: success={row['success']}, {row['latency_ms']:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
