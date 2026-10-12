"""Phase 15 paired evidence diagnostic helpers.

This module owns only new Phase 15 data preparation and result aggregation;
Phase 7--14 measurement files are read-only inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re
from statistics import fmean
import tarfile
from typing import Any
from urllib.parse import unquote

from scaleflow.baseline import file_sha256, percentile
from scaleflow.rag import html_text, load_jsonl, normalize_answer, score_answer

FINAL_RE = re.compile(r"(?im)^[ \t]*FINAL:[ \t]*(\S[^\r\n]*)[ \t]*\r?$")
MAX_TOKENS = 256
METHODS = ("A_phase13_evidence", "B_sufficient_evidence")


def extract_final(raw: str) -> str | None:
    matches = FINAL_RE.findall(raw or "")
    return matches[-1].strip() if matches else None


def output_flags(raw: str, output_tokens: int | None, finish_reason: str | None) -> dict[str, Any]:
    final = extract_final(raw)
    return {"has_final": final is not None, "final_answer": final,
            "output_budget_reached": (output_tokens is not None and output_tokens >= MAX_TOKENS)
            or finish_reason in {"length", "max_tokens"}}


def verify_excerpt(selection: dict[str, Any], texts: dict[str, dict[str, Any]]) -> str:
    doc = texts[selection["url"]]
    text = doc["text"]
    start, end = selection["char_start"], selection["char_end"]
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError("invalid excerpt offsets")
    excerpt = text[start:end]
    if sha256(excerpt.encode()).hexdigest() != selection["text_sha256"]:
        raise ValueError(f"excerpt hash mismatch: {selection['excerpt_id']}")
    return excerpt


def canonical_url(url: str) -> str:
    return url.split("#", 1)[0]


def read_reference_texts(config: dict) -> dict[str, dict[str, Any]]:
    """Read only selected reference documents, without retrieval or downloads."""
    required = {s["url"] for sample in config["samples"] for s in sample["selections"]}
    mapping = json.loads(Path(config["sources"]["mapping"]).read_text())
    texts = {}
    with tarfile.open(config["sources"]["archive"], "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".html"):
                continue
            url = canonical_url(mapping.get(Path(member.name).stem, ""))
            if url not in required:
                continue
            raw = archive.extractfile(member).read()
            text = html_text(raw.decode("utf-8"))
            texts[url] = {"text": text, "archive_member": member.name,
                          "html_sha256": sha256(raw).hexdigest(),
                          "extracted_text_sha256": sha256(text.encode()).hexdigest()}
    if required - texts.keys():
        raise ValueError(f"missing frozen reference documents: {sorted(required - texts.keys())}")
    return texts


def load_phase13_answers(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["mode"] == "multi_model_rag":
            if str(row["sample_id"]) in grouped:
                raise ValueError("duplicate Phase 13 sample")
            grouped[str(row["sample_id"])] = row
    if len(grouped) != 24:
        raise ValueError(f"expected 24 Phase 13 multi_model_rag rows, got {len(grouped)}")
    return grouped


def phase15_contexts(config: dict, phase13: dict[str, dict[str, Any]], texts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build paired contexts before inference, preserving A evidence verbatim."""
    rows = []
    ids = [str(sample["sample_id"]) for sample in config["samples"]]
    if len(ids) != len(set(ids)) or set(ids) != set(phase13):
        raise ValueError("evidence plan must retain every Phase 13 sample exactly once")
    for sample in config["samples"]:
        sid = str(sample["sample_id"])
        old = phase13[sid]
        retrieval = next(s for s in old["stages"] if s["stage"] == "retrieve_rerank")
        evidence_a = retrieval["output"]["evidence"]
        context_a = "\n\n".join(f"[{i+1}] {p['title']} ({p['url']})\n{p['text']}" for i, p in enumerate(evidence_a))
        excerpts = []
        verification = sample["verification"]
        if verification["status"] not in {"sufficient", "insufficient", "pending"}:
            raise ValueError("invalid evidence sufficiency status")
        if verification["human_confirmed"] or verification["verifier"] != "Codex agent":
            raise ValueError("this plan contains only agent verification")
        if verification["status"] == "sufficient" and any(f["assessment"] != "supported" for f in verification["required_facts"]):
            raise ValueError("sufficient evidence cannot contain missing/unverified facts")
        for selection in sample["selections"]:
            if selection["url"] not in {canonical_url(u) for u in old["reference_urls"]}:
                raise ValueError("B excerpt must come from a question reference document")
            doc = texts[selection["url"]]
            excerpts.append({**selection, "text": verify_excerpt(selection, texts),
                             "archive_member": doc["archive_member"], "html_sha256": doc["html_sha256"],
                             "extracted_text_sha256": doc["extracted_text_sha256"],
                             "title": unquote(selection["url"].rsplit('/', 1)[-1]).replace("_", " ")})
        context_b = "\n\n".join(f"[{i+1}] {e['title']} ({e['url']})\n{e['text']}" for i, e in enumerate(excerpts))
        reference_set = {canonical_url(u) for u in old["reference_urls"]}
        a_urls = {canonical_url(e["url"]) for e in evidence_a}
        b_urls = {e["url"] for e in excerpts}
        rows.append({"sample_id": sid, "question": old["question"], "gold_answer": old["gold_answer"],
                     "reference_urls": old["reference_urls"], "a_context": context_a,
                     "b_context": context_b, "a_passage_count": len(evidence_a),
                     "a_evidence_original": evidence_a,
                     "a_reference_hit_count_phase13_url_rule": old["retrieved_reference_count"],
                     "a_reference_document_count_canonical": len(a_urls & reference_set),
                     "a_nonreference_document_count_canonical": len(a_urls - reference_set),
                     "b_reference_document_count_canonical": len(b_urls & reference_set),
                     "b_nonreference_document_count_canonical": len(b_urls - reference_set),
                     "a_context_chars": len(context_a), "b_context_chars": len(context_b),
                     "b_excerpts": excerpts, "b_verification": sample["verification"]})
    return rows


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(r["sample_id"], r["method"]) for r in results]
    if len(keys) != len(set(keys)) or any(m not in METHODS for _, m in keys):
        raise ValueError("duplicate or unsupported paired measurement")
    ids = {sid for sid, _ in keys}
    if set(keys) != {(sid, method) for sid in ids for method in METHODS}:
        raise ValueError("incomplete paired measurements")
    out: dict[str, Any] = {"question_count": len(ids), "generation_count": len(results), "methods": {}}
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        em = [r["normalized_exact_match"] for r in rows]
        out["methods"][method] = {
            "task_count": len(rows), "exact_match_count": sum(em),
            "exact_match_rate": fmean(em) if rows else 0.0,
            "unknown_count": sum(r["final_answer_normalized"] == "unknown" for r in rows),
            "missing_final_count": sum(not r["has_final"] for r in rows),
            "budget_reached_count": sum(r["output_budget_reached"] for r in rows),
            "pipeline_failure_count": sum(not r["success"] for r in rows),
            "output_format_violation_count": sum(not r["output_format_valid"] for r in rows),
            "mean_output_tokens": fmean(r["output_tokens"] for r in rows if r["output_tokens"] is not None) if any(r["output_tokens"] is not None for r in rows) else None,
            "mean_input_tokens": fmean(r["input_tokens"] for r in rows if r["input_tokens"] is not None) if any(r["input_tokens"] is not None for r in rows) else None,
            "mean_latency_ms": fmean(r["latency_ms"] for r in rows),
            "p95_latency_ms": percentile([r["latency_ms"] for r in rows], 95),
            "mean_context_chars": fmean(r["context_chars"] for r in rows),
            "mean_context_tokens": fmean(r["context_tokens"] for r in rows if r["context_tokens"] is not None) if any(r["context_tokens"] is not None for r in rows) else None,
            "mean_reference_document_count_canonical": fmean(r["reference_document_count_canonical"] for r in rows),
            "mean_nonreference_document_count_canonical": fmean(r["nonreference_document_count_canonical"] for r in rows),
            "sample_ids": [r["sample_id"] for r in rows],
        }
    # Only B agent-verified sufficient samples enter the paired comparison.
    b_rows = {r["sample_id"]: r for r in results if r["method"] == "B_sufficient_evidence"}
    sufficient = {sid for sid, r in b_rows.items() if r["b_verification_status"] == "sufficient"}
    pairs = []
    for sid in sorted(sufficient, key=lambda x: int(x)):
        a = next(r for r in results if r["sample_id"] == sid and r["method"] == METHODS[0])
        b = b_rows[sid]
        pairs.append({"sample_id": sid, "a_exact": a["normalized_exact_match"], "b_exact": b["normalized_exact_match"],
                      "outcome": "both_correct" if a["normalized_exact_match"] and b["normalized_exact_match"] else
                      "only_a_correct" if a["normalized_exact_match"] else "only_b_correct" if b["normalized_exact_match"] else "both_wrong"})
    status_counts = Counter(r["b_verification_status"] for r in b_rows.values())
    counts = Counter(p["outcome"] for p in pairs)
    out["b_evidence_status_counts"] = {s: status_counts[s] for s in ("sufficient", "insufficient", "pending")}
    out["paired_on_agent_verified_sufficient"] = {"task_count": len(pairs), "human_confirmed_count": 0,
        "outcomes": {s: counts[s] for s in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong")}, "rows": pairs}
    out["semantic_review_note"] = "Exact match is primary. Candidate equivalences and unresolved semantic cases are recorded per task; no token score or model self-score is promoted to correctness."
    return out


def build_prompt(question: str, context: str, template: str) -> str:
    return template.format(question=question, context=context)


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def exclusive_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def chat_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return encoded if isinstance(encoded, list) else encoded.tolist()


def validate_budget(input_tokens: int, max_model_len: int, max_tokens: int) -> None:
    if input_tokens + max_tokens > max_model_len:
        raise ValueError(f"context overflow: {input_tokens}+{max_tokens}>{max_model_len}; truncation is forbidden")


def prepare(config_path: Path, model_path: Path, output_dir: Path) -> dict:
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    paths = {name: output_dir / f"phase15_{name}.{ext}" for name, ext in
             (("evidence_checks", "jsonl"), ("paired_inputs", "jsonl"), ("job", "json"))}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    for source, expected in config["source_sha256"].items():
        if file_sha256(Path(source)) != expected:
            raise ValueError(f"frozen source hash mismatch: {source}")
    manifest = json.loads(Path(config["sources"]["manifest"]).read_text())
    phase13 = load_phase13_answers(Path(config["sources"]["measurements"]))
    if [s["sample_id"] for s in manifest["samples"]] != [s["sample_id"] for s in config["samples"]]:
        raise ValueError("sample order differs from fixed Phase 13 manifest")
    for sample in manifest["samples"]:
        if any(sample[k] != phase13[sample["sample_id"]][k] for k in ("question", "gold_answer", "reference_urls")):
            raise ValueError("Phase 13 manifest and measurements disagree")
    checks = phase15_contexts(config, phase13, read_reference_texts(config))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    # Balance order effects: exactly twelve A-first pairs and twelve B-first pairs.
    a_first = [True] * 12 + [False] * 12
    Random(config["seed"]).shuffle(a_first)
    inputs = []
    for check, first in zip(checks, a_first, strict=True):
        for group in (("a", "b") if first else ("b", "a")):
            context = check[group + "_context"]
            prompt = build_prompt(check["question"], context, config["prompt_template"])
            token_count = len(chat_token_ids(tokenizer, prompt))
            validate_budget(token_count, config["backend"]["max_model_len"], config["backend"]["max_tokens"])
            row = {"request_id": f"{check['sample_id']}-{group.upper()}", "sample_id": check["sample_id"],
                   "method": METHODS[group == "b"], "question": check["question"], "prompt": prompt,
                   "prompt_sha256": sha256(prompt.encode()).hexdigest(), "preflight_input_tokens": token_count,
                   "context_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
                   "context_chars": len(context), "context_words": len(context.split()),
                   "reference_document_count_canonical": check[group + "_reference_document_count_canonical"],
                   "nonreference_document_count_canonical": check[group + "_nonreference_document_count_canonical"],
                   "b_verification_status": check["b_verification"]["status"],
                   "execution_index": len(inputs), "context_truncated": False}
            inputs.append(row)
    backend = {**config["backend"], "model_id": str(model_path)}
    job = {"phase": 15, "model_id": config["model_id"], "revision": config["revision"],
           "model_path": str(model_path), "backend": backend, "gpu": config["gpu"],
           "requests_path": str(paths["paired_inputs"]), "request_count": len(inputs),
           "max_tokens": config["backend"]["max_tokens"], "max_model_len": config["backend"]["max_model_len"],
           "config_path": str(config_path), "config_sha256": file_sha256(config_path),
           "source_sha256": config["source_sha256"], "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
           "warmup_prompts": ["Reply with the word ready.",
               "Reply with the word ready after reading this neutral warmup text.\n" + "Neutral text for warmup. " * 600],
           "max_preflight_input_tokens": max(r["preflight_input_tokens"] for r in inputs),
           "execution_order": [r["request_id"] for r in inputs],
           "measurement_scope": "Sequential real vLLM answer calls only; no queue/network simulation, no query generation, no retrieval; load and warmups excluded."}
    exclusive_jsonl(paths["evidence_checks"], checks)
    exclusive_jsonl(paths["paired_inputs"], inputs)
    job["inputs_sha256"] = file_sha256(paths["paired_inputs"])
    job["evidence_checks_sha256"] = file_sha256(paths["evidence_checks"])
    exclusive_json(paths["job"], job)
    return {"outputs": {k: str(v) for k, v in paths.items()}, "request_count": len(inputs),
            "max_preflight_input_tokens": job["max_preflight_input_tokens"]}


def collect(output_dir: Path, review_path: Path | None = None) -> dict:
    output_paths = [output_dir / f"phase15_{stem}.{ext}" for stem, ext in
                    (("answer_measurements", "jsonl"), ("paired_review", "csv"), ("summary", "json"))]
    for path in output_paths:
        if path.exists():
            raise FileExistsError(path)
    inputs_path = output_dir / "phase15_paired_inputs.jsonl"
    checks_path = output_dir / "phase15_evidence_checks.jsonl"
    raw_path = output_dir / "phase15_generations.jsonl"
    job_path = output_dir / "phase15_job.json"
    job = json.loads(job_path.read_text())
    if file_sha256(inputs_path) != job["inputs_sha256"] or file_sha256(checks_path) != job["evidence_checks_sha256"]:
        raise ValueError("frozen evidence/inputs changed after preparation")
    metadata_path = output_dir / "phase15_generations.meta.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["job_sha256"] != file_sha256(job_path) or metadata["inputs_sha256"] != job["inputs_sha256"]:
        raise ValueError("generation provenance differs from frozen job")
    checks = {r["sample_id"]: r for r in load_jsonl(checks_path)}
    inputs = {r["request_id"]: r for r in load_jsonl(inputs_path)}
    generated = load_jsonl(raw_path)
    reviews = json.loads(review_path.read_text()) if review_path else {}
    if set(reviews) - set(inputs):
        raise ValueError("semantic reviews contain unknown request IDs")
    if len(generated) != 48 or len({r["request_id"] for r in generated}) != 48 or set(inputs) != {r["request_id"] for r in generated}:
        raise ValueError("require exactly 48 distinct real paired generations")
    results = []
    for g in generated:
        inp = inputs[g["request_id"]]
        if g["sample_id"] != inp["sample_id"] or g["method"] != inp["method"]:
            raise ValueError("generation/request identity mismatch")
        check = checks[g["sample_id"]]
        if any(g[k] != v for k, v in output_flags(g["raw_output"], g["output_tokens"], g["finish_reason"]).items()):
            raise ValueError("saved FINAL/budget flags disagree with raw generation")
        final = g["final_answer"] or ""
        exact = bool(g["success"] and g["has_final"] and score_answer(final, check["gold_answer"]))
        if g["success"] and g["input_tokens"] != inp["preflight_input_tokens"]:
            raise ValueError("measured prompt tokens differ from frozen preflight")
        # Missing review entries are explicitly unresolved, never a correctness claim.
        semantic = reviews.get(g["request_id"], {"status": "not_reviewed", "reason": "No semantic verdict has been confirmed."})
        if semantic["status"] not in {"exact_match", "possible_equivalent", "not_equivalent_candidate", "ambiguous", "abstention", "not_reviewed"}:
            raise ValueError("invalid semantic review status")
        results.append({**g, "question": check["question"], "gold_answer": check["gold_answer"],
                        "normalized_exact_match": exact, "final_answer_normalized": normalize_answer(final),
                        "output_format_valid": bool(FINAL_RE.fullmatch(g["raw_output"].strip())),
                        "semantic_review": {**semantic, "verifier": "Codex agent", "human_confirmed": False,
                                            "affects_primary_score": False},
                        **{k: inp[k] for k in ("prompt_sha256", "execution_index", "context_tokens", "context_chars", "context_words",
                                              "reference_document_count_canonical", "nonreference_document_count_canonical",
                                              "context_truncated", "b_verification_status")}})
    summary = aggregate(results)
    summary.update(phase=15, measurement_type="real_vllm_answer_only", scoring="Unchanged Phase 13 normalized exact match applied to extracted FINAL answer.",
        sources_sha256={str(p): file_sha256(p) for p in (inputs_path, checks_path, raw_path)},
        notes=["All 24 fixed Phase 13 samples are kept. B uses question reference information and is diagnostic, not normal retrieval performance.",
               "Sufficiency is checked by Codex agent; zero samples are human-confirmed. Insufficient and pending samples stay in full-24 results but are excluded from the sufficient-evidence paired subset.",
               "A and B share a new FINAL-only prompt and identical generation/context budgets. Differences from Phase 13 cannot all be attributed to evidence changes.",
               "A and B differ in length, distractors and reference-document inclusion. Pair improvement is not an isolated causal estimate of document coverage.",
        "Semantic candidates are for human review only; no token F1, overlap or model self-judgment establishes correctness."])
    summary["unconfirmed_semantic_candidates"] = [{"request_id": r["request_id"], "prediction": r["final_answer"],
        "gold": r["gold_answer"], **r["semantic_review"]} for r in results if not r["normalized_exact_match"] and r["semantic_review"]["status"] in {"possible_equivalent", "ambiguous", "not_reviewed"}]
    paired_csv = []
    for sid, check in checks.items():
        a, b = (next(r for r in results if r["sample_id"] == sid and r["method"] == m) for m in METHODS)
        row = {"sample_id": sid, "question": check["question"], "gold_answer": check["gold_answer"],
               "b_evidence_status": check["b_verification"]["status"], "b_status_reason": check["b_verification"]["reason"]}
        for label, r in (("a", a), ("b", b)):
            for key in ("final_answer", "normalized_exact_match", "input_tokens", "output_tokens", "has_final", "output_budget_reached", "finish_reason", "stop_reason", "latency_ms", "context_tokens", "context_chars"):
                row[label + "_" + key] = r[key]
            row[label + "_semantic_status"] = r["semantic_review"]["status"]
            row[label + "_semantic_reason"] = r["semantic_review"]["reason"]
        paired_csv.append(row)
    exclusive_jsonl(output_paths[0], results)
    with output_paths[1].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_csv[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_csv)
    exclusive_json(output_paths[2], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze evidence and preflight all 48 inputs; no inference")
    prep.add_argument("--config", type=Path, default=Path("configs/qwen35_frames_evidence_diagnostics.json"))
    prep.add_argument("--model-path", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, default=Path("results"))
    summarizer = sub.add_parser("summarize", help="Score saved real generations without model calls")
    summarizer.add_argument("--output-dir", type=Path, default=Path("results"))
    summarizer.add_argument("--semantic-review", type=Path)
    args = parser.parse_args()
    result = prepare(args.config, args.model_path, args.output_dir) if args.command == "prepare" else collect(args.output_dir, args.semantic_review)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
