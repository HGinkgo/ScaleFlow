"""Phase 14: offline diagnostics of saved Phase 13 answers, without inference.

Run with ``python -m scaleflow.quality_diagnostics --help``. Existing scoring,
measurement code and source records remain unchanged. Failure labels are
observable heuristics, not a semantic judge or a causal attribution experiment.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean

from scaleflow.baseline import file_sha256
from scaleflow.rag import load_jsonl, normalize_answer, score_answer


MODES = ("direct", "single_node_rag", "multi_model_rag")
FAILURE_PRIORITY = (
    "pipeline_failure", "query_generation_failure", "retrieval_miss",
    "missing_final_answer", "output_budget_reached",
    "exact_match_only_failure", "genuinely_wrong_answer",
)
TOKEN_BUDGET = 256
CANDIDATE_F1_THRESHOLD = 2 / 3
_FINAL_LINE = re.compile(r"(?im)^[ \t]*FINAL:[ \t]*(\S[^\r\n]*)\r?$")
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")


def token_scores(prediction: str, gold: str) -> dict:
    """Bag-of-word multiset precision/recall/F1 after Phase 13 normalization."""
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * overlap / (len(predicted) + len(expected)) if predicted and expected else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "overlap_tokens": overlap, "predicted_tokens": len(predicted),
            "gold_tokens": len(expected)}


def numeric_tokens(text: str) -> list[str]:
    # Keep decimals intact, unlike punctuation removal in the primary metric.
    return sorted(str(Decimal(value.replace(",", "")).normalize()) for value in _NUMBER.findall(text))


def possible_exact_match_only_failure(prediction: str, gold: str, scores: dict,
                                      threshold: float) -> bool:
    """Flag a possible wording/unit mismatch; never certify semantic correctness."""
    predicted, expected = Counter(normalize_answer(prediction).split()), Counter(normalize_answer(gold).split())
    return (not score_answer(prediction, gold) and bool(predicted) and bool(expected)
            and scores["f1"] >= threshold
            and (predicted <= expected or expected <= predicted)
            and numeric_tokens(prediction) == numeric_tokens(gold))


def classify_failure(diagnostic: dict) -> str:
    """One label per unsuccessful task; successful EM results remain successful."""
    if not diagnostic["pipeline_success"]:
        return "pipeline_failure"
    if diagnostic["current_task_success"]:
        return "success"
    conditions = {
        "query_generation_failure": diagnostic["query_generation_failure"],
        "retrieval_miss": diagnostic["retrieval_applicable"] and diagnostic["reference_document_hit"] is False,
        "missing_final_answer": not diagnostic["has_final_line"],
        "output_budget_reached": diagnostic["answer_output_budget_reached"],
        "exact_match_only_failure": diagnostic["possible_exact_match_only_failure"],
        "genuinely_wrong_answer": True,
    }
    return next(label for label in FAILURE_PRIORITY[1:] if conditions[label])


def diagnose_record(record: dict, threshold: float = CANDIDATE_F1_THRESHOLD) -> dict:
    """Return a new record retaining every original field and nested stage value."""
    if not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("candidate F1 threshold must be finite and in (0, 1]")
    if "quality_diagnostics" in record:
        raise ValueError("record already contains quality_diagnostics")
    mode = record["mode"]
    expected_stages = ["answer"] if mode == "direct" else ["query", "retrieve_rerank", "answer"]
    if mode not in MODES or [s["stage"] for s in record["stages"]] != expected_stages:
        raise ValueError("unsupported mode or Phase 13 stage chain")
    stages = {s["stage"]: s for s in record["stages"]}
    answer, query, retrieval = stages["answer"], stages.get("query"), stages.get("retrieve_rerank")
    pipeline = record["pipeline_success"]
    if pipeline != all(s["success"] for s in stages.values()):
        raise ValueError("saved pipeline_success disagrees with stage success")
    predicted, gold = record["predicted_answer"], record["gold_answer"]
    exact_match = score_answer(predicted, gold)
    if record["correct"] != (pipeline and exact_match):
        raise ValueError("saved correctness disagrees with unchanged Phase 13 scoring")
    final_lines = _FINAL_LINE.findall(answer["raw_output"])
    evidence = retrieval["output"]["evidence"] if retrieval else []
    references = set(record["reference_urls"])
    urls = {p["url"] for p in evidence}
    hits = sorted(references & urls)
    if record["retrieved_reference_count"] != len(hits):
        raise ValueError("saved retrieved_reference_count disagrees with evidence URLs")

    def token_count(stage: dict | None) -> int | None:
        if stage is None:
            return None
        value = stage.get("generated_token_count")
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError("generated_token_count must be a nonnegative integer")
        return value

    answer_tokens, query_tokens = token_count(answer), token_count(query)
    if answer_tokens is None or query is not None and query_tokens is None:
        raise ValueError("Phase 14 requires saved generated_token_count; no tokenizer or inference fallback")
    scores = token_scores(predicted, gold)
    diagnostic = {
        "gold_answer": gold, "predicted_answer": predicted,
        "normalized_gold_answer": normalize_answer(gold),
        "normalized_predicted_answer": normalize_answer(predicted),
        "normalized_exact_match": exact_match, "current_task_success": record["correct"],
        "token_level": scores, "pipeline_success": pipeline,
        "has_final_line": bool(final_lines), "final_line_answer": final_lines[-1].strip() if final_lines else None,
        "answer_generated_token_count": answer_tokens, "token_budget": TOKEN_BUDGET,
        "answer_output_budget_reached": answer_tokens >= TOKEN_BUDGET,
        "query_applicable": query is not None, "query_fallback": record["query_fallback"],
        "query_generation_failure": query is not None and (record["query_fallback"] or not query["success"]),
        "query_generated_token_count": query_tokens,
        "query_output_budget_reached": query_tokens >= TOKEN_BUDGET if query_tokens is not None else None,
        "retrieval_applicable": retrieval is not None,
        "reference_document_hit": bool(hits) if retrieval else None,
        "retrieved_reference_document_count": len(hits),
        "reference_document_count": len(references),
        "reference_document_recall": len(hits) / len(references) if retrieval and references else None,
        "all_reference_documents_hit": references <= urls if retrieval and references else None,
        "matched_reference_urls": hits,
        "missing_reference_urls": sorted(references - urls) if retrieval else [],
        "retrieved_document_urls": sorted(urls),
        "answer_is_unknown": normalize_answer(predicted) in {"", "unknown", "i dont know", "cannot be determined"},
        "numeric_tokens_predicted": numeric_tokens(predicted), "numeric_tokens_gold": numeric_tokens(gold),
        "possible_exact_match_only_failure": possible_exact_match_only_failure(predicted, gold, scores, threshold),
        "stage_compute_ms": {"query": query["compute_ms"] if query else None,
                             "retrieval": retrieval["compute_ms"] if retrieval else None,
                             "answer": answer["compute_ms"]},
    }
    diagnostic["failure_category"] = classify_failure(diagnostic)
    # Review flags may overlap; only failure_category is used in exclusive counts.
    diagnostic["review_flags"] = {
        "wrong_or_unsupported_answer_candidate": diagnostic["failure_category"] == "genuinely_wrong_answer",
        "possible_correct_but_exact_match_rejected": pipeline and not exact_match and diagnostic["possible_exact_match_only_failure"],
        "missing_final_line": not diagnostic["has_final_line"],
        "no_reference_document_retrieved": diagnostic["reference_document_hit"] is False,
        "incomplete_reference_document_coverage": diagnostic["all_reference_documents_hit"] is False,
        "answer_budget_reached": diagnostic["answer_output_budget_reached"],
        "query_fallback": diagnostic["query_fallback"],
    }
    return {**record, "quality_diagnostics": diagnostic}


def summarize_diagnostics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("diagnostic records are empty")
    result = {}
    for mode in MODES:
        rows = [r["quality_diagnostics"] for r in records if r["mode"] == mode]
        if not rows:
            continue
        failures = Counter(d["failure_category"] for d in rows)
        count = len(rows)
        result[mode] = {
            "task_count": count, "pipeline_execution_success_rate": fmean(d["pipeline_success"] for d in rows),
            "normalized_exact_match_success_rate": fmean(d["current_task_success"] for d in rows),
            "answer_text_exact_match_rate": fmean(d["normalized_exact_match"] for d in rows),
            "token_level_diagnostics": {"mean_" + field: fmean(d["token_level"][field] for d in rows) for field in ("f1", "precision", "recall")},
            "success_count": failures["success"], "failed_task_count": count - failures["success"],
            "failure_categories": {label: {"count": failures[label], "proportion_of_all_tasks": failures[label] / count}
                                   for label in FAILURE_PRIORITY},
            "retrieved_reference_document_count_mean": fmean(d["retrieved_reference_document_count"] for d in rows),
            "retrieval_applicable_task_count": sum(d["retrieval_applicable"] for d in rows),
            "reference_document_hit_count": sum(d["reference_document_hit"] is True for d in rows),
            "all_reference_documents_hit_count": sum(d["all_reference_documents_hit"] is True for d in rows),
            "raw_observable_flag_counts_nonexclusive": {
                "missing_final_line": sum(not d["has_final_line"] for d in rows),
                "answer_output_budget_reached": sum(d["answer_output_budget_reached"] for d in rows),
                "query_output_budget_reached": sum(d["query_output_budget_reached"] is True for d in rows),
                "query_fallback": sum(d["query_fallback"] for d in rows),
                "reference_document_miss": sum(d["reference_document_hit"] is False for d in rows),
                "unknown_answer": sum(d["answer_is_unknown"] for d in rows),
            },
            "stage_compute_ms_mean": {
                stage: fmean(values) if (values := [d["stage_compute_ms"][stage] for d in rows if d["stage_compute_ms"][stage] is not None]) else None
                for stage in ("query", "retrieval", "answer")
            },
            "sample_ids_by_failure_category": {label: [r["sample_id"] for r in records if r["mode"] == mode and r["quality_diagnostics"]["failure_category"] == label]
                                               for label in ("success",) + FAILURE_PRIORITY},
        }
        assert sum(failures.values()) == count
        assert sum(result[mode]["failure_categories"][label]["count"] for label in FAILURE_PRIORITY) == result[mode]["failed_task_count"]
    return result


def review_row(record: dict) -> dict:
    """Compact full-answer CSV; detailed raw reasoning/evidence remain in JSONL."""
    d = record["quality_diagnostics"]
    query = next((s for s in record["stages"] if s["stage"] == "query"), None)
    return {
        "sample_id": record["sample_id"], "mode": record["mode"], "question": record["question"],
        "gold_answer": d["gold_answer"], "predicted_answer": d["predicted_answer"],
        "normalized_exact_match": d["normalized_exact_match"], "pipeline_success": d["pipeline_success"],
        "token_f1": d["token_level"]["f1"], "token_precision": d["token_level"]["precision"], "token_recall": d["token_level"]["recall"],
        "failure_category": d["failure_category"], "has_final_line": d["has_final_line"],
        "answer_generated_token_count": d["answer_generated_token_count"], "answer_output_budget_reached": d["answer_output_budget_reached"],
        "query_fallback": d["query_fallback"], "query_output_budget_reached": d["query_output_budget_reached"],
        "retrieval_applicable": d["retrieval_applicable"], "reference_document_hit": d["reference_document_hit"],
        "retrieved_reference_document_count": d["retrieved_reference_document_count"], "reference_document_count": d["reference_document_count"],
        "all_reference_documents_hit": d["all_reference_documents_hit"],
        "possible_correct_but_exact_match_rejected": d["review_flags"]["possible_correct_but_exact_match_rejected"],
        "answer_is_unknown": d["answer_is_unknown"],
        "numeric_tokens_predicted": json.dumps(d["numeric_tokens_predicted"], ensure_ascii=False),
        "numeric_tokens_gold": json.dumps(d["numeric_tokens_gold"], ensure_ascii=False),
        "parsed_queries": json.dumps(query["output"]["queries"], ensure_ascii=False) if query else "",
        "raw_query_output": query["raw_output"] if query else "",
        "matched_reference_urls": json.dumps(d["matched_reference_urls"], ensure_ascii=False),
        "missing_reference_urls": json.dumps(d["missing_reference_urls"], ensure_ascii=False),
        "retrieved_document_urls": json.dumps(d["retrieved_document_urls"], ensure_ascii=False),
        **{stage + "_compute_ms": value for stage, value in d["stage_compute_ms"].items()},
    }


def run_diagnostics(inputs: Path, output: Path, summary: Path, review: Path,
                    threshold: float = CANDIDATE_F1_THRESHOLD) -> dict:
    paths = [output, summary, review]
    if len({p.resolve() for p in paths + [inputs]}) != 4:
        raise ValueError("input and three output paths must be distinct")
    for path in paths:
        if not re.match(r"^(phase14|quality[-_]diagnostics)", path.name, re.I):
            raise ValueError("new output filenames must start with phase14 or quality-diagnostics")
        if path.exists():
            raise ValueError(f"diagnostic output already exists: {path}")
    source_hash = file_sha256(inputs)
    source = load_jsonl(inputs)
    if not source:
        raise ValueError("source measurements are empty")
    source_ids = {mode: [r["sample_id"] for r in source if r["mode"] == mode] for mode in MODES}
    if any(len(ids) != len(set(ids)) for ids in source_ids.values()):
        raise ValueError("source has duplicate sample IDs within a mode")
    if not source_ids["direct"] or any(set(ids) != set(source_ids["direct"]) for ids in source_ids.values()):
        raise ValueError("three source modes must contain the same sample IDs")
    by_id = {r["sample_id"]: r for r in source if r["mode"] == "direct"}
    for record in source:
        if any(record[field] != by_id[record["sample_id"]][field] for field in ("question", "gold_answer", "reference_urls")):
            raise ValueError("questions/gold/references are not aligned across modes")
    records = [diagnose_record(record, threshold) for record in source]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    review_records = [review_row(r) for r in sorted(records, key=lambda r: (str(r["sample_id"]).zfill(16), MODES.index(r["mode"])))]
    with review.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_records)
    if file_sha256(inputs) != source_hash:
        raise ValueError("source measurements changed during diagnostics")
    report = {
        "phase": 14, "result_type": "offline_quality_diagnostics_of_existing_phase13_results",
        "new_model_calls": 0, "input": {"path": str(inputs), "sha256": source_hash},
        "task_record_count": len(records), "unique_question_count": len(by_id),
        "method": {
            "primary_metric": "Unchanged Phase 13 normalized exact match gated by pipeline success; exact match remains the main indicator.",
            "normalization": "Reuse Phase 13 lowercase, ASCII punctuation deletion, English article removal, whitespace collapse.",
            "token_score": "Multiset word tokens from the same normalized saved predicted/gold answers; F1=2*overlap/(predicted_count+gold_count). Empty answers score 0. Not model/subword tokens.",
            "failure_priority": list(FAILURE_PRIORITY),
            "priority_contract": "Pipeline failure first; existing successful tasks stay successful. Failed tasks receive the first applicable label. Raw flags may overlap but are not summed as failure categories.",
            "category_rules": {
                "pipeline_failure": "Saved pipeline_success is false; stage success consistency is validated.",
                "query_generation_failure": "RAG query stage failed or required recorded JSON-query fallback; does not assess the semantic quality of a well-formed query.",
                "retrieval_miss": "RAG retrieved zero distinct URLs from the question's recorded reference URLs. Direct answering is not applicable.",
                "missing_final_answer": "Answer raw output has no nonempty FINAL answer on the same physical line.",
                "output_budget_reached": "Saved answer generated_token_count >= 256. Count-based budget diagnostic, not a known vLLM finish reason or proof of truncation.",
                "exact_match_only_failure": "Possible wording/unit candidate only: F1 >= threshold, one normalized token multiset is contained in the other, and numeric tokens from raw answers agree. Requires human review; not confirmed correct.",
                "genuinely_wrong_answer": "Residual wrong/unsupported answer candidate after preceding checks; includes unknown/abstentions. Name is a diagnostic bucket, not semantic certification.",
            },
            "candidate_f1_threshold": threshold, "answer_token_budget": TOKEN_BUDGET,
            "retrieval_hit_definition": "Distinct exact URL intersection, identical to Phase 13. Document-level hit/complete reference coverage does not establish answer-bearing passage sufficiency.",
            "stage_latency": "Means of saved real service measurements only; absent direct query/retrieval stages are null, not zero-time measured stages.",
        },
        "notes": [
            "exact match 仍然是主指标；保留原始字段和已有评分，不提升或重写成功率。",
            "token-level 分数仅用于诊断，不能替代官方评分。高 overlap 或 exact_match_only_failure 不代表语义正确；部分列表、顺序和否定可能误导词项匹配。",
            "本任务只分析已有 Phase 13 结果，没有调用模型、重新检索或进行新的调度实验，不产生新的模型实验结论。",
            "失败标签按优先级互斥；重叠的原始异常仅作为检查线索。参考文档命中并不证明证据充分，单轮日志不能确定检索、推理与评分的因果贡献。",
        ],
        "metrics": summarize_diagnostics(records),
        "manual_review": {
            "csv_path": str(review), "jsonl_path": str(output),
            "csv_order": "sample_id then direct/single_node_rag/multi_model_rag; full answers retained, no truncation",
            "jsonl_contract": "Every original field/stage is retained; only quality_diagnostics is added. Full raw answer/query and evidence remain available.",
            "filter_columns": ["failure_category", "possible_correct_but_exact_match_rejected", "has_final_line", "reference_document_hit", "all_reference_documents_hit", "answer_output_budget_reached", "query_fallback"],
        },
        "outputs": {"diagnostics": {"path": str(output), "sha256": file_sha256(output)},
                    "review": {"path": str(review), "sha256": file_sha256(review)}},
        "run_command": f"python -m scaleflow.quality_diagnostics --inputs {inputs} --output {output} --summary {summary} --review {review} --candidate-f1-threshold {threshold}",
    }
    with summary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 14 offline RAG quality diagnostics; no model calls")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL retaining source fields")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True, help="new human-review CSV")
    parser.add_argument("--candidate-f1-threshold", type=float, default=CANDIDATE_F1_THRESHOLD)
    args = parser.parse_args(argv)
    report = run_diagnostics(args.inputs, args.output, args.summary, args.review, args.candidate_f1_threshold)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
