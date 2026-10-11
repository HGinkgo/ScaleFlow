import copy
import csv
import json
from pathlib import Path

import pytest

from scaleflow.quality_diagnostics import (
    CANDIDATE_F1_THRESHOLD, FAILURE_PRIORITY, MODES, classify_failure,
    diagnose_record, main, normalize_answer, numeric_tokens,
    possible_exact_match_only_failure, run_diagnostics, summarize_diagnostics,
    token_scores,
)
from scaleflow.rag import score_answer


def _record(mode="single_node_rag", prediction="London", gold="Paris", raw=None,
            answer_tokens=12, query_tokens=20, fallback=False, urls=None):
    references = ["https://example.org/a", "https://example.org/b"]
    urls = [references[0]] if urls is None else urls
    stages = []
    if mode != "direct":
        stages = [
            {"stage": "query", "compute_ms": 20, "success": True,
             "generated_token_count": query_tokens, "raw_output": '["location"]',
             "output": {"queries": ["location"]}},
            {"stage": "retrieve_rerank", "compute_ms": 5, "success": True,
             "output": {"evidence": [{"passage_id": str(i), "url": url,
                                       "text": "evidence"} for i, url in enumerate(urls)]}},
        ]
    stages.append({"stage": "answer", "compute_ms": 100, "success": True,
                   "generated_token_count": answer_tokens,
                   "raw_output": "FINAL: " + prediction if raw is None else raw,
                   "output": {"answer": prediction}})
    return {"sample_id": "1", "mode": mode, "question": "Where?",
            "gold_answer": gold, "predicted_answer": prediction,
            "pipeline_success": True, "correct": score_answer(prediction, gold),
            "query_fallback": fallback, "reference_urls": references,
            "retrieved_reference_count": len(set(urls) & set(references)) if mode != "direct" else 0,
            "stages": stages, "original_extra_field": {"nested": ["preserve", 3]}}


@pytest.mark.parametrize(("text", "expected"), [
    ("The Eiffel Tower!", "eiffel tower"), ("  A\tTower\n", "tower"),
    ("AN apple.", "apple"), ("Don't", "dont"), ("1,000", "1000"),
    ("", ""), ("The", ""),
])
def test_normalization_reuses_phase13_definition(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(("prediction", "gold", "precision", "recall", "f1"), [
    ("The Eiffel Tower!", "Eiffel Tower", 1, 1, 1),
    ("74", "74 years", 1, 0.5, 2 / 3),
    ("103 years", "103", 0.5, 1, 2 / 3),
    ("red red blue", "red blue blue", 2 / 3, 2 / 3, 2 / 3),
    ("red red red", "red", 1 / 3, 1, 0.5),
    ("London", "Paris", 0, 0, 0), ("", "", 0, 0, 0),
    ("The", "Paris", 0, 0, 0), ("Paris", "", 0, 0, 0),
])
def test_token_f1_is_multiset_overlap_and_empty_answers_are_zero(prediction, gold, precision, recall, f1):
    result = token_scores(prediction, gold)
    assert result["precision"] == pytest.approx(precision)
    assert result["recall"] == pytest.approx(recall)
    assert result["f1"] == pytest.approx(f1)


@pytest.mark.parametrize(("prediction", "gold", "candidate"), [
    ("74", "74 years", True), ("103 years", "103", True),
    ("Colin Firth, 2010, The King's Speech", "Colin Firth, 2011, The King's Speech", False),
    ("74.0 years", "74.1 years", False),
    ("John Smith", "Jane Smith", False), ("Paris", "Paris", False),
    ("1,000", "1000 meters", True), ("UNKNOWN", "103", False),
])
def test_exact_match_only_candidates_reject_conflicting_numbers(prediction, gold, candidate):
    assert possible_exact_match_only_failure(prediction, gold, token_scores(prediction, gold), CANDIDATE_F1_THRESHOLD) is candidate
    assert numeric_tokens("1,000 and 10.50") == ["10.5", "1E+3"]


@pytest.mark.parametrize(("raw", "has_final"), [
    ("reasoning\nFINAL: London", True), ("  final: London\r\n", True),
    ("Answer: London", False), ("FINAL:", False),
    ("FINAL: \nreasoning continues", False), ("text FINAL: London", False),
])
def test_nonempty_final_must_be_on_same_physical_line(raw, has_final):
    d = diagnose_record(_record(raw=raw))["quality_diagnostics"]
    assert d["has_final_line"] is has_final
    assert d["failure_category"] == ("genuinely_wrong_answer" if has_final else "missing_final_answer")


@pytest.mark.parametrize(("count", "reached"), [(255, False), (256, True), (257, True)])
def test_answer_budget_boundary_with_final_line(count, reached):
    d = diagnose_record(_record(answer_tokens=count))["quality_diagnostics"]
    assert d["answer_output_budget_reached"] is reached
    assert d["failure_category"] == ("output_budget_reached" if reached else "genuinely_wrong_answer")


def test_missing_final_precedes_answer_budget_and_does_not_extract_new_prediction():
    record = _record(prediction="UNKNOWN", raw="unfinished reasoning mentioning Paris", answer_tokens=256)
    d = diagnose_record(record)["quality_diagnostics"]
    assert not d["has_final_line"] and d["answer_output_budget_reached"]
    assert d["failure_category"] == "missing_final_answer"
    assert d["predicted_answer"] == "UNKNOWN" and not d["normalized_exact_match"]


def test_query_budget_is_separate_from_answer_budget_and_fallback():
    d = diagnose_record(_record(query_tokens=256))["quality_diagnostics"]
    assert d["query_output_budget_reached"]
    assert not d["query_generation_failure"] and not d["answer_output_budget_reached"]
    assert d["failure_category"] == "genuinely_wrong_answer"
    fallback = diagnose_record(_record(fallback=True))["quality_diagnostics"]
    assert fallback["query_generation_failure"]
    assert fallback["failure_category"] == "query_generation_failure"


@pytest.mark.parametrize(("urls", "count", "hit", "full"), [
    (["https://example.org/a", "https://example.org/a"], 1, True, False),
    (["https://example.org/a", "https://example.org/b"], 2, True, True),
    (["https://example.org/other"], 0, False, False), ([], 0, False, False),
])
def test_reference_hits_use_unique_document_urls_and_keep_partial_coverage(urls, count, hit, full):
    d = diagnose_record(_record(urls=urls))["quality_diagnostics"]
    assert d["retrieved_reference_document_count"] == count
    assert d["reference_document_hit"] is hit
    assert d["all_reference_documents_hit"] is full
    assert d["reference_document_recall"] == count / 2
    assert d["failure_category"] == ("genuinely_wrong_answer" if hit else "retrieval_miss")


def test_direct_does_not_get_a_retrieval_failure_and_absent_stages_are_null():
    d = diagnose_record(_record(mode="direct"))["quality_diagnostics"]
    assert not d["retrieval_applicable"] and d["reference_document_hit"] is None
    assert d["all_reference_documents_hit"] is None
    assert d["query_generated_token_count"] is None
    assert d["stage_compute_ms"] == {"query": None, "retrieval": None, "answer": 100}
    assert d["failure_category"] == "genuinely_wrong_answer"


def test_failure_priority_is_exclusive_even_when_every_signal_is_present():
    d = {"pipeline_success": False, "current_task_success": False,
         "query_generation_failure": True, "retrieval_applicable": True,
         "reference_document_hit": False, "has_final_line": False,
         "answer_output_budget_reached": True, "possible_exact_match_only_failure": True}
    changes = [("pipeline_success", True), ("query_generation_failure", False),
               ("reference_document_hit", True), ("has_final_line", True),
               ("answer_output_budget_reached", False), ("possible_exact_match_only_failure", False)]
    assert classify_failure(d) == FAILURE_PRIORITY[0]
    for (field, value), label in zip(changes, FAILURE_PRIORITY[1:]):
        d[field] = value
        assert classify_failure(d) == label


def test_pipeline_failure_precedes_text_exact_match_but_existing_success_is_preserved():
    record = _record(prediction="Paris", fallback=True, answer_tokens=256)
    good = diagnose_record(record)["quality_diagnostics"]
    assert good["failure_category"] == "success"
    assert good["query_fallback"] and good["answer_output_budget_reached"]
    record["pipeline_success"] = record["correct"] = record["stages"][0]["success"] = False
    bad = diagnose_record(record)["quality_diagnostics"]
    assert bad["normalized_exact_match"] and not bad["current_task_success"]
    assert bad["failure_category"] == "pipeline_failure"


def test_all_source_fields_and_nested_stages_are_unchanged():
    record = _record()
    original = copy.deepcopy(record)
    result = diagnose_record(record)
    assert record == original
    assert {key: value for key, value in result.items() if key != "quality_diagnostics"} == original
    with pytest.raises(ValueError, match="already contains"):
        diagnose_record(result)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_candidate_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        diagnose_record(_record(), threshold)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("correct", True, "correctness"), ("pipeline_success", False, "pipeline_success"),
    ("retrieved_reference_count", 2, "retrieved_reference_count"),
])
def test_inconsistent_source_fields_are_rejected_without_repair(field, value, message):
    record = _record()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        diagnose_record(record)


@pytest.mark.parametrize("count", [None, -1, 1.5, True])
def test_token_diagnostics_require_saved_valid_counts_without_tokenizer_fallback(count):
    with pytest.raises(ValueError, match="generated_token_count"):
        diagnose_record(_record(answer_tokens=count))


def test_aggregate_uses_all_tasks_and_primary_scores_not_diagnostic_candidates():
    records = [_record(prediction="Paris"), _record(prediction="74", gold="74 years"),
               _record(raw="unfinished", answer_tokens=256), _record(fallback=True),
               _record(urls=[]), _record(mode="direct")]
    rows = [diagnose_record(r) for r in records]
    summary = summarize_diagnostics(rows)
    rag = summary["single_node_rag"]
    assert rag["task_count"] == 5 and rag["failed_task_count"] == 4
    assert rag["pipeline_execution_success_rate"] == 1
    assert rag["normalized_exact_match_success_rate"] == 0.2
    assert rag["token_level_diagnostics"]["mean_f1"] == pytest.approx((1 + 2 / 3) / 5)
    assert rag["failure_categories"]["exact_match_only_failure"]["count"] == 1
    assert rag["failure_categories"]["missing_final_answer"]["proportion_of_all_tasks"] == 0.2
    assert rag["failure_categories"]["output_budget_reached"]["count"] == 0
    assert rag["raw_observable_flag_counts_nonexclusive"]["answer_output_budget_reached"] == 1
    assert sum(f["count"] for f in rag["failure_categories"].values()) + rag["success_count"] == 5
    assert rag["stage_compute_ms_mean"] == {"query": 20, "retrieval": 5, "answer": 100}
    assert rag["retrieved_reference_document_count_mean"] == 0.8
    assert summary["direct"]["stage_compute_ms_mean"]["query"] is None


def _write_source(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_cli_is_reproducible_retains_records_and_writes_full_human_review(tmp_path: Path, capsys):
    records = [_record(mode=mode) for mode in MODES]
    source = tmp_path / "source.jsonl"
    _write_source(source, records)
    original_bytes = source.read_bytes()
    result_bytes = []
    for directory in (tmp_path / "first", tmp_path / "second"):
        outputs = [directory / "phase14_diagnostics.jsonl", directory / "phase14_summary.json", directory / "phase14_review.csv"]
        assert main(["--inputs", str(source), "--output", str(outputs[0]), "--summary", str(outputs[1]), "--review", str(outputs[2])]) == 0
        diagnosed = [json.loads(line) for line in outputs[0].read_text().splitlines()]
        assert [{k: v for k, v in row.items() if k != "quality_diagnostics"} for row in diagnosed] == records
        report = json.loads(outputs[1].read_text())
        assert report["new_model_calls"] == 0 and report["unique_question_count"] == 1
        assert report["method"]["failure_priority"] == list(FAILURE_PRIORITY)
        with outputs[2].open(newline="") as stream:
            review = list(csv.DictReader(stream))
        assert [row["mode"] for row in review] == list(MODES)
        assert review[0]["gold_answer"] == "Paris" and review[0]["predicted_answer"] == "London"
        assert "raw_query_output" in review[1] and "matched_reference_urls" in review[1]
        assert b"\r\n" not in outputs[2].read_bytes()
        assert "exact match" in " ".join(report["notes"])
        result_bytes.append((outputs[0].read_bytes(), outputs[2].read_bytes()))
        with pytest.raises(ValueError, match="already exists"):
            run_diagnostics(source, *outputs)
    assert result_bytes[0] == result_bytes[1]
    assert source.read_bytes() == original_bytes
    assert capsys.readouterr().out


def test_output_names_and_paths_cannot_replace_earlier_phases_or_input(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_source(source, [_record(mode=mode) for mode in MODES])
    summary, review = tmp_path / "phase14_summary.json", tmp_path / "phase14_review.csv"
    with pytest.raises(ValueError, match="distinct"):
        run_diagnostics(source, source, summary, review)
    with pytest.raises(ValueError, match="filenames"):
        run_diagnostics(source, tmp_path / "phase13_diagnostics.jsonl", summary, review)
    assert not summary.exists() and not review.exists()


@pytest.mark.parametrize("invalid", ["missing_mode", "duplicate", "misaligned"])
def test_three_modes_must_have_identical_unique_samples_and_gold(tmp_path: Path, invalid):
    records = [_record(mode=mode) for mode in MODES]
    if invalid == "missing_mode":
        records.pop()
    elif invalid == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[-1]["gold_answer"] = "different"
    source = tmp_path / "source.jsonl"
    _write_source(source, records)
    with pytest.raises(ValueError):
        run_diagnostics(source, tmp_path / "phase14.jsonl", tmp_path / "phase14_summary.json", tmp_path / "phase14_review.csv")
    assert not (tmp_path / "phase14.jsonl").exists()
