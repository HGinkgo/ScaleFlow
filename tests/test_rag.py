import csv
from pathlib import Path

import pytest

from scaleflow.cli import build_parser
from scaleflow.rag import (
    BM25Retriever, execute_rag, generation_diagnostics, html_text, json_bytes, parse_queries,
    score_answer, select_frames, summarize_measurements,
)
from scaleflow.rag_network import replay_workflow, summarize_replay, transfer


def _records() -> list[dict]:
    records = []
    for mode in ["direct", "single_node_rag", "multi_model_rag"]:
        stages = []
        if mode != "direct":
            stages = [
                {"stage": "query", "model_id": "query" if mode == "multi_model_rag" else "answer",
                 "compute_ms": 20, "input": {"question": "Q"}, "output": {"queries": ["中文"]}},
                {"stage": "retrieve_rerank", "model_id": "retriever", "compute_ms": 5,
                 "input": {"queries": ["中文"]}, "output": {"evidence": "x" * 100}},
            ]
        stages.append({"stage": "answer", "model_id": "answer", "compute_ms": 100,
                       "input": {"evidence": "x" * 100} if stages else {"question": "Q"},
                       "output": {"answer": "A"}})
        records.append({"sample_id": "1", "mode": mode, "correct": True,
                        "pipeline_success": True, "predicted_answer": "A", "gold_answer": "A", "stages": stages})
    return records


def _simulation() -> dict:
    return {"nodes": [
        {"node_id": "edge", "capacity": 1, "model_ids": ["query", "answer", "retriever"], "service_time_multiplier": {}},
        {"node_id": "cloud", "capacity": 1, "model_ids": ["answer"], "service_time_multiplier": {"answer": 0.5}},
    ], "client_node": "edge", "single_node": "edge", "seed": 42,
        "arrival": {"process": "periodic", "request_count": 3},
        "fixed_stage_nodes": {"query": "edge", "retrieve_rerank": "edge", "answer": "cloud"}}


def _scenario(bandwidth: float = 100) -> dict:
    return {"name": "test", "links": [{"source": "edge", "destination": "cloud", "bandwidth_mbps": bandwidth, "propagation_ms": 1}]}


def test_fixed_random_sampling_before_inference(tmp_path: Path) -> None:
    path = tmp_path / "frames.tsv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "Prompt", "Answer", "wiki_links"], delimiter="\t")
        writer.writeheader()
        for i in range(30):
            writer.writerow({"": str(i), "Prompt": f"question {i}", "Answer": str(i), "wiki_links": "['url']"})
    a, population = select_frames(path, 8, 42)
    assert population == 30
    assert a == select_frames(path, 8, 42)[0]
    assert a != select_frames(path, 8, 43)[0]
    assert len({row["sample_id"] for row in a}) == 8
    with pytest.raises(ValueError):
        select_frames(path, 31, 42)


def test_html_retains_table_evidence_and_removes_scripts() -> None:
    assert html_text('<div class="mw-parser-output"><script>secret</script><table><tr><td>A</td><td>42</td></tr></table></div>') == "A | 42 |"


def test_query_fallback_and_exact_match() -> None:
    assert parse_queries('["alpha", "alpha", "beta"]', "Q", 2) == (["alpha", "beta"], False)
    assert parse_queries("not JSON", "Q", 2) == (["Q"], True)
    assert score_answer("The Eiffel Tower!", "Eiffel Tower")
    assert not score_answer("possibly Paris but London too", "Paris")
    assert not score_answer("", "")


def test_output_budget_diagnostics_do_not_claim_completion() -> None:
    unfinished = {"text": "Reasoning that never reached an answer", "token_logprobs": [-1] * 4}
    assert generation_diagnostics(unfinished, 4) == {
        "generated_token_count": 4, "output_budget_reached": True, "has_final_line": False,
    }
    assert generation_diagnostics({"text": "FINAL: Paris", "token_logprobs": [-1]}, 4)["has_final_line"]


def test_real_multistage_chain_consumes_query_and_evidence() -> None:
    samples = [{"sample_id": "0", "question": "Where is Alpha?", "gold_answer": "Paris", "reference_urls": ["url"]}]
    config = {"stages": {"multi_query_model": "small", "answer_model": "bigger", "max_queries": 2},
              "prompts": {"query": "query {question}", "direct": "direct {question}", "rag_answer": "{question}\n{context}"}}
    retriever = BM25Retriever([{"passage_id": "p", "title": "Alpha", "url": "url", "text": "Alpha is in Paris."}],
                             {"candidates_per_query": 2, "top_k": 1, "max_passages_per_document": 1, "rerank_overlap_weight": 0.02})
    calls = []
    def batch(model, requests, name):
        calls.append((model, requests, name))
        return {r["request_id"]: {"response": {"text": '["Alpha location"]' if name.startswith("queries") else "FINAL: Paris",
                                               "success": True, "latency_ms": 10}, "model_info": {"backend": "vllm"}} for r in requests}
    rows = execute_rag(config, {"samples": samples}, retriever, batch)
    assert len(rows) == 3
    multi = next(r for r in rows if r["mode"] == "multi_model_rag")
    assert [s["model_id"] for s in multi["stages"]] == ["small", "bm25-rrf-lexical", "bigger"]
    assert multi["stages"][1]["input"]["queries"] == ["Alpha location"]
    assert "Alpha is in Paris" in multi["answer_prompt"]
    assert all(r["correct"] for r in rows)
    assert summarize_measurements(rows)["multi_model_rag"]["task_success_rate"] == 1


def test_multistage_failures_remain_in_denominator() -> None:
    retriever = BM25Retriever([{"passage_id": "p", "title": "A", "url": "url", "text": "Paris"}],
                             {"candidates_per_query": 2, "top_k": 1, "max_passages_per_document": 1, "rerank_overlap_weight": 0.02})
    config = {"stages": {"multi_query_model": "q", "answer_model": "a", "max_queries": 2},
              "prompts": {"query": "{question}", "direct": "{question}", "rag_answer": "{context}"}}
    def batch(model, requests, name):
        return {r["request_id"]: {"response": {"text": "FINAL: Paris", "success": not name.startswith("queries"), "latency_ms": 1}, "model_info": {}} for r in requests}
    rows = execute_rag(config, {"samples": [{"sample_id": "0", "question": "Paris", "gold_answer": "Paris", "reference_urls": []}]}, retriever, batch)
    assert rows[0]["correct"]
    assert not rows[1]["correct"] and not rows[2]["correct"]


def test_utf8_exact_payload_bandwidth_and_local_transfer() -> None:
    payload = {"query": "中文"}
    size = len(json_bytes(payload))
    assert size > len(json_bytes(payload).decode())
    result = transfer(payload, "edge", "cloud", _scenario(2)["links"])
    assert result["transmitted_bytes"] == size
    assert result["transmission_ms"] == size * 8 / 2000
    assert transfer(payload, "edge", "edge", [])["transmitted_bytes"] == 0
    with pytest.raises(ValueError, match="no fixed topology"):
        transfer(payload, "edge", "unknown", [])
    with pytest.raises(ValueError, match="bandwidth"):
        transfer(payload, "edge", "cloud", _scenario(0)["links"])


def test_fcfs_capacity_dependencies_and_latency_decomposition() -> None:
    rows = replay_workflow(_records(), _simulation(), _scenario(), "multi_model_rag", "fixed", 100)
    assert len(rows) == 3
    for row in rows:
        assert [s["stage"] for s in row["stages"]] == ["query", "retrieve_rerank", "answer"]
        assert row["total_latency_ms"] == pytest.approx(sum(row[k] for k in ["queue_wait_ms", "compute_ms", "transmission_ms", "propagation_ms"]))
        for before, after in zip(row["stages"], row["stages"][1:]):
            assert after["service_start_ms"] >= before["service_finish_ms"]
        assert row["communication_bytes"] == sum(t["transmitted_bytes"] for t in row["transfers"])
    for node in ["edge", "cloud"]:
        ordered = sorted([s for r in rows for s in r["stages"] if s["node_id"] == node], key=lambda s: s["service_start_ms"])
        assert all(a["service_finish_ms"] <= b["service_start_ms"] for a, b in zip(ordered, ordered[1:]))
    assert any(r["queue_wait_ms"] > 0 for r in rows)


def test_network_cost_changes_placement_with_same_answers_and_arrivals() -> None:
    simulation = _simulation()
    a = replay_workflow(_records(), simulation, _scenario(100), "multi_model_rag", "network_aware", 0.1)
    b = replay_workflow(_records(), simulation, _scenario(0.001), "multi_model_rag", "network_aware", 0.1)
    assert a == replay_workflow(_records(), simulation, _scenario(100), "multi_model_rag", "network_aware", 0.1)
    assert [r["arrival_time_ms"] for r in a] == [r["arrival_time_ms"] for r in b]
    assert all(r["correct"] for r in a + b)
    assert a[0]["stages"][-1]["node_id"] == "cloud"
    assert b[0]["stages"][-1]["node_id"] == "edge"
    assert a[0]["communication_bytes"] > b[0]["communication_bytes"]


def test_parallel_capacity_and_same_time_finish_before_enqueue() -> None:
    simulation = _simulation()
    simulation["arrival"]["request_count"] = 3
    one = replay_workflow(_records(), simulation, _scenario(), "direct", "fixed", 1000)
    simulation["nodes"][0]["capacity"] = 2
    two = replay_workflow(_records(), simulation, _scenario(), "direct", "fixed", 1000)
    assert [r["stages"][0]["service_start_ms"] for r in one] == [0, 100, 200]
    assert [r["stages"][0]["service_start_ms"] for r in two] == [0, 1, 100]
    simulation["nodes"][0]["capacity"] = 1
    exact = replay_workflow(_records(), simulation, _scenario(), "direct", "fixed", 10)
    assert [r["queue_wait_ms"] for r in exact] == [0, 0, 0]


def test_replay_rejects_invalid_resources_and_missing_stage_models() -> None:
    simulation = _simulation()
    simulation["nodes"][0]["capacity"] = 0
    with pytest.raises(ValueError, match="capacity"):
        replay_workflow(_records(), simulation, _scenario(), "direct", "fixed", 1)
    simulation = _simulation()
    simulation["nodes"][1]["model_ids"] = []
    with pytest.raises(ValueError, match="fixed node"):
        replay_workflow(_records(), simulation, _scenario(), "multi_model_rag", "fixed", 1)
    records = _records()
    records[0]["stages"][0]["compute_ms"] = float("nan")
    with pytest.raises(ValueError, match="compute_ms"):
        replay_workflow(records, _simulation(), _scenario(), "direct", "fixed", 1)


def test_aggregate_uses_project_p95_and_all_payload_bytes() -> None:
    rows = [{"correct": v == 1, "total_latency_ms": v, "communication_bytes": 2*v,
             "queue_wait_ms": 0, "compute_ms": v, "transmission_ms": 0, "propagation_ms": 0, "stages": []} for v in [1, 101]]
    result = summarize_replay(rows)
    assert result["task_success_rate"] == 0.5
    assert result["latency_ms"] == {"mean": 51, "p95": 96}
    assert result["communication_bytes"]["total"] == 204


def test_rag_cli_entrypoints() -> None:
    parser = build_parser()
    assert parser.parse_args(["prepare-rag", "--config", "c", "--manifest", "m"]).manifest == Path("m")
    assert parser.parse_args(["run-rag", "--config", "c", "--manifest", "m", "--output", "o", "--summary", "s"]).command == "run-rag"
    assert parser.parse_args(["replay-rag-network", "--config", "c", "--inputs", "i", "--output", "o", "--summary", "s"]).inputs == Path("i")
