"""Small-model FRAMES RAG feasibility measurement, separate from GSM8K phases."""

import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import md5
from html.parser import HTMLParser
import json
from math import log
import os
from pathlib import Path
from random import Random
import re
from statistics import fmean
import string
import subprocess
import sys
import tarfile
from time import perf_counter
from typing import Any
from urllib.parse import unquote
from urllib.request import urlopen

from scaleflow.baseline import file_sha256, percentile


def json_bytes(value: Any) -> bytes:
    """The exact application payload used by network replay (no protocol headers)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_frames(path: Path, count: int, seed: int) -> tuple[list[dict], int]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(rows):
        raise ValueError("FRAMES sample_count must be between 1 and dataset size")
    indices = sorted(Random(seed).sample(range(len(rows)), count))
    samples = [{
        "sample_id": str(rows[index].get("", index)), "row_index": index,
        "question": rows[index]["Prompt"].strip(), "gold_answer": rows[index]["Answer"].strip(),
        "reference_urls": ast.literal_eval(rows[index]["wiki_links"]),
        "reasoning_types": rows[index].get("reasoning_types", ""),
    } for index in indices]
    return samples, len(rows)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"p", "li", "tr", "h2", "h3", "h4", "br"} and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"td", "th"} and not self.skip:
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def html_text(html: str) -> str:
    start = html.find("mw-parser-output")
    if start >= 0:
        html = html[html.find(">", start) + 1:]
    end = html.find('class="printfooter"')
    if end >= 0:
        html = html[:end]
    parser = _TextParser()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def prepare_rag(config: dict, manifest_path: Path) -> dict:
    if manifest_path.exists():
        raise ValueError(f"manifest already exists: {manifest_path}")
    dataset = config["dataset"]
    corpus = config["corpus"]
    sources = []
    for item in (dataset, corpus["archive"], corpus["mapping"]):
        path = Path(item["path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(item["url"], timeout=120) as response:
                temporary = path.with_suffix(path.suffix + ".tmp")
                with temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                temporary.replace(path)
        digest = md5(path.read_bytes()).hexdigest()
        if digest != item["md5"]:
            raise ValueError(f"source checksum mismatch: {path}")
        sources.append({"url": item["url"], "path": str(path), "sha256": file_sha256(path), "md5": digest})
    samples, population = select_frames(Path(dataset["path"]), dataset["sample_count"], dataset["seed"])
    mapping = json.loads(Path(corpus["mapping"]["path"]).read_text())
    index_path = Path(corpus["index_path"])
    if index_path.exists():
        raise ValueError(f"corpus index already exists: {index_path}")
    words_per_chunk, overlap = corpus["chunk_words"], corpus["overlap_words"]
    if not 0 <= overlap < words_per_chunk:
        raise ValueError("corpus overlap_words must be smaller than chunk_words")
    document_count = passage_count = 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(corpus["archive"]["path"], "r|gz") as archive, index_path.open("w") as stream:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".html"):
                continue
            key = Path(member.name).stem
            url = mapping.get(key)
            if url is None:
                raise ValueError(f"frozen document missing URL mapping: {key}")
            text = html_text(archive.extractfile(member).read().decode("utf-8"))
            words = text.split()
            title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
            document_count += 1
            for start in range(0, len(words), words_per_chunk - overlap):
                passage = {"passage_id": f"{key}:{start}", "url": url, "title": title, "text": " ".join(words[start:start + words_per_chunk])}
                stream.write(json.dumps(passage, ensure_ascii=False) + "\n")
                passage_count += 1
    manifest = {
        "dataset": "google/frames-benchmark", "selection_method": "sorted(random.Random(seed).sample(row_indices, count))",
        "sample_count": len(samples), "population_count": population, "seed": dataset["seed"],
        "samples": samples, "sources": sources, "index_sha256": file_sha256(index_path),
        "corpus": {"scope": "entire MLCommons frozen Wikipedia corpus; no per-question reference filtering",
                   "document_count": document_count, "passage_count": passage_count,
                   "chunk_words": words_per_chunk, "overlap_words": overlap,
                   "index_path": str(index_path), "html_extractor": "mw-parser-output text incl. table cells; script/style removed"},
    }
    write_json(manifest_path, manifest)
    return manifest


_STOP = set("a an the is are was were be been of to in on at for from and or by with as which what who when where how that this it its i my me would did does do has have had between than".split())


def tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOP]


class BM25Retriever:
    """Inverted-index BM25 + RRF query fusion and deterministic lexical reranking."""
    def __init__(self, passages: list[dict], config: dict) -> None:
        if not passages:
            raise ValueError("RAG corpus is empty")
        self.passages = passages
        self.config = config
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, passage in enumerate(passages):
            counts = Counter(tokens(passage["title"] + " " + passage["text"]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.mean_length = fmean(self.lengths)

    def _search(self, query: str) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokens(query))):
            entries = self.postings.get(term, [])
            idf = log(1 + (len(self.passages) - len(entries) + 0.5) / (len(entries) + 0.5))
            for index, frequency in entries:
                denominator = frequency + 1.2 * (0.25 + 0.75 * self.lengths[index] / self.mean_length)
                scores[index] += idf * frequency * 2.2 / denominator
        return sorted(scores, key=lambda index: (-scores[index], self.passages[index]["passage_id"]))[:self.config["candidates_per_query"]]

    def retrieve(self, question: str, queries: list[str]) -> list[dict]:
        fused: dict[int, float] = defaultdict(float)
        question_terms = set(tokens(question))
        for query in dict.fromkeys([question] + queries):
            for rank, index in enumerate(self._search(query)):
                fused[index] += 1 / (60 + rank + 1)
        def score(index: int) -> float:
            overlap = len(question_terms & set(tokens(self.passages[index]["text"]))) / max(1, len(question_terms))
            return fused[index] + self.config["rerank_overlap_weight"] * overlap
        ordered = sorted(fused, key=lambda index: (-score(index), self.passages[index]["passage_id"]))
        counts: Counter = Counter()
        selected = []
        for index in ordered:
            passage = self.passages[index]
            if counts[passage["url"]] >= self.config["max_passages_per_document"]:
                continue
            counts[passage["url"]] += 1
            selected.append(passage)
            if len(selected) == self.config["top_k"]:
                break
        return selected


def parse_queries(text: str, question: str, limit: int) -> tuple[list[str], bool]:
    try:
        start, end = text.index("["), text.rindex("]") + 1
        values = json.loads(text[start:end])
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("invalid queries")
        return list(dict.fromkeys(v.strip() for v in values))[:limit], False
    except (ValueError, TypeError):
        return [question], True


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction: str, gold: str) -> bool:
    """Strict normalized exact match; deliberately no LLM judge or substring credit."""
    return bool(normalize_answer(gold)) and normalize_answer(prediction) == normalize_answer(gold)


def answer_text(text: str) -> str:
    # Prompts request a single FINAL line; preserve full raw output for auditing.
    matches = re.findall(r"(?im)^\s*FINAL:\s*(.+)$", text)
    return matches[-1].strip() if matches else text.strip()


def generation_diagnostics(response: dict, budget: int | None) -> dict:
    """Keep budget exhaustion observable without guessing vLLM finish reasons."""
    count = len(response.get("token_logprobs", []))
    return {"generated_token_count": count,
            "output_budget_reached": count >= budget if budget is not None else None,
            "has_final_line": bool(re.search(r"(?im)^\s*FINAL:\s*\S", response["text"]))}


def run_vllm_batch(model: dict, requests: list[dict], workdir: Path, name: str, backend_common: dict) -> dict:
    """Launch existing VLLMBackend in an isolated process, with local pinned weights."""
    job_path, out_path = workdir / f"{name}_job.json", workdir / f"{name}_outputs.jsonl"
    backend = dict(backend_common)
    backend.update(model["backend"])
    write_json(job_path, {"backend": backend, "requests": requests})
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(model["gpu"]), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    with (workdir / f"{name}.log").open("w") as log_stream:
        subprocess.run([sys.executable, "-m", "scaleflow.rag_worker", str(job_path), str(out_path)], env=env, stdout=log_stream, stderr=subprocess.STDOUT, check=True)
    outputs = {row["request_id"]: row for row in load_jsonl(out_path)}
    if set(outputs) != {request["request_id"] for request in requests}:
        raise ValueError("vLLM worker outputs do not align with requests")
    return outputs


def execute_rag(config: dict, manifest: dict, retriever: BM25Retriever, batch: Any) -> list[dict]:
    """Measure direct, same-model RAG, and different-model RAG on identical samples."""
    samples = manifest["samples"]
    stages = config["stages"]
    if stages["multi_query_model"] == stages["answer_model"]:
        raise ValueError("multi-model RAG requires different query and answer models")
    if not isinstance(stages["max_queries"], int) or stages["max_queries"] < 1:
        raise ValueError("max_queries must be a positive integer")
    query_models = dict.fromkeys([stages["multi_query_model"], stages["answer_model"]])
    def budget(model_id: str) -> int | None:
        backend = dict(config.get("measurement", {}).get("backend_common", {}))
        backend.update(next((m["backend"] for m in config.get("models", []) if m["model_id"] == model_id), {}))
        return backend.get("max_tokens")
    query_outputs = {}
    for model_id in query_models:
        requests = [{"request_id": s["sample_id"], "prompt": config["prompts"]["query"].replace("{question}", s["question"])} for s in samples]
        query_outputs[model_id] = batch(model_id, requests, "queries-" + model_id.split("/")[-1])
    cases = []
    answer_requests = []
    for mode in ("direct", "single_node_rag", "multi_model_rag"):
        for sample in samples:
            question = sample["question"]
            measurements = []
            evidence = []
            fallback = False
            if mode != "direct":
                query_id = stages["answer_model"] if mode == "single_node_rag" else stages["multi_query_model"]
                query_result = query_outputs[query_id][sample["sample_id"]]
                response = query_result["response"]
                queries, fallback = parse_queries(response["text"], question, stages["max_queries"])
                request_payload = {"question": question}
                query_payload = {"question": question, "queries": queries}
                measurements.append({"stage": "query", "model_id": query_id, "compute_ms": response["latency_ms"], "success": response["success"], "error": response.get("error"), "input": request_payload, "output": query_payload, "raw_output": response["text"], "model_info": query_result["model_info"]})
                measurements[-1].update(generation_diagnostics(response, budget(query_id)))
                started = perf_counter()
                evidence = retriever.retrieve(question, queries)
                retrieval_ms = (perf_counter() - started) * 1000
                evidence_payload = {"question": question, "evidence": evidence}
                measurements.append({"stage": "retrieve_rerank", "model_id": "bm25-rrf-lexical", "compute_ms": retrieval_ms, "success": bool(evidence), "input": query_payload, "output": evidence_payload})
                context = "\n\n".join(f"[{i+1}] {p['title']} ({p['url']})\n{p['text']}" for i, p in enumerate(evidence))
                prompt = config["prompts"]["rag_answer"].replace("{question}", question).replace("{context}", context)
                final_input = evidence_payload
            else:
                final_input = {"question": question}
                prompt = config["prompts"]["direct"].replace("{question}", question)
            request_id = f"{mode}:{sample['sample_id']}"
            answer_requests.append({"request_id": request_id, "prompt": prompt})
            cases.append({"sample_id": sample["sample_id"], "mode": mode, "question": question, "gold_answer": sample["gold_answer"], "reference_urls": sample["reference_urls"], "query_fallback": fallback, "stages": measurements, "answer_input": final_input, "answer_prompt": prompt, "retrieved_reference_count": len({p["url"] for p in evidence} & set(sample["reference_urls"]))})
    answers = batch(stages["answer_model"], answer_requests, "answers")
    for case in cases:
        item = answers[f"{case['mode']}:{case['sample_id']}"]
        response = item["response"]
        final = answer_text(response["text"])
        case["stages"].append({"stage": "answer", "model_id": stages["answer_model"], "compute_ms": response["latency_ms"], "success": response["success"], "error": response.get("error"), "input": case.pop("answer_input"), "output": {"answer": final}, "raw_output": response["text"], "model_info": item["model_info"]})
        case["stages"][-1].update(generation_diagnostics(response, budget(stages["answer_model"])))
        case["predicted_answer"] = final
        case["pipeline_success"] = all(stage["success"] for stage in case["stages"])
        case["correct"] = case["pipeline_success"] and score_answer(final, case["gold_answer"])
        case["measurement_type"] = "real_vllm_and_cpu_retrieval"
        case["compute_ms"] = sum(stage["compute_ms"] for stage in case["stages"])
        for stage in case["stages"]:
            stage["input_bytes"] = len(json_bytes(stage["input"]))
            stage["output_bytes"] = len(json_bytes(stage["output"]))
    return cases


def summarize_measurements(records: list[dict]) -> dict:
    return {mode: {
        "task_count": len(rows), "task_success_rate": fmean(row["correct"] for row in rows),
        "pipeline_execution_success_rate": fmean(row["pipeline_success"] for row in rows),
        "cumulative_compute_ms": {"mean": fmean(row["compute_ms"] for row in rows), "p95": percentile([row["compute_ms"] for row in rows], 95)},
        "query_fallback_count": sum(row["query_fallback"] for row in rows),
        "answer_missing_final_line_count": sum(not row["stages"][-1].get("has_final_line", bool(re.search(r"(?im)^\s*FINAL:\s*\S", row["stages"][-1]["raw_output"]))) for row in rows),
        "answer_output_budget_reached_count": sum(row["stages"][-1].get("output_budget_reached") is True for row in rows),
        "retrieved_reference_count_mean": fmean(row["retrieved_reference_count"] for row in rows),
    } for mode in dict.fromkeys(r["mode"] for r in records) if (rows := [r for r in records if r["mode"] == mode])}


def measurement_report(config: dict, manifest: dict, records: list[dict],
                       config_path: Path, manifest_path: Path, output: Path) -> dict:
    """Build an auditable report from saved measurements without running models."""
    samples = manifest["samples"]
    return {
        "result_type": "real_vllm_stage_measurements_and_cpu_retrieval",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config_path), "manifest_sha256": file_sha256(manifest_path),
        "records_sha256": file_sha256(output), "sample_count": len(samples),
        "sources": manifest["sources"], "corpus": manifest["corpus"],
        "sampling": {"population_count": manifest["population_count"], "seed": manifest["seed"],
                     "method": manifest["selection_method"], "sample_ids": [s["sample_id"] for s in samples]},
        "models": config["models"], "stages": config["stages"], "retrieval": config["retrieval"],
        "scoring": "normalized exact match after FINAL extraction; lowercase/punctuation/articles/whitespace normalization",
        "scoring_source": "Reproducible local fallback: MLCommons accuracy_eval.py defaults to gpt-oss-20b LLM judge; no judge endpoint configured and large-model deployment excluded by experiment scope",
        "measurement_contract": {"model_load_and_warmup_excluded": True, "sequential_batches": True,
                                 "latency_scope": "sum of measured query model, CPU retrieval/reranking, answer model service times; isolated stages, not full online E2E wall clock",
                                 "real_distributed_online_measurement": False, "actual_network_measurement": False,
                                 "protocol_overhead_included": False, "payload_format": "compact sorted UTF-8 JSON",
                                 "payload_assumption": "question, parsed queries, evidence, final answer; stage code/prompt templates preinstalled on nodes; raw model reasoning stays local",
                                 "query_prompt": config["prompts"]["query"], "answer_prompt": config["prompts"]["rag_answer"]},
        "metrics": summarize_measurements(records),
        "limitations": ["Small-sample pilot is not a full FRAMES or MLPerf result", "Exact match penalizes valid paraphrases and alternate units",
                        "BM25/RRF/lexical reranking replaces dense/ColBERT retrieval", "One retrieval round; no iterative sufficiency checking",
                        "Network and heterogeneous node speed/capacity are replay assumptions", "Model load, ingestion/index construction and orchestration overhead outside task latency",
                        "Short warmup does not guarantee all JIT kernels are compiled; residual compilation can inflate early measured requests",
                        "Fixed generation budget; budget reached is a token-count diagnostic, not a recorded vLLM finish reason"],
    }


def run_rag(config: dict, config_path: Path, manifest_path: Path, output: Path, summary: Path) -> dict:
    if output.exists() or summary.exists():
        raise ValueError("RAG outputs already exist; choose a fresh run path")
    manifest = json.loads(manifest_path.read_text())
    index_path = Path(config["corpus"]["index_path"])
    if manifest["index_sha256"] != file_sha256(index_path):
        raise ValueError("RAG corpus index fingerprint mismatch")
    samples, _ = select_frames(Path(config["dataset"]["path"]), config["dataset"]["sample_count"], config["dataset"]["seed"])
    if samples != manifest["samples"]:
        raise ValueError("RAG sample manifest no longer matches configuration/dataset")
    # Separate worker artifacts for every output path, including repeated trials.
    workdir = Path(config["measurement"]["workdir"]) / output.stem
    if workdir.exists():
        raise ValueError(f"RAG worker directory already exists; choose a fresh output path: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)
    retriever = BM25Retriever(load_jsonl(index_path), config["retrieval"])
    models = {item["model_id"]: item for item in config["models"]}
    def batch(model_id: str, requests: list[dict], name: str) -> dict:
        return run_vllm_batch(models[model_id], requests, workdir, name, config["measurement"]["backend_common"])
    records = execute_rag(config, manifest, retriever, batch)
    write_jsonl(output, records)
    report = measurement_report(config, manifest, records, config_path, manifest_path, output)
    write_json(summary, report)
    return report
