"""Isolated real vLLM batch worker; process exit releases model resources."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

from scaleflow.backends import VLLMBackend
from scaleflow.schemas import InferenceRequest


def main() -> None:
    job_path, output_path = map(Path, sys.argv[1:])
    job = json.loads(job_path.read_text())
    backend = VLLMBackend(**job["backend"])
    warmup = backend.generate(InferenceRequest("warmup", "Reply with the word ready."))
    if not warmup.success:
        raise RuntimeError(warmup.error)
    with output_path.open("x", encoding="utf-8") as stream:
        for item in job["requests"]:
            response = backend.generate(InferenceRequest(item["request_id"], item["prompt"]))
            stream.write(json.dumps({
                "request_id": item["request_id"],
                "response": asdict(response),
                "model_info": backend.get_model_info(),
            }, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{item['request_id']}: success={response.success}, {response.latency_ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
