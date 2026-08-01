from importlib import import_module
from pathlib import Path


def test_default_yaml_config_can_be_loaded() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")

    config = load_config(Path("configs/mock_qwen35.yaml"))

    assert config["project"]["seed"] == 42
    assert config["scheduler"]["policy"] == "confidence_cascade"
    assert config["scheduler"]["confidence_threshold"] == 0.8
    assert config["scheduler"]["model_order"] == [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ]


def test_real_qwen35_config_is_text_only_and_requests_logprobs() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")

    config = load_config(Path("configs/qwen35_0_8b_vllm.yaml"))

    assert config["backend"]["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert config["backend"]["language_model_only"] is True
    assert config["backend"]["enable_thinking"] is False
    assert config["sampling"]["logprobs"] >= 1
    assert 5 <= len(config["requests"]) <= 20
