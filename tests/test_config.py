from importlib import import_module
from pathlib import Path
import random


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


def test_gsm8k_baseline_config_pins_data_samples_prompt_and_generation() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")

    config = load_config(Path("configs/qwen35_0_8b_gsm8k.yaml"))

    assert config["project"]["seed"] == 42
    assert config["dataset"]["commit"] == (
        "3101c7d5072418e28b9008a6636bde82a006892c"
    )
    assert config["dataset"]["sha256"] == (
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    )
    assert config["dataset"]["expected_record_count"] == 1319
    assert config["dataset"]["selection_method"] == "python_random_sample"
    assert config["dataset"]["selection_seed"] == 42
    assert config["dataset"]["sample_indices"] == random.Random(42).sample(
        range(1319), 64
    )
    assert len(set(config["dataset"]["sample_indices"])) == 64
    assert config["prompt"]["template"].count("{question}") == 1
    assert "#### <number>" in config["prompt"]["template"]
    assert len(config["warmup"]["prompts"]) == 8
    assert config["backend"]["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert config["backend"]["max_model_len"] == 2048
    assert config["backend"]["gpu_memory_utilization"] == 0.25
    assert config["backend"]["enable_prefix_caching"] is False
    assert config["sampling"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 384,
        "logprobs": 1,
    }


def test_qwen35_2b_config_reuses_the_exact_gsm8k_experiment_contract() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")
    baseline = load_config(Path("configs/qwen35_0_8b_gsm8k.yaml"))
    candidate = load_config(Path("configs/qwen35_2b_gsm8k.yaml"))

    for section in ("dataset", "prompt", "warmup", "sampling"):
        assert candidate[section] == baseline[section]

    shared_backend_fields = (
        "language_model_only",
        "enable_thinking",
        "dtype",
        "max_model_len",
        "enforce_eager",
        "enable_prefix_caching",
    )
    for field in shared_backend_fields:
        assert candidate["backend"][field] == baseline["backend"][field]

    assert candidate["project"] == baseline["project"]
    assert candidate["backend"]["model_id"] == "Qwen/Qwen3.5-2B"
    assert candidate["backend"]["revision"] == (
        "15852e8c16360a2fea060d615a32b45270f8a8fc"
    )
    assert 0 < candidate["backend"]["gpu_memory_utilization"] <= 1


def test_qwen35_4b_config_reuses_the_exact_gsm8k_experiment_contract() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")
    baseline = load_config(Path("configs/qwen35_0_8b_gsm8k.yaml"))
    candidate = load_config(Path("configs/qwen35_4b_gsm8k.yaml"))

    for section in ("dataset", "prompt", "warmup", "sampling"):
        assert candidate[section] == baseline[section]

    shared_backend_fields = (
        "language_model_only",
        "enable_thinking",
        "dtype",
        "max_model_len",
        "enforce_eager",
        "enable_prefix_caching",
    )
    for field in shared_backend_fields:
        assert candidate["backend"][field] == baseline["backend"][field]

    assert candidate["project"] == baseline["project"]
    assert candidate["backend"]["model_id"] == "Qwen/Qwen3.5-4B"
    assert candidate["backend"]["revision"] == (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert 0 < candidate["backend"]["gpu_memory_utilization"] <= 1


def test_qwen35_9b_config_reuses_the_exact_gsm8k_experiment_contract() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")
    baseline = load_config(Path("configs/qwen35_0_8b_gsm8k.yaml"))
    candidate = load_config(Path("configs/qwen35_9b_gsm8k.yaml"))

    for section in ("dataset", "prompt", "warmup", "sampling"):
        assert candidate[section] == baseline[section]

    shared_backend_fields = (
        "language_model_only",
        "enable_thinking",
        "dtype",
        "max_model_len",
        "enforce_eager",
        "enable_prefix_caching",
    )
    for field in shared_backend_fields:
        assert candidate["backend"][field] == baseline["backend"][field]

    assert candidate["project"] == baseline["project"]
    assert candidate["backend"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert candidate["backend"]["revision"] == (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert candidate["backend"]["gpu_memory_utilization"] == 0.90


def test_full_gsm8k_configs_use_all_records_and_preserve_model_contracts() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")
    model_files = (
        ("0_8b", "Qwen/Qwen3.5-0.8B"),
        ("2b", "Qwen/Qwen3.5-2B"),
        ("4b", "Qwen/Qwen3.5-4B"),
        ("9b", "Qwen/Qwen3.5-9B"),
    )

    for suffix, model_id in model_files:
        baseline = load_config(Path(f"configs/qwen35_{suffix}_gsm8k.yaml"))
        full = load_config(Path(f"configs/qwen35_{suffix}_gsm8k_full.yaml"))

        assert full["dataset"]["selection_method"] == "all_records"
        assert "selection_seed" not in full["dataset"]
        assert "sample_indices" not in full["dataset"]
        assert full["dataset"]["expected_record_count"] == 1319
        for field in (
            "name",
            "split",
            "commit",
            "source_url",
            "local_path",
            "sha256",
            "expected_record_count",
        ):
            assert full["dataset"][field] == baseline["dataset"][field]
        assert full["backend"]["model_id"] == model_id
        assert full["backend"] == baseline["backend"]
        for section in ("project", "prompt", "warmup", "sampling"):
            assert full[section] == baseline[section]


def test_existing_gsm8k_64_experiment_fingerprint_is_unchanged() -> None:
    cli_module = import_module("scaleflow.cli")
    config_module = import_module("scaleflow.config")
    config = getattr(config_module, "load_config")(
        Path("configs/qwen35_0_8b_gsm8k.yaml")
    )
    experiment_config = getattr(cli_module, "_common_experiment_config")(config)
    fingerprint = getattr(cli_module, "_experiment_fingerprint")(
        experiment_config
    )

    assert fingerprint == (
        "559d16675569b089f98f8cd453b88936f4b183e43ebc5938585c994a20bc1554"
    )


def test_phase10_routing_config_freezes_models_split_and_lightweight_baselines() -> None:
    config_module = import_module("scaleflow.config")
    load_config = getattr(config_module, "load_config")
    config = load_config(Path("configs/qwen35_gsm8k_routing.yaml"))

    assert config["project"] == {"name": "ScaleFlow", "seed": 42}
    assert config["phase8_split"] == {
        "report_path": "results/phase8_confidence_development.json",
        "method": "sha256_seed_sample_id",
        "seed": 42,
        "development_count": 660,
        "evaluation_count": 659,
    }
    assert config["models"] == [
        {
            "model_id": "Qwen/Qwen3.5-2B",
            "label": "2B",
        },
        {
            "model_id": "Qwen/Qwen3.5-4B",
            "label": "4B",
        },
        {
            "model_id": "Qwen/Qwen3.5-9B",
            "label": "9B",
        },
    ]
    assert config["rule"]["weights"] == {
        "char_count": 0.01,
        "word_count": 1.0,
        "number_count": 4.0,
        "operator_count": 2.0,
        "keyword_count": 3.0,
    }
    assert config["rule"]["threshold_quantiles"] == [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]
    assert config["tfidf"] == {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_features": 5000,
        "sublinear_tf": True,
        "lowercase": True,
    }
    assert config["logistic_regression"] == {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }
    assert config["random_baseline"] == {
        "seed_start": 1000,
        "seed_count": 1000,
    }
