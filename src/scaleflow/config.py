from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a ScaleFlow YAML configuration cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in configuration: {config_path}") from error

    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    return data
