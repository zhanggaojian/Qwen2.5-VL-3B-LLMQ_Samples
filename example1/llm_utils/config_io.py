from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm_quant_config.yaml"


def load_yaml_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return config


def save_yaml_config(config: Dict[str, Any], config_path: Optional[Union[str, Path]] = None) -> None:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)


def get_required_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required config section: {section}")
    return value
