"""Utility functions for loading and validating YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import os

import yaml


_REQUIRED_KEYS: dict[str, Any] = {
    "data": ["path", "batch_size"],
    "training": ["epochs", "learning_rate", "early_stop_patience"],
    "loss_configs": list,
    "model": dict,
    "attention_types": list,
}


def validate_config(cfg: dict[str, Any]) -> None:
    """Basic validation of the parsed configuration."""
    for section, req in _REQUIRED_KEYS.items():
        if section not in cfg:
            raise KeyError(f"Missing required section '{section}' in config")
        if isinstance(req, list):
            for key in req:
                if key not in cfg[section]:
                    raise KeyError(f"Missing key '{section}.{key}' in config")


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML file. ``~`` and environment variables are expanded.

    Returns:
        Parsed configuration dictionary.
    """
    # Expand and normalise the path in case a string is provided
    path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)
    return cfg

