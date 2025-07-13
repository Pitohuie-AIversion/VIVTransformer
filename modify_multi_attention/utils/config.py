"""Utility functions for loading and validating YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

