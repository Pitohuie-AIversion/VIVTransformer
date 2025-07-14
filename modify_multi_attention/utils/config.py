"""Utility functions for loading and validating YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import os

import yaml


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
        return yaml.safe_load(f)

