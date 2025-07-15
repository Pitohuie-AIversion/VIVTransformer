"""Logging helpers for training scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: Path | None = None) -> None:
    """Configure root logger.

    Args:
        log_file: Optional path to a file to write logs.
    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
