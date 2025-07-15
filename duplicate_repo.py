"""Utility to duplicate the repository without its ``.git`` folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional


def copy_repo(src: Path, dest: Path, ignore_dirs: Optional[Iterable[str]] = None) -> None:
    """Copy ``src`` to ``dest`` excluding any ``ignore_dirs``."""
    if not src.is_dir():
        raise ValueError(f"Source directory '{src}' does not exist")
    if dest.exists():
        raise ValueError(f"Destination directory '{dest}' already exists")

    ignore_dirs = set(ignore_dirs or (".git",))

    def ignore_func(dirpath: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignore_dirs}

    shutil.copytree(src, dest, ignore=ignore_func)
    print(f"Repository copied from {src} to {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate repository for optimisation")
    parser.add_argument(
        "destination",
        type=Path,
        help="Path to the new copy of the repository",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Source repository directory (defaults to current repo root)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[".git"],
        help="Directory names to exclude while copying",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_repo(args.source, args.destination, args.exclude)


if __name__ == '__main__':
    main()
