import os
import shutil
import argparse


def copy_repo(src_dir: str, dest_dir: str) -> None:
    """Copy repository content to a new directory without the .git folder."""
    if not os.path.isdir(src_dir):
        raise ValueError(f"Source directory '{src_dir}' does not exist")
    if os.path.exists(dest_dir):
        raise ValueError(f"Destination directory '{dest_dir}' already exists")

    def ignore_git(dirpath, names):
        return {'.git'} if '.git' in names else set()

    shutil.copytree(src_dir, dest_dir, ignore=ignore_git)
    print(f"Repository copied from {src_dir} to {dest_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate repository for optimisation")
    parser.add_argument('destination', help="Path to the new copy of the repository")
    parser.add_argument('--source', default=os.path.dirname(os.path.abspath(__file__)), help="Source repository directory (defaults to current repo root)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_repo(args.source, args.destination)


if __name__ == '__main__':
    main()
