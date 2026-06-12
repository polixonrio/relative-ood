"""Restore upstream OpenOOD source files for the thesis analysis workspace.

The analysis code is kept in this repository. OpenOOD is treated as upstream
source and can be restored into the repository root when the analysis needs to
run. If local OpenOOD changes become necessary, store them as patch files under
analysis/openood_patches and this script will apply them before copying files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = Path(__file__).resolve().parents[1] / "openood_patches"
UPSTREAM_URL = "https://github.com/Jingkang50/OpenOOD.git"
DEFAULT_REF = os.environ.get("OPENOOD_REF", "main")
DEFAULT_PATHS = (
    "openood",
    "configs",
    "postprocessors",
    "tools",
    "tests",
    "main.py",
    "setup.py",
    "pyproject.toml",
    "poetry.lock",
    "imglist_generator.py",
    "scripts/download",
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def copy_path(source_root: Path, rel_path: str, *, force: bool) -> None:
    source = source_root / rel_path
    target = REPO_ROOT / rel_path
    if not source.exists():
        raise FileNotFoundError(f"Upstream path does not exist: {rel_path}")
    if target.exists():
        if not force:
            raise FileExistsError(
                f"{rel_path} already exists. Re-run with --force to replace it."
            )
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def apply_patches(source_root: Path, patch_dir: Path) -> list[Path]:
    patches = sorted(patch_dir.glob("*.patch"))
    for patch in patches:
        run(["git", "apply", str(patch.resolve())], cwd=source_root)
    return patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=UPSTREAM_URL)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="Upstream path to copy. May be repeated. Defaults to the core OpenOOD source paths.",
    )
    parser.add_argument(
        "--include-scripts",
        action="store_true",
        help="Also copy the upstream scripts/ tree. This can replace any local scripts/ tree when --force is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing target files or directories.",
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Do not apply analysis/openood_patches/*.patch before copying.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.paths or DEFAULT_PATHS)
    if args.include_scripts and "scripts" not in paths:
        paths.append("scripts")

    with tempfile.TemporaryDirectory(prefix="openood-src-") as temp_dir:
        source_root = Path(temp_dir) / "OpenOOD"
        run(["git", "clone", "--filter=blob:none", args.repo_url, str(source_root)])
        run(["git", "checkout", args.ref], cwd=source_root)

        patches: list[Path] = []
        if not args.no_patches:
            patches = apply_patches(source_root, PATCH_DIR)

        for rel_path in paths:
            copy_path(source_root, rel_path, force=args.force)

    print()
    print(f"Restored OpenOOD from {args.repo_url} at {args.ref}.")
    if patches:
        print("Applied patches:")
        for patch in patches:
            print(f"  {patch.relative_to(REPO_ROOT)}")
    else:
        print("No OpenOOD patches were applied.")
    print("Copied paths:")
    for rel_path in paths:
        print(f"  {rel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
