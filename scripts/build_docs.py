from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run_sphinx(source_dir: Path, output_dir: Path, language: str) -> None:
    environment = os.environ.copy()
    environment["DOCS_LANGUAGE"] = language
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            "-E",
            "-a",
            str(source_dir),
            str(output_dir),
        ],
        check=True,
        env=environment,
    )


def build_site(repo_root: Path, output_dir: Path) -> None:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo_root or output_dir == Path(output_dir.anchor):
        raise ValueError(
            "documentation output must not be the repository or filesystem root"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    source_dir = repo_root / "docs"
    _run_sphinx(source_dir, output_dir, "en")
    _run_sphinx(source_dir, output_dir / "zh", "zh_CN")
    (output_dir / ".nojekyll").touch()

    if not (output_dir / "404.html").is_file():
        raise RuntimeError("English build did not produce 404.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bilingual Pages artifact")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/web-docs"),
        help="artifact directory (default: build/web-docs)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    build_site(repo_root, args.output_dir)


if __name__ == "__main__":
    main()
