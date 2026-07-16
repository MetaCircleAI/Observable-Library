#!/usr/bin/env python3
"""Verify the approved release artifact pair against its manifest."""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifacts(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("manifest must list exactly two artifacts")

    expected = {
        item.get("filename"): item.get("sha256")
        for item in artifacts
        if isinstance(item, dict)
    }
    if len(expected) != 2 or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in expected.items()
    ):
        raise ValueError("manifest must define exactly one sdist and one wheel")
    if (
        sum(name.endswith(".tar.gz") for name in expected) != 1
        or sum(name.endswith(".whl") for name in expected) != 1
    ):
        raise ValueError("manifest must define exactly one sdist and one wheel")
    return expected


def verify(manifest_path: Path, artifact_paths: list[Path]) -> None:
    expected = expected_artifacts(manifest_path)
    provided = {path.name: path for path in artifact_paths}
    if len(provided) != 2 or set(provided) != set(expected):
        raise ValueError("artifacts must exactly match the manifest pair")

    for name, path in provided.items():
        if not path.is_file():
            raise ValueError(f"artifact is missing: {path}")
        if sha256(path) != expected[name]:
            raise ValueError(f"SHA-256 mismatch: {path}")


def main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        print(
            "usage: verify_release_artifacts.py MANIFEST SDIST WHEEL", file=sys.stderr
        )
        return 2
    try:
        verify(Path(arguments[0]), [Path(arguments[1]), Path(arguments[2])])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"release artifact verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
