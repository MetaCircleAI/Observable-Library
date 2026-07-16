from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/verify_release_artifacts.py")


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    sdist = tmp_path / "observable_library-0.1.0.tar.gz"
    wheel = tmp_path / "observable_library-0.1.0-py3-none-any.whl"
    sdist.write_bytes(b"sdist")
    wheel.write_bytes(b"wheel")
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "filename": sdist.name,
                        "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
                    },
                    {
                        "filename": wheel.name,
                        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest, sdist, wheel


def _verify(
    manifest: Path, sdist: Path, wheel: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest), str(sdist), str(wheel)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_verify_release_artifacts_accepts_the_manifest_pair(tmp_path: Path) -> None:
    manifest, sdist, wheel = _write_artifacts(tmp_path)

    result = _verify(manifest, sdist, wheel)

    assert result.returncode == 0, result.stderr


def test_verify_release_artifacts_rejects_a_corrupted_artifact(tmp_path: Path) -> None:
    manifest, sdist, wheel = _write_artifacts(tmp_path)
    wheel.write_bytes(b"corrupted")

    assert _verify(manifest, sdist, wheel).returncode != 0


def test_verify_release_artifacts_rejects_a_missing_artifact(tmp_path: Path) -> None:
    manifest, sdist, wheel = _write_artifacts(tmp_path)
    sdist.unlink()

    assert _verify(manifest, sdist, wheel).returncode != 0


def test_verify_release_artifacts_rejects_an_extra_manifest_artifact(
    tmp_path: Path,
) -> None:
    manifest, sdist, wheel = _write_artifacts(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["artifacts"].append(data["artifacts"][0])
    manifest.write_text(json.dumps(data), encoding="utf-8")

    assert _verify(manifest, sdist, wheel).returncode != 0


def test_verify_release_artifacts_does_not_invoke_twine() -> None:
    assert "twine" not in SCRIPT.read_text(encoding="utf-8").lower()
