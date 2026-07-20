from __future__ import annotations

import re
from pathlib import Path


SOURCE_SHA = "57a5790d0e4c6665dcfab5f66ff5243adb6a3851"
WORKFLOW_PATH = Path(".github/workflows/publish.yml")


def _workflow() -> str:
    assert WORKFLOW_PATH.exists(), "production publish workflow is missing"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        workflow,
    )
    assert match, f"publish job {name!r} is missing"
    return match.group("body")


def test_publish_workflow_builds_the_accepted_source_and_uses_oidc() -> None:
    workflow = _workflow()
    build = _job(workflow, "build")
    publish = _job(workflow, "publish")
    verify = _job(workflow, "verify-production")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert SOURCE_SHA in workflow
    assert "git rev-parse HEAD" in build
    assert "python -m build" in build
    assert "python -m twine check --strict dist/*" in build
    assert "actions/upload-artifact@v4" in build
    assert "id-token:" not in build

    assert "environment: pypi-production" in publish
    permissions = re.search(
        r"(?ms)^    permissions:\n(?P<body>.*?)(?=^    steps:)", publish
    )
    assert permissions and permissions.group("body").strip() == "id-token: write"
    assert "pypa/gh-action-pypi-publish@release/v1" in publish
    assert "actions/download-artifact@v4" in publish

    assert "id-token:" not in verify
    assert "pip install" in verify
    assert "observable-library==0.1.0" in verify
    assert "pip check" in verify
    assert "Runtime" in verify

    for forbidden in [
        "testpypi",
        "skip-existing",
        "twine_password",
        "twine_username",
        "api-token",
        "github-release",
    ]:
        assert forbidden not in workflow.lower()
