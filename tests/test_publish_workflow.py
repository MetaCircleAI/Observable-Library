from __future__ import annotations

import re
from pathlib import Path


SOURCE_SHA = "57a5790d0e4c6665dcfab5f66ff5243adb6a3851"
WORKFLOW_PATH = Path(".github/workflows/publish.yml")


def _workflow() -> str:
    assert WORKFLOW_PATH.exists(), "publish workflow is missing"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        workflow,
    )
    assert match, f"publish job {name!r} is missing"
    return match.group("body")


def _permissions(job: str) -> str:
    match = re.search(
        r"(?ms)^    permissions:\n(?P<body>.*?)(?=^    steps:|^    outputs:)",
        job,
    )
    assert match, "job permissions are missing"
    return match.group("body").strip()


def _assert_downloads_build_artifacts(job: str) -> None:
    assert "actions/download-artifact@v4" in job
    assert "name: release-dists" in job
    assert "path: dist" in job
    assert "needs.build.outputs.wheel_sha256" in job
    assert "needs.build.outputs.sdist_sha256" in job
    assert "python -m build" not in job


def test_build_job_creates_one_hashed_artifact_pair_from_the_accepted_source() -> None:
    workflow = _workflow()
    build = _job(workflow, "build")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert SOURCE_SHA in workflow
    assert "git rev-parse HEAD" in build
    assert "python -m build" in build
    assert "python -m twine check --strict dist/*" in build
    assert "actions/upload-artifact@v4" in build
    assert "id-token:" not in build
    assert _permissions(build) == "contents: read"
    assert workflow.count("python -m build") == 1


def test_testpypi_verification_precedes_production_and_only_publish_jobs_get_oidc() -> (
    None
):
    workflow = _workflow()
    build = _job(workflow, "build")
    publish_testpypi = _job(workflow, "publish-testpypi")
    verify_testpypi = _job(workflow, "verify-testpypi")
    publish_production = _job(workflow, "publish-production")
    verify_production = _job(workflow, "verify-production")

    assert "environment: testpypi" in publish_testpypi
    assert _permissions(publish_testpypi) == "id-token: write"
    assert "repository-url: https://test.pypi.org/legacy/" in publish_testpypi
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_testpypi
    _assert_downloads_build_artifacts(publish_testpypi)

    assert "needs: publish-testpypi" in verify_testpypi
    assert "id-token:" not in verify_testpypi
    assert "numpy>=1.24" in verify_testpypi
    assert "torch>=2.4.1" in verify_testpypi
    assert "--no-deps --index-url https://test.pypi.org/simple" in verify_testpypi
    assert 'PIP_EXTRA_INDEX_URL: ""' in verify_testpypi
    assert "--extra-index-url" not in verify_testpypi
    assert "for attempt in 1 2 3 4 5" in verify_testpypi
    assert "pip check" in verify_testpypi
    assert "Runtime" in verify_testpypi

    assert "needs: [build, verify-testpypi]" in publish_production
    assert "environment: pypi-production" in publish_production
    assert _permissions(publish_production) == "id-token: write"
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_production
    _assert_downloads_build_artifacts(publish_production)

    assert "needs: publish-production" in verify_production
    assert "id-token:" not in verify_production
    assert "--index-url https://pypi.org/simple" in verify_production
    assert 'PIP_EXTRA_INDEX_URL: ""' in verify_production
    assert "--extra-index-url" not in verify_production
    assert "observable-library==0.1.0" in verify_production
    assert "pip check" in verify_production
    assert "Runtime" in verify_production

    assert "id-token: write" not in build
    assert workflow.count("id-token: write") == 2

    for forbidden in [
        "skip-existing",
        "twine_password",
        "twine_username",
        "api-token",
        "github-release",
        "secrets",
        "github_token",
        "password",
        "username",
        "api_token",
    ]:
        assert forbidden not in workflow.lower()
    assert not re.search(r"(?m)^\s+token:", workflow)
