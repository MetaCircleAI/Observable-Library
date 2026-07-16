from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path


WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def _workflow() -> str:
    assert WORKFLOW_PATH.exists(), "CI workflow is missing"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        workflow,
    )
    assert match, f"CI job {name!r} is missing"
    return match.group("body")


def _step_run(job: str, name: str) -> str:
    step = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - |\Z)",
        job,
    )
    assert step, f"CI step {name!r} is missing"
    run = re.search(r"(?ms)^        run: \|\n(?P<script>.*)\Z", step.group("body"))
    assert run, f"CI step {name!r} has no block run payload"
    return textwrap.dedent(run.group("script"))


def test_required_and_advisory_test_jobs_match_supported_versions() -> None:
    workflow = _workflow()
    current = _job(workflow, "current-tests")
    minimum = _job(workflow, "minimum-tests")
    advisory = _job(workflow, "advisory-tests")

    for version in ["3.10", "3.11", "3.12"]:
        assert version in current
    assert "torch==" not in current
    assert "python -m pytest tests -q" in current

    assert 'python-version: "3.10"' in minimum
    assert "torch==2.4.1" in minimum
    assert 'assert torch.__version__.split("+")[0] == "2.4.1"' in minimum
    assert "python -m pytest tests -q" in minimum

    assert "continue-on-error: true" in advisory
    assert 'python-version: "3.13"' in advisory
    assert "torch==" not in advisory
    assert "python -m pytest tests -q" in advisory


def test_required_quality_job_runs_all_checks() -> None:
    quality = _job(_workflow(), "quality")

    for command in [
        "git ls-files -ci --exclude-standard",
        "ruff format --check .",
        "ruff check .",
        "mypy observable_library",
    ]:
        assert command in quality


def test_release_job_verifies_built_wheel_outside_checkout() -> None:
    workflow = _workflow()
    release = _job(workflow, "release")

    assert "needs: [quality, current-tests, minimum-tests]" in release
    assert "advisory-tests" not in release
    assert "python -m build" in release
    assert "dist/*.tar.gz" in release
    assert "dist/*.whl" in release
    assert "python -m venv" in release
    assert "python -m twine check --strict dist/*" in release
    assert "Check: inspect wheel and sdist contents" in release
    assert re.search(
        r'cp -R [^\n]*\bscripts\b[^\n]* "\$RUN_DIR/"',
        release,
    )
    assert "release-manifest.json" not in release
    assert re.search(r'cp -R [^\n]*\bpytest\.ini\b[^\n]* "\$RUN_DIR/"', release)
    assert re.search(r"cp -R [^\n]*\bLICENSE\b[\s\S]*\bMANIFEST\.in\b", release)
    assert 'cp observable_library/py.typed "$RUN_DIR/observable_library/"' in release
    assert 'cd "$RUN_DIR"' in release
    assert "observable_library.__file__" in release
    assert "for example in examples/*.py" in release
    assert "python -m pytest tests -q" in release
    assert "python -m pytest tests/performance -q -s" in release
    assert "python -m pytest tests/integration -q" in release
    assert (
        "python -m mypy --python-version 3.12 tests/fixtures/downstream_typing.py"
        in release
    )
    assert "-m pip uninstall -y observable-library" in release
    assert '-m pip install --no-deps "$SDIST"' in release
    assert "Check: sdist smoke import" in release
    assert "Check: sdist pip check" in release
    assert "pip install -e" not in release

    for forbidden in ["download=True", "MNIST", "datasets"]:
        assert forbidden not in workflow


def test_release_run_payload_is_valid_bash() -> None:
    release = _job(_workflow(), "release")
    script = _step_run(release, "Build and verify distributions")

    result = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_twine_is_installed_only_for_the_release_job() -> None:
    workflow = _workflow()

    assert 'python -m pip install "twine>=6"' in _job(workflow, "release")
    for name in ["quality", "current-tests", "minimum-tests", "advisory-tests"]:
        assert "twine" not in _job(workflow, name)


def test_ci_records_dependency_check_and_job_timings() -> None:
    workflow = _workflow()

    for name in [
        "quality",
        "current-tests",
        "minimum-tests",
        "advisory-tests",
        "release",
    ]:
        job = _job(workflow, name)
        assert "JOB_START" in job
        assert "Dependency install" in job
        assert "Check:" in job
        assert "GITHUB_STEP_SUMMARY" in job
        assert "Job total" in job or "Release end-to-end" in job

    assert 'echo "$line"' in workflow
    assert 'echo "- $line" >> "$GITHUB_STEP_SUMMARY"' in workflow


def test_ci_uses_current_actions_and_job_timeouts() -> None:
    workflow = _workflow()

    assert set(re.findall(r"actions/checkout@v\d+", workflow)) == {
        "actions/checkout@v7"
    }
    assert set(re.findall(r"actions/setup-python@v\d+", workflow)) == {
        "actions/setup-python@v6"
    }
    for name in ["quality", "current-tests", "minimum-tests", "advisory-tests"]:
        assert re.search(r"(?m)^    timeout-minutes: 5$", _job(workflow, name))
    assert re.search(r"(?m)^    timeout-minutes: 4$", _job(workflow, "release"))
    assert "continue-on-error: true" in _job(workflow, "advisory-tests")


def test_every_commit_jobs_enforce_one_combined_check_budget() -> None:
    workflow = _workflow()
    for name, count in {
        "quality": 4,
        "current-tests": 2,
        "minimum-tests": 2,
        "advisory-tests": 2,
    }.items():
        job = _job(workflow, name)
        calls = re.findall(r'(?m)^          timed "Check:[^\n]+$', job)
        assert job.count("timed() {") == 1
        assert job.index('timed "Dependency install"') < job.index("check_start=")
        assert len(calls) == count
        assert all(call.endswith(" || check_status=$?") for call in calls)
        assert "check_total=$(( $(date +%s) - check_start ))" in job
        assert 'line="Check total: ${check_total}s (limit 120s)"' in job
        assert "(( check_total <= 120 )) || check_status=1" in job
        assert job.count('exit "$check_status"') == 1


def test_release_enforces_only_the_combined_extended_check_budget() -> None:
    release = _job(_workflow(), "release")
    default = 'timed "Check: default pytest" python -m pytest tests -q'
    extended_checks = re.findall(
        r'(?m)^          timed "Check: (?:performance and trajectory|integration)"[^\n]+$',
        release,
    )

    assert release.count("timed() {") == 1
    assert release.index(default) < release.index("extended_start=")
    assert len(extended_checks) == 2
    assert all(call.endswith(" || extended_status=$?") for call in extended_checks)
    assert "extended_total=$(( $(date +%s) - extended_start ))" in release
    assert 'line="Extended check total: ${extended_total}s (limit 300s)"' in release
    assert "(( extended_total <= 300 )) || extended_status=1" in release
    assert release.count('exit "$extended_status"') == 1
    assert (
        "The 240s release timeout is stricter and may terminate before the 300s "
        "extended-check budget." in release
    )


def test_critical_path_monitor_is_guarded_and_compares_matching_boundaries() -> None:
    monitor = _job(_workflow(), "critical-path-monitor")
    permissions = re.search(
        r"(?ms)^    permissions:\n(?P<body>.*?)(?=^    steps:)", monitor
    )

    assert (
        "needs: [quality, current-tests, minimum-tests, advisory-tests, release]"
        in monitor
    )
    assert re.search(r"(?m)^    if: always\(\)$", monitor)
    assert re.search(r"(?m)^    timeout-minutes: 1$", monitor)
    assert "continue-on-error: true" in monitor
    assert permissions and permissions.group("body").strip() == "actions: read"
    assert "contents:" not in monitor
    assert "actions/checkout@" not in monitor

    for text in [
        "set +e",
        '[[ "${GITHUB_RUN_ATTEMPT:-1}" == "1" ]]',
        "rerun attempts are not independent workflow runs",
        "/actions/runs/$run_id/jobs",
        'select(.name != "Critical path monitor"',
        "| max // empty",
        'current_duration=$(get_duration "$GITHUB_RUN_ID")',
        'previous_duration=$(get_duration "$previous_id")',
        '-f "branch=$branch"',
        '-f "status=completed"',
        '-f "created=<${current_created_at}"',
        "max_by(.created_at)",
        "no previous completed run",
        "::warning title=Critical path over target",
        "::warning title=Critical path review required",
        "exit 0",
    ]:
        assert text in monitor
    assert monitor.count("get_duration() {") == 1
    for forbidden in [".updated_at", "::error", "contents:", "actions/checkout@"]:
        assert forbidden not in monitor
