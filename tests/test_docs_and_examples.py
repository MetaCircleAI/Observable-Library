from __future__ import annotations

import math
import re
import runpy
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ONLINE_EXAMPLES = (
    ROOT / "examples/mlp_online.py",
    ROOT / "examples/cnn_online.py",
    ROOT / "examples/small_transformer_online.py",
)
ADVANCED_ONLINE_EXAMPLE = ROOT / "examples/custom_gradient_online.py"
OFFLINE_EXAMPLE = ROOT / "examples/offline_file.py"
USAGE_GUIDE = ROOT / "docs/usage.md"
TRACKED_PUBLIC_DOCS = (README, ROOT / "docs/api.md", USAGE_GUIDE)


def _readme_python_blocks() -> list[str]:
    readme = README.read_text(encoding="utf-8")
    return re.findall(r"```python[^\n]*\n(.*?)\n```", readme, flags=re.DOTALL)


def _run_path(path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return runpy.run_path(str(path), run_name="__main__")


def test_every_readme_python_block_executes_in_an_isolated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = _readme_python_blocks()
    assert blocks

    for index, block in enumerate(blocks):
        cwd = tmp_path / f"readme-block-{index}"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        exec(
            compile(block, f"README.md:block-{index}", "exec"), {"__name__": "__main__"}
        )


def test_readme_quickstart_shows_the_complete_online_path() -> None:
    quickstart = _readme_python_blocks()[0]

    for text in [
        "ol.generate(",
        "ol.HookSource(model)",
        "source.attach()",
        "ol.Runtime(",
        "source=source",
        "loss.backward()",
        "optimizer.step()",
        "runtime.observe(",
        "source.detach()",
    ]:
        assert text in quickstart

    assert "ol.Observable(" not in quickstart
    assert "ol.ObservableSpec(" not in quickstart
    assert quickstart.index("loss.backward()") < quickstart.index("runtime.observe(")
    assert quickstart.index("runtime.observe(") < quickstart.index("optimizer.step()")


def test_readme_quickstart_prints_and_reads_back_a_stored_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    quickstart = _readme_python_blocks()[0]

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(quickstart, "README.md:quickstart", "exec"), namespace)

    output = capsys.readouterr().out
    assert "ol.query(storage, observable_id, step=0)" in quickstart
    assert namespace["observable_id"] in namespace["values"]
    assert namespace["stored_value"] == namespace["values"][namespace["observable_id"]]
    assert str(namespace["observable_id"]) in output
    assert str(namespace["stored_value"]) in output


def test_readme_links_to_canonical_docs_and_shipped_examples() -> None:
    readme = README.read_text(encoding="utf-8")

    docs_url = "https://metacircleai.github.io/Observable-Library/"
    for path in ["api.html", "usage.html"]:
        assert f"{docs_url}{path}" in readme

    source_url = "https://github.com/MetaCircleAI/Observable-Library/blob/main/"
    example_paths = [
        *(f"examples/{path.name}" for path in ONLINE_EXAMPLES),
        ADVANCED_ONLINE_EXAMPLE.relative_to(ROOT).as_posix(),
        OFFLINE_EXAMPLE.relative_to(ROOT).as_posix(),
    ]
    for path in example_paths:
        assert f"{source_url}{path}" in readme


def test_readme_states_the_exact_local_storage_boundary() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "SQLite metadata plus NumPy NPZ payloads" in readme
    assert re.search(r"exact\s+observable id and step readback only", readme)
    assert "general query API" in readme


def test_readme_states_the_overall_package_boundary() -> None:
    readme = README.read_text(encoding="utf-8")

    for text in [
        "parameter observable generation",
        "online `HookSource`",
        "offline `CheckpointSource` and `FileSource`",
        "shared `Runtime`",
        "optional `ValueSink` and `LocalStorage`",
        "exact id/step",
        "basic budget/frequency",
        "generation-stage template filters",
        "multi-run comparison",
        "research workflows",
        "no CLI, UI, or general query",
    ]:
        assert text in readme


def test_readme_contains_an_inline_advanced_custom_observable_example() -> None:
    readme = README.read_text(encoding="utf-8")
    advanced_example = next(
        block for block in _readme_python_blocks()[1:] if "ol.Observable(" in block
    )

    assert "advanced" in readme.lower()
    assert (
        "https://github.com/MetaCircleAI/Observable-Library/blob/main/"
        "examples/custom_gradient_online.py"
    ) in readme
    assert "ol.ObservableSpec(" in advanced_example
    assert "ol.HookSource(model)" in advanced_example
    assert "ol.generate(" in advanced_example
    assert "[*parameter_observables, gradient_observable]" in advanced_example
    assert advanced_example.index("loss.backward()") < advanced_example.index(
        "runtime.observe("
    )
    assert advanced_example.index("runtime.observe(") < advanced_example.index(
        "optimizer.step()"
    )


@pytest.mark.parametrize("example_path", ONLINE_EXAMPLES, ids=lambda path: path.name)
def test_online_examples_execute_a_real_training_loop(
    example_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_path(example_path, tmp_path / example_path.stem, monkeypatch)
    losses = result["LOSSES"]
    gradient_norms = result["GRADIENT_NORMS"]
    observations = result["RESULTS"]
    initial_parameters = result["INITIAL_PARAMETERS"]
    final_parameters = list(result["model"].parameters())

    assert len(losses) >= 2
    assert len(gradient_norms) == len(losses)
    assert len(observations) == len(losses)
    assert all(math.isfinite(loss) for loss in losses)
    assert all(gradient_norm > 0 for gradient_norm in gradient_norms)
    assert all(values for values in observations)
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, final_parameters, strict=True)
    )

    observable_ids = result["OBSERVABLE_IDS"]
    assert observable_ids
    assert all(set(observable_ids) <= values.keys() for values in observations)

    source = example_path.read_text(encoding="utf-8")
    for call in [
        "source.attach()",
        "loss.backward()",
        "optimizer.step()",
        "runtime.observe(",
        "source.detach()",
    ]:
        assert call in source
    assert "tensors=" not in source
    assert "ol.generate(" in source
    assert "ol.Observable(" not in source
    assert "ol.ObservableSpec(" not in source
    assert source.index("loss.backward()") < source.index("runtime.observe(")
    assert source.index("runtime.observe(") < source.index("optimizer.step()")


def test_advanced_custom_gradient_example_executes_and_matches_direct_gradient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_path(
        ADVANCED_ONLINE_EXAMPLE,
        tmp_path / ADVANCED_ONLINE_EXAMPLE.stem,
        monkeypatch,
    )

    assert result["OBSERVED_VALUES"]
    assert result["OBSERVED_VALUES"] == pytest.approx(result["EXPECTED_VALUES"])

    source = ADVANCED_ONLINE_EXAMPLE.read_text(encoding="utf-8")
    for text in [
        "ol.Observable(",
        "ol.ObservableSpec(",
        "ol.HookSource(model)",
        "loss.backward()",
        "runtime.observe(",
        "optimizer.step()",
    ]:
        assert text in source
    assert source.index("loss.backward()") < source.index("runtime.observe(")
    assert source.index("runtime.observe(") < source.index("optimizer.step()")


def test_offline_example_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_path(OFFLINE_EXAMPLE, tmp_path / OFFLINE_EXAMPLE.stem, monkeypatch)

    assert result["RESULTS"]


def test_api_reference_documents_complete_public_surface() -> None:
    api_reference = (ROOT / "docs/api.md").read_text(encoding="utf-8")

    for name in [
        "Observable",
        "ObservableSpec",
        "CheckpointSource",
        "FileSource",
        "Filter",
        "Budget",
        "LocalStorage",
        "OfflineAnalyzer",
        "Pack",
        "Runtime",
        "HookSource",
        "TensorSource",
        "TypedTensor",
        "ValueSink",
        "__version__",
        "delta",
        "ema",
        "generate",
        "get_reduction",
        "get_transform",
        "query",
        "register_reduction",
        "register_transform",
        "rolling_std",
        "slope",
    ]:
        assert re.search(rf"`{re.escape(name)}(?:`|\()", api_reference)

    for term in [
        "temporal",
        "frequency",
        "budget_hint",
        "compute_ms",
        "max_compute_ms",
        "selector='all'",
        "provenance",
        "heuristic",
    ]:
        assert f"`{term}`" in api_reference

    assert "not a calibrated cost-accuracy estimate" in api_reference


def test_api_reference_marks_api_boundaries() -> None:
    api_reference = (ROOT / "docs/api.md").read_text(encoding="utf-8").lower()

    assert "`generate` is the canonical" in api_reference
    assert re.search(
        r"manual `observable` construction\s+is an advanced", api_reference
    )


def test_usage_guide_documents_current_generation_and_runtime_behavior() -> None:
    usage = USAGE_GUIDE.read_text(encoding="utf-8")

    for text in [
        "model.named_parameters()",
        "does not expose a `sources=` argument",
        "does not generate `activation.*`",
        "Parameter-only generated observables do not need hooks",
        "latest captured tensor",
        "does not track freshness per step",
        "after `loss.backward()` and before `optimizer.step()`",
        "Runtime requires every observable id to be unique",
        "no built-in\n`BySource` or `ByReduction` filters",
        "deterministic 16-character hash",
        "exact id/step readback only",
    ]:
        assert text in usage

    assert '`transforms=["center", "normalize"]` executes' in usage
    assert "normalize(center(tensor))" in usage


def test_tracked_public_docs_do_not_use_obsolete_api_calls() -> None:
    current_docs = "\n".join(
        path.read_text(encoding="utf-8") for path in TRACKED_PUBLIC_DOCS
    )
    for stale_text in [
        "ol.filters",
        "runtime.attach(",
        "Runtime.attach(",
        "runtime.query(",
    ]:
        assert stale_text not in current_docs


def test_tracked_public_docs_use_sqlite_metadata_and_npz_arrays() -> None:
    for path in TRACKED_PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "SQLite" in text
        assert "NPZ" in text
        for line in text.splitlines():
            if "parquet" not in line.lower():
                continue
            assert any(
                marker in line.lower()
                for marker in (
                    "future",
                    "optional backend",
                    "custom",
                )
            )
