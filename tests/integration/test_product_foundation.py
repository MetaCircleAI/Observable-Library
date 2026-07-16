from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import observable_library as ol


pytestmark = pytest.mark.integration


def test_storage_dependencies_exclude_pyarrow_and_parquet() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert dependencies == ["numpy>=1.24", "torch>=2.4.1"]
    assert not any("pyarrow" in item or "parquet" in item for item in dependencies)


def test_runtime_source_storage_and_budget_integration(tmp_path) -> None:
    tensor_path = tmp_path / "tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})
    observables = ol.generate(_TinyModel(), reductions=["sum"])
    observable = observables[0]
    observable = ol.Observable(
        spec=replace(observable.spec, budget_hint={"compute_ms": 0.5}),
        compute=observable.compute,
        tags=observable.tags,
    )

    storage = ol.LocalStorage(tmp_path / "run")
    pack = ol.Pack([observable])
    analyzer = ol.OfflineAnalyzer(
        observables=pack,
        source=ol.FileSource(tensor_path),
        sink=storage,
        budget=ol.Budget(max_compute_ms=1.0),
    )

    values = analyzer.analyze(step=0)
    stored = ol.query(storage, observable.spec.id, 0)

    assert float(next(iter(values.values()))) == 3.0
    assert stored == 3.0


def test_registry_and_filter_composition_integration() -> None:
    @ol.register_transform("integration_double")
    def double(tensor):
        return tensor * 2

    @ol.register_reduction("integration_sum")
    def total(tensor):
        return tensor.sum()

    observable = ol.generate(
        _TinyModel(),
        transforms=["integration_double"],
        reductions=["integration_sum"],
    )[0]

    class KeepAll(ol.Filter):
        def apply(self, observables):
            return list(observables)

    filtered = (KeepAll() & KeepAll()).apply([observable])
    values = ol.Runtime(
        ol.Pack(filtered),
        tensors={"param.layer.weight": np.array([[1.0, 2.0]])},
    ).observe(step=0)

    assert next(iter(values.values())) == 6.0


class _TinyModel:
    def named_parameters(self):
        return [("layer.weight", np.array([[1.0, 2.0]]))]
