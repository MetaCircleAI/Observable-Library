from __future__ import annotations

import numpy as np

import observable_library as ol


def test_offline_analyzer_delegates_to_runtime_compute_path(tmp_path) -> None:
    tensor_path = tmp_path / "tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})
    spec = ol.ObservableSpec(
        source="param.layer.weight",
        selector="all",
        reduction="sum",
    )
    calls = []
    observable = ol.Observable(
        spec=spec,
        compute=lambda tensors, context: (
            calls.append(context["step"]) or tensors[spec.source].sum()
        ),
    )
    pack = ol.Pack([observable])

    analyzer = ol.OfflineAnalyzer(pack, ol.FileSource(tensor_path))

    assert analyzer.analyze(step=7)[spec.id].item() == 3.0
    assert calls == [7]


def test_offline_analyzer_materializes_one_shot_observable_iterable(tmp_path) -> None:
    tensor_path = tmp_path / "tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})
    spec = ol.ObservableSpec(
        source="param.layer.weight",
        selector="all",
        reduction="sum",
    )
    observable = ol.Observable(
        spec=spec,
        compute=lambda tensors, context: tensors[spec.source].sum(),
    )
    analyzer = ol.OfflineAnalyzer(
        (item for item in [observable]),
        ol.FileSource(tensor_path),
    )

    assert analyzer.analyze(step=0)[spec.id].item() == 3.0
    assert analyzer.analyze(step=1)[spec.id].item() == 3.0
