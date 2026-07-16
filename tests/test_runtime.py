from typing import cast

import pytest

from observable_library.observable.observable import Observable
from observable_library.observable.spec import ObservableSpec
from observable_library.runtime.runtime import Runtime
from observable_library.tensor.source import TensorSource


def test_runtime_observe_returns_values_keyed_by_observable_id():
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=[],
        reduction="sum",
    )
    observable = Observable(
        spec=spec,
        compute=lambda tensors, context: tensors[spec.source] + context["offset"],
        tags={"identity"},
    )
    runtime = Runtime(
        observables=[observable],
        tensors={"param.layers.0.weight": 5},
    )

    values = runtime.observe(step=2, offset=6)

    assert values == {spec.id: 11}


def test_runtime_observe_sends_values_to_optional_sink():
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=[],
        reduction="sum",
    )
    observable = Observable(
        spec=spec,
        compute=lambda tensors, context: tensors[spec.source],
        tags={"identity"},
    )
    records = []

    runtime = Runtime(
        observables=[observable],
        tensors={"param.layers.0.weight": 5},
        sink=lambda observable_id, step, value, meta: records.append(
            (observable_id, step, value, meta)
        ),
    )

    values = runtime.observe(step=2)

    assert values == {spec.id: 5}
    assert records == [
        (spec.id, 2, 5, {"source": spec.source, "reduction": spec.reduction})
    ]


def test_runtime_materializes_one_shot_observable_iterable() -> None:
    spec = ObservableSpec(source="param.weight", selector="all", reduction="sum")
    observable = Observable(spec, lambda tensors, context: tensors[spec.source])
    runtime = Runtime((item for item in [observable]), tensors={"param.weight": 1})

    assert runtime.observe(step=0) == {spec.id: 1}
    assert runtime.observe(step=1) == {spec.id: 1}


def test_runtime_rejects_duplicate_observable_ids() -> None:
    spec = ObservableSpec(source="param.weight", selector="all", reduction="sum")
    observable = Observable(spec, lambda tensors, context: tensors[spec.source])

    with pytest.raises(ValueError, match="unique observable ids"):
        Runtime([observable, observable], tensors={"param.weight": 1})


def test_runtime_rejects_tensors_and_source_together() -> None:
    spec = ObservableSpec(source="param.weight", selector="all", reduction="sum")
    observable = Observable(spec, lambda tensors, context: tensors[spec.source])

    with pytest.raises(ValueError, match="not both"):
        Runtime(
            [observable],
            tensors={"param.weight": 1},
            source=cast(TensorSource, object()),
        )


def test_runtime_rejects_non_all_selector() -> None:
    spec = ObservableSpec(source="param.weight", selector="row[0]", reduction="sum")
    observable = Observable(spec, lambda tensors, context: tensors[spec.source])
    runtime = Runtime([observable], tensors={"param.weight": 1})

    with pytest.raises(NotImplementedError, match="row\\[0\\]"):
        runtime.observe(step=0)
