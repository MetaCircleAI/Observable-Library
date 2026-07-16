from __future__ import annotations

import pytest
import torch

import observable_library as ol
from observable_library.observable.observable import Observable
from observable_library.observable.spec import ObservableSpec
from observable_library.tensor.typed_tensor import TypedTensor


class CountingSource:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, source_id: str, step: int) -> TypedTensor:
        self.calls += 1
        return TypedTensor(
            value=torch.ones(2, 2),
            axes=("out_feature", "in_feature"),
        )


def test_runtime_skips_observable_when_frequency_does_not_divide_step() -> None:
    calls = 0
    spec = ObservableSpec(
        source="param.weight",
        selector="all",
        reduction="sum",
        frequency=2,
    )

    def compute(tensors, context):
        nonlocal calls
        calls += 1
        return tensors[spec.source]

    runtime = ol.Runtime(
        observables=[Observable(spec=spec, compute=compute)],
        tensors={"param.weight": 3},
    )

    assert runtime.observe(step=1) == {}
    assert calls == 0
    assert runtime.observe(step=2) == {spec.id: 3}
    assert calls == 1


def test_runtime_budget_prevents_compute_when_hint_exceeds_remaining_budget() -> None:
    calls = 0
    spec = ObservableSpec(
        source="param.weight",
        selector="all",
        reduction="sum",
        budget_hint={"compute_ms": 2.0},
    )

    def compute(tensors, context):
        nonlocal calls
        calls += 1
        return tensors[spec.source]

    runtime = ol.Runtime(
        observables=[Observable(spec=spec, compute=compute)],
        tensors={"param.weight": 3},
        budget=ol.Budget(max_compute_ms=1.0),
    )

    assert runtime.observe(step=0) == {}
    assert calls == 0


def test_zero_budget_skips_generated_observables_before_source_access() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    source = CountingSource()
    runtime = ol.Runtime(
        observables=ol.generate(model),
        source=source,
        budget=ol.Budget(max_compute_ms=0),
    )

    assert runtime.observe(step=0) == {}
    assert source.calls == 0


@pytest.mark.parametrize("frequency", [0, -1])
def test_runtime_rejects_non_positive_frequency_before_compute(frequency: int) -> None:
    calls = 0
    spec = ObservableSpec(
        source="param.weight",
        selector="all",
        reduction="sum",
        frequency=frequency,
    )

    def compute(tensors, context):
        nonlocal calls
        calls += 1
        return tensors[spec.source]

    runtime = ol.Runtime(
        observables=[Observable(spec=spec, compute=compute)],
        tensors={"param.weight": 3},
    )
    with pytest.raises(
        ValueError, match=r"ObservableSpec\.frequency must be greater than zero"
    ):
        runtime.observe(step=0)
    assert calls == 0


@pytest.mark.parametrize(
    "compute_ms",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_runtime_rejects_invalid_compute_hint_before_scheduling(
    compute_ms: float,
) -> None:
    calls = 0
    spec = ObservableSpec(
        source="param.weight",
        selector="all",
        reduction="sum",
        frequency=2,
        budget_hint={"compute_ms": compute_ms},
    )

    def compute(tensors, context):
        nonlocal calls
        calls += 1
        return tensors[spec.source]

    runtime = ol.Runtime(
        observables=[Observable(spec=spec, compute=compute)],
        tensors={"param.weight": 3},
    )
    with pytest.raises(
        ValueError,
        match=r"ObservableSpec\.budget_hint\['compute_ms'\] must be finite and non-negative",
    ):
        runtime.observe(step=1)
    assert calls == 0


@pytest.mark.parametrize(
    "max_compute_ms",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_runtime_rejects_invalid_budget_before_scheduling(
    max_compute_ms: float,
) -> None:
    calls = 0
    spec = ObservableSpec(
        source="param.weight",
        selector="all",
        reduction="sum",
        frequency=2,
    )

    def compute(tensors, context):
        nonlocal calls
        calls += 1
        return tensors[spec.source]

    runtime = ol.Runtime(
        observables=[Observable(spec=spec, compute=compute)],
        tensors={"param.weight": 3},
        budget=ol.Budget(max_compute_ms=max_compute_ms),
    )

    with pytest.raises(
        ValueError,
        match=r"Budget\.max_compute_ms must be finite and non-negative",
    ):
        runtime.observe(step=1)
    assert calls == 0
