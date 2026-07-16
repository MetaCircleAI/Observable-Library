from __future__ import annotations

import numpy as np
import torch

from observable_library.filter.identity import identity_observables
from observable_library.observable.observable import Observable
from observable_library.runtime.runtime import Runtime
from observable_library.tensor.source import FileSource
from observable_library.tensor.typed_tensor import TypedTensor


def test_runtime_uses_tensor_source_with_existing_observable_compute_path(
    tmp_path,
) -> None:
    tensor_path = tmp_path / "tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})
    observable = identity_observables(
        [{"source": "param.layer.weight", "selector": "all"}],
        reductions=["sum"],
    )[0]

    direct_runtime = Runtime(
        observables=[observable],
        tensors={"param.layer.weight": torch.tensor([[1.0, 2.0]])},
    )
    source_runtime = Runtime(
        observables=[observable],
        source=FileSource(tensor_path),
    )

    direct_values = direct_runtime.observe(step=0)
    source_values = source_runtime.observe(step=0)

    assert source_values.keys() == direct_values.keys()
    assert next(iter(source_values.values())) == next(iter(direct_values.values()))


def test_runtime_caches_source_and_passes_typed_tensor_metadata() -> None:
    class CountingSource:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, source_id: str, step: int) -> TypedTensor:
            self.calls += 1
            return TypedTensor(
                value=torch.tensor([2.0]),
                axes=("feature",),
                stage="offline",
                provenance={"checkpoint": "model.pt"},
            )

    spec_one = identity_observables(
        [{"source": "param.weight", "selector": "all"}], reductions=["sum"]
    )[0].spec
    spec_two = identity_observables(
        [{"source": "param.weight", "selector": "all"}], reductions=["mean"]
    )[0].spec
    observed_contexts = []
    records = []
    observables = [
        Observable(
            spec_one,
            lambda tensors, context: (
                observed_contexts.append(context["typed_tensor"])
                or tensors[spec_one.source].sum()
            ),
        ),
        Observable(spec_two, lambda tensors, context: tensors[spec_two.source].mean()),
    ]
    source = CountingSource()
    runtime = Runtime(
        observables,
        source=source,
        sink=lambda observable_id, step, value, meta: records.append(meta),
    )

    runtime.observe(step=3)

    assert source.calls == 1
    assert observed_contexts == [
        TypedTensor(
            value=torch.tensor([2.0]),
            axes=("feature",),
            stage="offline",
            provenance={"checkpoint": "model.pt"},
        )
    ]
    assert records[0] == {
        "source": "param.weight",
        "reduction": "sum",
        "axes": ("feature",),
        "stage": "offline",
        "provenance": {"checkpoint": "model.pt"},
    }
