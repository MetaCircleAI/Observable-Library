from __future__ import annotations

import torch
import pytest

import observable_library as ol
from observable_library.filter.identity import identity_observables
from observable_library.observable import registry


def test_registered_reduction_runs_through_generated_observable() -> None:
    @ol.register_reduction("test_range")
    def tensor_range(tensor):
        return tensor.max() - tensor.min()

    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 4.0]]))

    observable = ol.generate(model, reductions=["test_range"])[0]
    runtime = ol.Runtime(
        [observable],
        tensors={"param.0.weight": model[0].weight},
    )

    assert runtime.observe(step=0)[observable.spec.id].item() == 3.0


def test_registered_transform_runs_before_generated_reduction() -> None:
    @ol.register_transform("test_double")
    def double(tensor):
        return tensor * 2

    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(3.0)

    observable = ol.generate(
        model,
        transforms=["test_double"],
        reductions=["sum"],
    )[0]
    runtime = ol.Runtime(
        [observable],
        tensors={"param.0.weight": model[0].weight},
    )

    assert runtime.observe(step=0)[observable.spec.id].item() == 6


def test_registry_rejects_duplicate_transform_names() -> None:
    @ol.register_transform("test_duplicate_transform")
    def first(tensor):
        return tensor

    with pytest.raises(ValueError, match="test_duplicate_transform"):

        @ol.register_transform("test_duplicate_transform")
        def second(tensor):
            return tensor


def test_identity_observable_captures_registered_operators() -> None:
    @ol.register_transform("test_captured_transform")
    def add_one(tensor):
        return tensor + 1

    @ol.register_reduction("test_captured_reduction")
    def total(tensor):
        return tensor.sum()

    observable = identity_observables(
        [{"source": "param.weight", "selector": "all"}],
        reductions=["test_captured_reduction"],
        transforms=["test_captured_transform"],
    )[0]
    registry._transforms["test_captured_transform"] = lambda tensor: tensor
    registry._reductions["test_captured_reduction"] = lambda tensor: tensor.max()

    assert observable.compute({"param.weight": torch.tensor([2.0, 3.0])}, {}) == 7.0


def test_identity_observable_rejects_unknown_operator_at_construction() -> None:
    with pytest.raises(KeyError, match="missing_transform"):
        identity_observables(
            [{"source": "param.weight", "selector": "all"}],
            reductions=["sum"],
            transforms=["missing_transform"],
        )

    with pytest.raises(NotImplementedError, match="missing_reduction"):
        identity_observables(
            [{"source": "param.weight", "selector": "all"}],
            reductions=["missing_reduction"],
        )
