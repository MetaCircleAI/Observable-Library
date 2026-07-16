import torch

import observable_library as ol
from observable_library.filter.identity import identity_observables
from observable_library.generator.api import generate
from observable_library.observable.registry import get_reduction
from observable_library.runtime.runtime import Runtime


def test_generated_reductions_match_direct_tensor_calculations():
    tensor = torch.tensor([[3.0, 4.0]])
    observables = identity_observables(
        [{"source": "param.weight", "selector": "all"}],
        reductions=["sum", "mean", "l2_norm", "max", "numel"],
    )

    values = Runtime(
        observables=observables,
        tensors={"param.weight": tensor},
    ).observe(step=0)

    by_reduction = {
        observable.spec.reduction: values[observable.spec.id]
        for observable in observables
    }

    assert torch.equal(by_reduction["sum"], tensor.sum())
    assert torch.equal(by_reduction["mean"], tensor.mean())
    assert torch.equal(by_reduction["l2_norm"], torch.linalg.vector_norm(tensor))
    assert torch.equal(by_reduction["max"], tensor.max())
    assert by_reduction["numel"] == tensor.numel()


def test_default_generated_reductions_match_direct_tensor_calculations():
    tensor = torch.tensor([[-2.0, 0.0], [3.0, 4.0]])
    observables = identity_observables(
        [{"source": "param.weight", "selector": "all"}],
        reductions=ol.generator.api.DEFAULT_REDUCTIONS,
    )

    values = Runtime(
        observables=observables,
        tensors={"param.weight": tensor},
    ).observe(step=0)
    by_reduction = {
        observable.spec.reduction: values[observable.spec.id]
        for observable in observables
    }

    assert torch.equal(by_reduction["sum"], tensor.sum())
    assert torch.equal(by_reduction["mean"], tensor.mean())
    assert torch.equal(by_reduction["l1_norm"], tensor.abs().sum())
    assert torch.equal(by_reduction["l2_norm"], tensor.norm())
    assert torch.equal(by_reduction["max"], tensor.max())
    assert torch.equal(by_reduction["min"], tensor.min())
    assert torch.equal(by_reduction["std"], tensor.std(correction=0))
    assert torch.equal(by_reduction["variance"], tensor.var(correction=0))
    assert torch.equal(by_reduction["abs_mean"], tensor.abs().mean())
    assert torch.equal(by_reduction["nonzero_count"], (tensor != 0).sum())
    assert torch.equal(by_reduction["positive_fraction"], (tensor > 0).float().mean())
    assert torch.equal(by_reduction["negative_fraction"], (tensor < 0).float().mean())
    assert by_reduction["numel"] == tensor.numel()


def test_generated_singleton_bias_uses_population_std_and_variance():
    model = torch.nn.Linear(1, 1)
    observables = generate(model, reductions=("std", "variance"))
    tensors = {
        f"param.{name}": parameter for name, parameter in model.named_parameters()
    }

    values = Runtime(observables=observables, tensors=tensors).observe(step=0)
    bias_values = {
        observable.spec.reduction: values[observable.spec.id]
        for observable in observables
        if observable.spec.source == "param.bias"
    }

    assert torch.equal(bias_values["std"], torch.tensor(0.0))
    assert torch.equal(bias_values["variance"], torch.tensor(0.0))


def test_std_and_variance_reductions_support_numpy_arrays():
    tensor = torch.tensor([2.0, 4.0])

    assert get_reduction("std")(tensor.numpy()) == 1.0
    assert get_reduction("variance")(tensor.numpy()) == 1.0
