import torch

import observable_library as ol


def test_generate_returns_identity_observables_from_public_api() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))

    observables = ol.generate(model, reductions=["l2_norm"])

    spec_keys = {
        (observable.spec.source, observable.spec.selector, observable.spec.reduction)
        for observable in observables
    }
    assert spec_keys == {
        ("param.0.weight", "all", "l2_norm"),
        ("param.0.bias", "all", "l2_norm"),
    }
    assert all("identity" in observable.tags for observable in observables)


def test_generated_observables_use_current_hook_source_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    observables = ol.generate(model)
    source = ol.HookSource(model)
    runtime = ol.Runtime(observables, source=source)
    sum_observable = next(
        observable for observable in observables if observable.spec.reduction == "sum"
    )

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))
    first = runtime.observe(step=0)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[4.0, 5.0]]))
    second = runtime.observe(step=1)

    assert torch.equal(first[sum_observable.spec.id], torch.tensor(5.0))
    assert torch.equal(second[sum_observable.spec.id], torch.tensor(9.0))


def test_generated_observables_have_positive_shape_aware_cost_hints() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(1_000, 1_000, bias=False),
    )

    observables = ol.generate(model, reductions=["sum"])
    hints = {
        observable.spec.source: observable.spec.budget_hint["compute_ms"]
        for observable in observables
    }

    assert all(compute_ms > 0 for compute_ms in hints.values())
    assert hints["param.1.weight"] > hints["param.0.weight"]
