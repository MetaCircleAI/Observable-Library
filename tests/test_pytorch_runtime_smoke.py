import torch

from observable_library.filter.identity import identity_observables
from observable_library.generator.introspector import introspect_parameters
from observable_library.runtime.runtime import Runtime


def test_generated_parameter_observable_computes_sum_on_real_tensor():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))

    metadata = introspect_parameters(model)
    observables = identity_observables(metadata, reductions=["sum"])
    tensors = {
        item["source"]: parameter
        for item, parameter in zip(metadata, model.parameters())
    }

    values = Runtime(observables=observables, tensors=tensors).observe(step=0)

    assert list(values.values()) == [torch.tensor(5.0)]
