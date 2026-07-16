from __future__ import annotations

import torch

import observable_library as ol


torch.manual_seed(3)
model = torch.nn.Sequential(torch.nn.Linear(2, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
targets = torch.tensor([[0.5], [-0.25]])

source = ol.HookSource(model)
source.attach()
gradient_observable = ol.Observable(
    spec=ol.ObservableSpec(source="grad.0.weight", selector="all", reduction="l2_norm"),
    compute=lambda tensors, _context: tensors["grad.0.weight"].norm(),
)
runtime = ol.Runtime(observables=ol.Pack([gradient_observable]), source=source)

OBSERVED_VALUES: list[float] = []
EXPECTED_VALUES: list[float] = []
for step in range(2):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    EXPECTED_VALUES.append(float(model[0].weight.grad.detach().norm()))
    values = runtime.observe(step=step)
    OBSERVED_VALUES.append(float(values[gradient_observable.spec.id]))
    optimizer.step()

source.detach()
