from __future__ import annotations

import torch

import observable_library as ol


torch.manual_seed(1)
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 2, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(18, 1),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
inputs = torch.linspace(-1.0, 1.0, steps=50).reshape(2, 1, 5, 5)
targets = torch.tensor([[0.25], [-0.5]])

source = ol.HookSource(model)
source.attach()
observables = ol.generate(model, reductions=["l2_norm"])
OBSERVABLE_IDS = [observable.spec.id for observable in observables]
runtime = ol.Runtime(observables=ol.Pack(observables), source=source)

INITIAL_PARAMETERS = [parameter.detach().clone() for parameter in model.parameters()]
LOSSES: list[float] = []
GRADIENT_NORMS: list[float] = []
RESULTS: list[dict[str, object]] = []
for step in range(2):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    GRADIENT_NORMS.append(
        sum(
            float(parameter.grad.detach().norm())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    LOSSES.append(float(loss.detach()))
    RESULTS.append(runtime.observe(step=step))
    optimizer.step()

source.detach()
