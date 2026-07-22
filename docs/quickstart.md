(quickstart)=
# Quickstart

This example generates parameter-norm observables, runs one training step, computes values after backward, stores them locally, and reads one value back.

```python
import torch
import observable_library as ol

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(2, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
observables = ol.generate(model, reductions=["l2_norm"])

source = ol.HookSource(model)
source.attach()
storage = ol.LocalStorage("./run")
runtime = ol.Runtime(
    observables=ol.Pack(observables),
    source=source,
    sink=storage,
    budget=ol.Budget(max_compute_ms=10.0),
)

inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
targets = torch.tensor([[0.5], [-0.25]])
optimizer.zero_grad()
predictions = model(inputs)
loss = torch.nn.functional.mse_loss(predictions, targets)
loss.backward()

values = runtime.observe(step=0)
assert values
observable_id, value = next(iter(values.items()))
stored_value = ol.query(storage, observable_id, step=0)
assert stored_value == value
print(f"{observable_id}: {stored_value}")

optimizer.step()
source.detach()
```

## What happened

1. `ol.generate()` inspected `model.named_parameters()` and produced one `l2_norm` observable for each parameter.
2. `HookSource` provided the current parameter tensors. Parameter reads are lazy, so this generated parameter-only example does not technically require hooks; `attach()` is shown because the same lifecycle is required for activation and gradient sources.
3. `loss.backward()` ran before `Runtime.observe()`. This ordering is required when a custom observable reads gradients.
4. `Runtime.observe()` returned values keyed by each observable's stable 16-character spec id.
5. `LocalStorage` wrote SQLite metadata and NumPy NPZ payloads. `query()` read one value back by exact id and step.

## Keep the lifecycle explicit

For activation or gradient sources, attach before the matching forward/backward work, observe before the optimizer mutates parameters, and always detach when finished. `HookSource` retains the latest captured tensor and does not enforce per-step freshness.

Continue with {ref}`concepts` or the task-oriented {ref}`how-to-guides`.
