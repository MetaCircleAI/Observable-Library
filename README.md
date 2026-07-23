# Observable Library

Headless Python package for generating and computing training observables.

The documentation site includes the current
[API reference](https://metacircleai.github.io/Observable-Library/api.html)
and a [practical usage guide](https://metacircleai.github.io/Observable-Library/usage.html).

Core boundary:

```text
model -> generate observables -> compute values -> return values
```

No CLI, web UI, notebook product surface, or agent skill is part of the core
package.

## Install

Python 3.10, 3.11, and 3.12 are supported. Python 3.13 is advisory until its
CI lane is promoted to required. Runtime dependencies are `numpy>=1.24` and
`torch>=2.4.1`.

Install the published package:

```bash
python -m pip install observable-library
```

Install from a checkout:

```bash
python -m pip install .
```

For development, install the quality and test tools too:

```bash
python -m pip install -e ".[dev]"
```

Licensed under Apache-2.0. Attribution: Jinxin.

## Quickstart

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
print(f"{observable_id}: {stored_value}")
optimizer.step()
source.detach()
```

`generate()` currently creates observables for every `model.named_parameters()`
entry; it does not generate activation or gradient observables. The Quickstart
attaches hooks to show the complete online lifecycle, but parameter-only
generated observables read parameters directly and do not require
`source.attach()`.

See the shipped examples:

- [MLP](https://github.com/MetaCircleAI/Observable-Library/blob/main/examples/mlp_online.py)
- [CNN](https://github.com/MetaCircleAI/Observable-Library/blob/main/examples/cnn_online.py)
- [Small transformer](https://github.com/MetaCircleAI/Observable-Library/blob/main/examples/small_transformer_online.py)
- [Advanced gradient](https://github.com/MetaCircleAI/Observable-Library/blob/main/examples/custom_gradient_online.py)
- [Offline file](https://github.com/MetaCircleAI/Observable-Library/blob/main/examples/offline_file.py)

Hand-written `Observable` construction is an advanced API. Generated and custom
observables can share one `Pack`. This example observes generated parameter
norms together with one causal gradient norm, after backward but before the
optimizer updates the model:

```python
import torch
import observable_library as ol

torch.manual_seed(3)
model = torch.nn.Sequential(torch.nn.Linear(2, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
targets = torch.tensor([[0.5], [-0.25]])

source = ol.HookSource(model)
source.attach()
parameter_observables = ol.generate(model, reductions=["l2_norm"])
gradient_observable = ol.Observable(
    spec=ol.ObservableSpec(source="grad.0.weight", selector="all", reduction="l2_norm"),
    compute=lambda tensors, _context: tensors["grad.0.weight"].norm(),
)
observables = [*parameter_observables, gradient_observable]
runtime = ol.Runtime(observables=ol.Pack(observables), source=source)

optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(inputs), targets)
loss.backward()
values = runtime.observe(step=0)
assert values[gradient_observable.spec.id] > 0
assert all(item.spec.id in values for item in parameter_observables)
optimizer.step()
source.detach()
```

## Practical Behavior

- `transforms=["a", "b"]` executes exactly `b(a(tensor))`, then the selected
  reduction. It does not generate transform subsets or permutations. Transform
  names are validated when observables are generated; tensor shape and dtype
  compatibility are checked only when they run.
- `generate()` has no public source allowlist. Filter the returned list
  to reduce Runtime work, or add a hand-written `Observable` for an activation,
  gradient, loss, or custom source.
- `Filter` is an extension foundation. Users can subclass it and compose
  filters with `&` and `|`, but these filters act on an existing observable list.
  Built-in generation-stage template filters are not implemented.
- `Runtime.observe()` returns values keyed by `observable.spec.id`, a stable
  16-character identifier derived from the full spec. `query()` supports exact
  id and step readback only; there is no lookup by display name, source, or
  reduction.
- The current public contract supports only `selector="all"`. See the
  [practical usage guide](https://metacircleai.github.io/Observable-Library/usage.html)
  for hook lifetime, source freshness, custom observable, transform, filter,
  and identity details.

## Offline File Source

```python
import numpy as np
import torch
import observable_library as ol

model = torch.nn.Linear(2, 1)
observables = ol.generate(model, reductions=["sum"])
payload = {
    f"param.{name}": parameter.detach().numpy()
    for name, parameter in model.named_parameters()
}
np.savez("tensors.npz", **payload)

source = ol.FileSource("tensors.npz")
runtime = ol.Runtime(observables=observables, source=source)
values = runtime.observe(step=0)
```

`ValueSink` is the storage contract. Built-in `LocalStorage` is only a
convenience sink: SQLite metadata plus NumPy NPZ payloads. It supports exact
observable id and step readback only. It does not provide a general query API.
Future Parquet support belongs in an optional backend or custom `ValueSink`; it
is not built into the current package.

## Current Support

The current package supports parameter observable generation, online `HookSource`,
and offline `CheckpointSource` and `FileSource` through the shared `Runtime`.
It also supports optional `ValueSink` and `LocalStorage` with exact id/step
readback and basic budget/frequency scheduling.

Planned capabilities include generation-stage template filters, equivalence and
cost calibration, multi-run comparison, and research workflows. The current
package has no CLI, UI, or general query surface.

## Validation

The examples use small in-memory inputs and do not download datasets.

```bash
ruff format --check .
ruff check .
mypy observable_library
python -m pytest tests -q
python -m pytest tests/performance -q -s
python -m pytest tests/integration -q
python -m build
```
