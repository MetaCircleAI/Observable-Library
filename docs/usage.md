# Practical Usage Guide

This guide describes the behavior implemented by the current Observable Library
package. Future capabilities are called out explicitly.

## What `generate()` Produces

`generate(model, reductions=..., transforms=...)` scans every entry returned by
`model.named_parameters()`. It produces one observable for every parameter and
requested reduction. The same ordered transform chain is attached to each one.

`generate()` does not expose a `sources=` argument and does not generate `activation.*`,
`grad.*`, or `loss` observables. Filtering the returned list reduces Runtime
work, although parameter metadata has already been enumerated:

```python
generated = ol.generate(model, reductions=["l2_norm"])
wanted = {"param.0.weight", "param.2.weight"}
selected = [item for item in generated if item.spec.source in wanted]
```

Generation-stage template filters for parameters, gradients, and activations
are not implemented. Their exact source-selection API is not part of the public
contract.

## Transform Order And Validation

Transforms run from left to right. For example,
`transforms=["center", "normalize"]` executes:

```text
normalize(center(tensor)) -> reduction
```

No subsets, reverse order, or permutations are generated. Every transform must
be registered before `generate()` is called, and duplicate names are rejected.
The registry validates names, not tensor contracts: incompatible rank, shape,
dtype, or device assumptions fail when the observable runs.

Generated observables capture the resolved transform and reduction callables.
Later registry changes therefore cannot silently change an existing
observable's computation.

## Online Source Lifetime

`HookSource.get("param.*", step)` reads the current model parameter directly.
Parameter-only generated observables do not need hooks, so `attach()` is
optional for that case.

`activation.*` and `grad.*` values are produced during forward and backward.
Call `source.attach()` before that work and `source.detach()` when finished.
Observe gradients after `loss.backward()` and before `optimizer.step()`.

Current `attach()` behavior is broad: it installs activation hooks on all
top-level children and gradient hooks on all trainable parameters. It stores the
latest captured tensor and does not track freshness per step. Run the matching
forward/backward before every `observe()` call; otherwise a previous value may
remain cached. Activation ids use `activation.<top-level-child-name>` and
gradient ids use `grad.<parameter-name>`.

Loss is not discovered automatically. To use a `source="loss"` observable,
call `source.record_loss(loss, step)` first.

Within one `observe()` call, Runtime caches each requested source, so several
reductions over the same tensor perform one source lookup.

## Mix Generated And Custom Observables

`generate()` returns `list[Observable]`. Add advanced custom observables before
constructing a `Pack`:

```python
generated = ol.generate(model, reductions=["l2_norm"])
gradient_spec = ol.ObservableSpec(
    source="grad.0.weight",
    selector="all",
    reduction="l2_norm",
    budget_hint={"compute_ms": 0.01},
)
gradient = ol.Observable(
    spec=gradient_spec,
    compute=lambda tensors, _context: tensors[gradient_spec.source].norm(),
)
runtime = ol.Runtime(ol.Pack([*generated, gradient]), source=source)
```

Runtime requires every observable id to be unique. A custom observable whose
spec is identical to a generated one is rejected. When a `Budget` is active,
custom observables should provide a realistic `budget_hint`; a missing estimate
counts as zero.

## User-Defined Filters

The package provides the `Filter` base class and `&` / `|` composition, but no built-in
`BySource` or `ByReduction` filters. A user-defined filter acts after generation:

```python
class ByReduction(ol.Filter):
    def __init__(self, name: str) -> None:
        self.name = name

    def apply(self, observables):
        return [
            item for item in observables if item.spec.reduction == self.name
        ]

keep = ByReduction("l2_norm") | ByReduction("mean")
selected = keep(observables)
```

This prevents unselected observables from running, but does not prevent their
initial generation. True generation-stage template filters are planned.

## Identity, Results, And Query

There is no separate user-facing name field. `observable.spec.id` is a
deterministic 16-character hash derived from the complete spec, including its
source, selector, ordered transforms, reduction, temporal settings, frequency,
and budget hint. Changing any identity field can change the id.

`Runtime.observe()` returns `dict[observable_id, value]`. `LocalStorage` stores
metadata in SQLite and array payloads in NumPy NPZ files; it and `query()` use
the same exact id and step:

```python
observable = observables[0]
values = runtime.observe(step=7)
value = values[observable.spec.id]
stored = ol.query(storage, observable.spec.id, step=7)
```

For logs or user interfaces, derive a display label without treating it as the
storage key:

```python
spec = observable.spec
chain = " -> ".join((*spec.transforms, spec.reduction))
label = f"{spec.source} | {chain}"
```

The package supports exact id/step readback only. Query by source, reduction, tag, or
display label belongs to later analysis tooling.
