(api-reference)=
# API Reference

The names below are available from the top-level `observable_library` 0.1.0 package.

## Core Types

`ObservableSpec(source, selector, transforms=[], reduction="", temporal=None,
frequency=1, budget_hint={})` describes one observable. Its stable `id` includes
non-default temporal and scheduling fields.

`Observable(spec, compute, tags=set())` pairs a spec with a callable that accepts
the source tensor mapping and runtime context. Manual `Observable` construction
is an advanced API; `Pack(observables)` is the stable iterable container passed
to execution APIs.

`TypedTensor(value, axes, stage="", provenance={})` carries a tensor with axis,
stage, and provenance metadata. `__version__` reports the installed package
version.

## Generation And Registry

`generate` is the canonical API for normal instrumentation.
`generate(model, reductions=..., transforms=...)` inspects model parameters and
returns a list of observables. The default uses all 13 registered reductions:
`sum`, `mean`, `l1_norm`, `l2_norm`, `max`, `min`, `std`, `variance`,
`abs_mean`, `nonzero_count`, `positive_fraction`, `negative_fraction`, and
`numel`.

M2 scans every `model.named_parameters()` entry. It has no public source
allowlist and does not generate activation, gradient, or loss observables.
Filter the returned list for a smaller Runtime pack, or use the advanced manual
`Observable` API for another source. See the practical usage guide for examples.

Transforms execute in the supplied order before the reduction. Thus
`transforms=["a", "b"]` means `b(a(tensor))`; no subsets or permutations are
generated. Registered names are resolved during generation. Tensor rank, shape,
dtype, and device compatibility remain the transform author's responsibility
and are checked only by the actual operation at runtime.

`register_transform(name)` and `register_reduction(name)` are decorators for
callable extensions. `get_transform(name)` and `get_reduction(name)` retrieve
registered callables.

## Tensor Sources

`TensorSource` is the protocol `get(source_id, step) -> TypedTensor`.

`HookSource(model)` supplies online parameters, activations, gradients, and
recorded losses. Call `attach()` before the training work and `detach()` when
finished. `FileSource(path)` reads tensors by key from an NPZ file.
`CheckpointSource(path)` reads `param.*` sources from a Torch state dict.

Parameter reads are lazy and do not require `attach()`. Current hook attachment
is broad: all top-level child activations and all trainable parameter gradients
are captured. HookSource retains the latest captured value without per-step
freshness validation, so run the corresponding forward/backward before each
observation. Record a loss explicitly with `record_loss(loss, step)`.

M2 supports only `selector='all'`. When Runtime reads a `TypedTensor`, it passes
the complete object in compute context as `typed_tensor` and forwards its `axes`,
`stage`, and `provenance` to `ValueSink` metadata.

## Execution And Budget

`Runtime(observables, tensors=None, source=None, budget=None, sink=None)` computes
each eligible observable when `observe(step, **context)` is called. Pass either a
tensor mapping or a `TensorSource`; a configured `ValueSink` receives each value,
and the same values are returned as a dictionary.

Generated and hand-written observables can be placed in the same list or
`Pack`. Runtime materializes the iterable and rejects duplicate spec ids. Within
one observation it caches each source id, so multiple reductions share one
source lookup.

`Budget(max_compute_ms=None)` limits the sum of estimated observable cost per
`observe()` call. Scheduling uses these `ObservableSpec` fields:

- `frequency` must be positive. A spec runs only when
  `step % frequency == 0`; the default is every step.
- `budget_hint` may contain a `compute_ms` estimate. The estimate must be finite
  and non-negative; a missing hint counts as zero.
- `max_compute_ms` must be finite and non-negative when set. An observable is
  skipped when its estimate would exceed the remaining budget. The scheduler
  does not measure execution time or change a spec's frequency.
- `generate()` gives generated observables a small positive shape-aware
  `heuristic`, with a conservative `0.01` ms floor. It is scheduling input only;
  custom observables should provide explicit `budget_hint` values when a budget
  matters. M2 makes no `M3` cost-accuracy claim: the heuristic is not a calibrated cost-accuracy estimate.

`OfflineAnalyzer(observables, source, budget=None, sink=None)` exposes
`analyze(step, **context)` and delegates to the same runtime compute path.

## Temporal Operators

The temporal functions operate on explicit numeric histories:

- `delta(values, lag=1)` returns the latest value minus the value at `lag`.
- `ema(values, alpha)` returns the exponential moving average for
  `0 < alpha <= 1`.
- `slope(values)` returns the least-squares slope over equally spaced samples.
- `rolling_std(values, window)` returns the population standard deviation of
  the latest window.

The `temporal` field on `ObservableSpec` is metadata included in its id. M2's
`Runtime` does not apply a temporal function automatically.

## Storage And Query

`ValueSink` is the callable protocol
`(observable_id, step, value, meta) -> None`. `LocalStorage(root)` is an optional
implementation using SQLite metadata and NumPy NPZ array payloads.
`query(storage, observable_id, step)` performs the supported id/step readback.
It is not a general analysis query engine.

Observable has no separate display-name field. `ObservableSpec.id` is the
16-character storage and result key derived from the complete spec. Applications
may derive a readable label from source, transforms, and reduction, but M2 does
not query by that label.

## Filter Foundation

Subclass `Filter` and implement `apply(observables)`. A filter is also callable.
Compose filters with `&` for intersection and `|` for stable union. M3 owns
template filters and their generation pipeline. M2 includes no built-in
`BySource` or `ByReduction` filters; user filters operate on an already-generated
observable list.

For executable patterns and current limitations, read
[`usage.md`](usage.md).
