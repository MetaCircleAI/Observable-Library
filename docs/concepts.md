(concepts)=
# Concepts

Observable Library separates **what to compute**, **where tensors come from**, and **when computation runs**.

## Observable specs and identity

`ObservableSpec` describes a source, selector, ordered transform chain, reduction, temporal metadata, frequency, and budget hint. Its `id` is a stable 16-character hash of the complete spec. Change an identity field and the id can change.

`Observable` pairs a spec with a compute callable. `Pack` is the stable iterable container accepted by execution APIs and rejects duplicate ids when materialized by Runtime.

## Generation

`generate(model, reductions=..., transforms=...)` enumerates every `model.named_parameters()` entry. For each parameter and requested reduction it creates one observable.

Generation in 0.1.0 does not discover activations, gradients, or loss values. Add those as hand-written observables and supply an appropriate source.

Transforms run from left to right before the reduction. The registry validates transform and reduction names during generation; tensor compatibility is checked only when the operation executes.

## Sources

All sources implement the `TensorSource` protocol:

- `HookSource` reads live parameters and captured activations, gradients, and explicitly recorded losses.
- `FileSource` reads arrays by key from an NPZ file.
- `CheckpointSource` reads `param.*` values from a Torch state dict.

Source values are wrapped in `TypedTensor`, which can carry axes, stage, and provenance metadata.

## Runtime and scheduling

`Runtime.observe()` computes every eligible observable for a step and returns a dictionary keyed by spec id. Within one call, Runtime caches each source lookup so multiple reductions can share one tensor read.

`frequency` controls step eligibility. `Budget(max_compute_ms=...)` limits the sum of declared `budget_hint["compute_ms"]` estimates. The scheduler does not measure wall-clock execution time or recalibrate estimates.

`OfflineAnalyzer` delegates to the same runtime path through `analyze(step, **context)`.

## Storage

`ValueSink` is the storage contract. `LocalStorage` is one optional implementation using SQLite metadata and NPZ payload files. `query()` supports exact observable id and step readback only; it is not a general analytics query engine.

## Selection and temporal functions

Observable Library 0.1.0 supports only `selector="all"`. The `delta`, `ema`, `slope`, and `rolling_std` functions operate on explicit numeric histories. A spec's `temporal` field contributes to identity but Runtime does not apply a temporal function automatically.
