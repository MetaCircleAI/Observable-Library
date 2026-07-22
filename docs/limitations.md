(limitations)=
# Limitations

Observable Library 0.1.0 is a deliberately small, headless package. These boundaries are part of the released behavior.

## Generation is parameter-only

`generate()` enumerates `model.named_parameters()`. It has no source allowlist and does not generate activation, gradient, or loss observables. Filter the returned list to reduce Runtime work or construct a custom `Observable` for another source.

## Selection supports only all values

The only supported selector is `selector="all"`. Slice, axis, and semantic selectors are not part of 0.1.0.

## Hooks retain the latest value

`HookSource.attach()` installs broad hooks on top-level child activations and trainable parameter gradients. It retains the latest captured tensor without checking that it belongs to the current step. Run the matching forward/backward before every observation and call `detach()` when finished.

Loss is not discovered automatically. Call `record_loss(loss, step)` before observing a custom `source="loss"` spec.

## Budgeting uses estimates

The scheduler sums `budget_hint["compute_ms"]` estimates. It does not measure execution time, learn a cost model, or change a spec's frequency. A missing custom estimate counts as zero.

## Temporal metadata does not execute

The `temporal` field contributes to `ObservableSpec.id`, but Runtime does not automatically call `delta`, `ema`, `slope`, or `rolling_std`. Apply those functions to explicit histories.

## Storage and query are intentionally narrow

`LocalStorage` is a convenience `ValueSink`: SQLite metadata plus NumPy NPZ payloads. `query()` reads by exact observable id and step only. There is no query by source, reduction, tag, or display label.

## No application surface

Version 0.1.0 has no CLI, Web UI, notebook product, dataset downloader, or general analysis service. Later roadmap ideas are not current package capabilities.
