---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
---

(home)=
# Generate and compute training observables without changing your training loop.

::::{grid} 1 1 2 2
:gutter: 4
:class-container: hero-grid

:::{grid-item}
:class: hero-copy

**PyTorch training observables · 0.1.0**

Observable Library is a small Python package for defining, generating, and computing observables from a running PyTorch training loop. Return values directly or store them for exact id-and-step readback.

{ref}`Get started <quickstart>` · {ref}`API reference <api-reference>`
:::

:::{grid-item}
:class: hero-code-stage

```{code-block} python
:class: hero-live-code

import observable_library as ol

# You choose the observables. Nothing is inferred automatically.
observables = ol.generate(model, reductions=["l2_norm"])
runtime = ol.Runtime(observables, source=ol.HookSource(model))

if step % 100 == 0:
    values = runtime.observe(step=step)
```

- `param_norm` · illustrative
- `observed_values` · illustrative
:::
::::

:::{container} workflow-strip
`generate` observables → `Runtime.observe()` values → optional `ValueSink` storage
:::

## Install

Requires Python 3.10–3.12 and `torch>=2.4.1`. The package is published on PyPI under Apache-2.0.

```bash
python -m pip install observable-library
```

## Core data flow

You explicitly select the reductions and the steps at which computation happens. Parameter generation does not change the training loop and does not automatically create activation, gradient, or loss observables.

```python
observables = ol.generate(model, reductions=["l2_norm"])
runtime = ol.Runtime(observables, source=ol.HookSource(model))
values = runtime.observe(step=0)
```

## Where to go next

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: ref

Go from installation to the first observed and stored value with a small CPU example.
:::

:::{grid-item-card} Concepts
:link: concepts
:link-type: ref

Understand specs, sources, packs, runtime scheduling, storage, and stable ids.
:::

:::{grid-item-card} How-to Guides
:link: how-to-guides
:link-type: ref

Use online hooks, custom observables, filters, transforms, reductions, and offline files.
:::

:::{grid-item-card} API Reference
:link: api-reference
:link-type: ref

Browse the complete top-level 0.1.0 public API and its current behavior.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
concepts
usage
api
limitations
```
