from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from observable_library.generator.composer import compose_observables
from observable_library.generator.introspector import introspect_parameters
from observable_library.generator.matcher import match_identity
from observable_library.generator.resolver import resolve_templates
from observable_library.observable.observable import Observable

DEFAULT_REDUCTIONS = (
    "sum",
    "mean",
    "l1_norm",
    "l2_norm",
    "max",
    "min",
    "std",
    "variance",
    "abs_mean",
    "nonzero_count",
    "positive_fraction",
    "negative_fraction",
    "numel",
)


def generate(
    model: Any,
    reductions: Iterable[str] = DEFAULT_REDUCTIONS,
    transforms: Iterable[str] = (),
) -> list[Observable]:
    """Generate observables for a model from registered reductions and transforms."""
    tensors = introspect_parameters(model)
    templates = match_identity(tensors, reductions=reductions)
    resolved = resolve_templates(templates)
    return compose_observables(resolved, transforms=transforms)
