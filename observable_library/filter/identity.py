from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from observable_library.observable.observable import Observable
from observable_library.observable.registry import (
    Reduction,
    Transform,
    get_reduction,
    get_transform,
)
from observable_library.observable.spec import ObservableSpec


def identity_observables(
    tensor_metadata: Iterable[Mapping[str, Any]],
    reductions: Iterable[str],
    transforms: Iterable[str] = (),
) -> list[Observable]:
    observables: list[Observable] = []
    transform_names = tuple(transforms)
    resolved_transforms = tuple(get_transform(name) for name in transform_names)
    for tensor in tensor_metadata:
        for reduction in reductions:
            try:
                resolved_reduction = get_reduction(reduction)
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"Unsupported reduction {reduction!r} for tensor source "
                    f"{tensor['source']!r}."
                ) from error
            spec = ObservableSpec(
                source=tensor["source"],
                selector=tensor["selector"],
                transforms=transform_names,
                reduction=reduction,
                budget_hint={
                    "compute_ms": _generated_compute_ms(
                        tensor.get("shape", ()), tensor.get("axes", ())
                    )
                },
            )
            observables.append(
                Observable(
                    spec=spec,
                    compute=_compute_reduction(
                        spec.source,
                        reduction,
                        resolved_transforms,
                        resolved_reduction,
                    ),
                    tags={"identity"},
                )
            )
    return observables


def _generated_compute_ms(shape: Any, axes: Any) -> float:
    size = 1
    for dimension in shape:
        size *= dimension
    return max(0.01, size * max(1, len(axes)) / 100_000)


def _compute_reduction(
    source: str,
    reduction_name: str,
    transforms: tuple[Transform, ...],
    reduction: Reduction,
):
    def compute(tensors: dict[str, Any], context: dict[str, Any]) -> Any:
        if source not in tensors:
            raise KeyError(
                f"Missing tensor source {source!r} for reduction {reduction_name!r}."
            )
        value = tensors[source]
        for transform in transforms:
            value = transform(value)
        return reduction(value)

    return compute
