from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Transform = Callable[[Any], Any]
Reduction = Callable[[Any], Any]

_transforms: dict[str, Transform] = {}
_reductions: dict[str, Reduction] = {}


def register_transform(name: str) -> Callable[[Transform], Transform]:
    def decorator(transform: Transform) -> Transform:
        if name in _transforms:
            raise ValueError(f"Transform {name!r} is already registered.")
        _transforms[name] = transform
        return transform

    return decorator


def register_reduction(name: str) -> Callable[[Reduction], Reduction]:
    def decorator(reduction: Reduction) -> Reduction:
        if name in _reductions:
            raise ValueError(f"Reduction {name!r} is already registered.")
        _reductions[name] = reduction
        return reduction

    return decorator


def get_transform(name: str) -> Transform:
    try:
        return _transforms[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown transform {name!r}; register it before use."
        ) from error


def get_reduction(name: str) -> Reduction:
    try:
        return _reductions[name]
    except KeyError as error:
        raise NotImplementedError(
            f"Unsupported reduction {name!r}; register it before use."
        ) from error


def _population_std(tensor: Any) -> Any:
    if isinstance(tensor, np.ndarray):
        return tensor.std(ddof=0)
    return tensor.std(correction=0)


def _population_variance(tensor: Any) -> Any:
    if isinstance(tensor, np.ndarray):
        return tensor.var(ddof=0)
    return tensor.var(correction=0)


register_reduction("sum")(lambda tensor: tensor.sum())
register_reduction("mean")(lambda tensor: tensor.mean())
register_reduction("l1_norm")(lambda tensor: tensor.abs().sum())
register_reduction("l2_norm")(lambda tensor: tensor.norm())
register_reduction("max")(lambda tensor: tensor.max())
register_reduction("min")(lambda tensor: tensor.min())
register_reduction("std")(_population_std)
register_reduction("variance")(_population_variance)
register_reduction("abs_mean")(lambda tensor: tensor.abs().mean())
register_reduction("nonzero_count")(lambda tensor: (tensor != 0).sum())
register_reduction("positive_fraction")(lambda tensor: (tensor > 0).float().mean())
register_reduction("negative_fraction")(lambda tensor: (tensor < 0).float().mean())
register_reduction("numel")(lambda tensor: tensor.numel())
