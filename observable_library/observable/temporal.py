from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def delta(values: Sequence[float], lag: int = 1) -> float:
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    if len(values) <= lag:
        raise ValueError("values must contain more items than lag.")
    return float(values[-1] - values[-1 - lag])


def ema(values: Sequence[float], alpha: float) -> float:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    if not values:
        raise ValueError("values must not be empty.")
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1 - alpha) * current
    return current


def slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("values must contain at least two items.")
    y = [float(value) for value in values]
    x_mean = (len(y) - 1) / 2
    y_mean = sum(y) / len(y)
    numerator = sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(y)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(y)))
    return float(numerator / denominator)


def rolling_std(values: Sequence[float], window: int) -> float:
    if window < 1:
        raise ValueError("window must be >= 1.")
    if len(values) < window:
        raise ValueError("values must contain at least window items.")
    return float(np.std(np.asarray(values[-window:], dtype=float)))
