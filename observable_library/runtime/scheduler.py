from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from observable_library.observable.spec import ObservableSpec


@dataclass(frozen=True)
class Budget:
    max_compute_ms: float | None = None


def estimate_compute_ms(spec: ObservableSpec) -> float:
    compute_ms = float(spec.budget_hint.get("compute_ms", 0.0))
    if compute_ms < 0 or not isfinite(compute_ms):
        raise ValueError(
            "ObservableSpec.budget_hint['compute_ms'] must be finite and "
            f"non-negative, got {compute_ms!r}."
        )
    return compute_ms


def should_compute(
    spec: ObservableSpec,
    step: int,
    budget: Budget | None = None,
    spent_compute_ms: float = 0.0,
) -> bool:
    if spec.frequency <= 0:
        raise ValueError(
            "ObservableSpec.frequency must be greater than zero, "
            f"got {spec.frequency!r}."
        )
    compute_ms = estimate_compute_ms(spec)
    max_compute_ms = None if budget is None else budget.max_compute_ms
    if max_compute_ms is not None:
        max_compute_ms = float(max_compute_ms)
        if max_compute_ms < 0 or not isfinite(max_compute_ms):
            raise ValueError(
                "Budget.max_compute_ms must be finite and non-negative, "
                f"got {max_compute_ms!r}."
            )
    if step % spec.frequency != 0:
        return False
    if max_compute_ms is None:
        return True
    return spent_compute_ms + compute_ms <= max_compute_ms
