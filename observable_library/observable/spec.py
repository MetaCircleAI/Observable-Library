from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


class _ImmutableBudgetHint(Mapping[str, float]):
    __slots__ = ("_items",)
    _items: tuple[tuple[str, float], ...]

    def __init__(self, values: Mapping[str, float]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __getitem__(self, key: str) -> float:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("budget_hint is immutable.")

    def __reduce__(self) -> tuple[Any, tuple[dict[str, float]]]:
        return type(self), (dict(self),)


@dataclass(frozen=True)
class ObservableSpec:
    source: str
    selector: str
    transforms: Iterable[str] = field(default_factory=tuple)
    reduction: str = ""
    temporal: str | None = None
    frequency: int = 1
    budget_hint: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "budget_hint", _ImmutableBudgetHint(self.budget_hint))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "selector": self.selector,
            "transforms": list(self.transforms),
            "reduction": self.reduction,
            "temporal": self.temporal,
            "frequency": self.frequency,
            "budget_hint": dict(self.budget_hint),
        }

    @property
    def id(self) -> str:
        payload = self.to_dict()
        if self.temporal is None:
            del payload["temporal"]
        if self.frequency == 1:
            del payload["frequency"]
        if not self.budget_hint:
            del payload["budget_hint"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
