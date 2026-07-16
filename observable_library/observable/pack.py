from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from observable_library.observable.observable import Observable


@dataclass(frozen=True)
class Pack:
    observables: tuple[Observable, ...]

    def __init__(self, observables: Iterable[Observable]) -> None:
        object.__setattr__(self, "observables", tuple(observables))

    def __iter__(self) -> Iterator[Observable]:
        return iter(self.observables)

    def __len__(self) -> int:
        return len(self.observables)
