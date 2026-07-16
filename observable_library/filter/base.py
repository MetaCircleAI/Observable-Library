from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from observable_library.observable.observable import Observable


class Filter(ABC):
    @abstractmethod
    def apply(self, observables: Iterable[Observable]) -> list[Observable]:
        raise NotImplementedError

    def __call__(self, observables: Iterable[Observable]) -> list[Observable]:
        return self.apply(observables)

    def __and__(self, other: Filter) -> Filter:
        return AndFilter(self, other)

    def __or__(self, other: Filter) -> Filter:
        return OrFilter(self, other)


@dataclass(frozen=True)
class AndFilter(Filter):
    left: Filter
    right: Filter

    def apply(self, observables: Iterable[Observable]) -> list[Observable]:
        candidates = list(observables)
        left_ids = {id(observable) for observable in self.left.apply(candidates)}
        right_ids = {id(observable) for observable in self.right.apply(candidates)}
        return [
            observable
            for observable in candidates
            if id(observable) in left_ids and id(observable) in right_ids
        ]


@dataclass(frozen=True)
class OrFilter(Filter):
    left: Filter
    right: Filter

    def apply(self, observables: Iterable[Observable]) -> list[Observable]:
        candidates = list(observables)
        combined = self.left.apply(candidates) + self.right.apply(candidates)
        selected_ids = {id(observable) for observable in combined}
        return [
            observable for observable in candidates if id(observable) in selected_ids
        ]
