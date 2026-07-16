from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from observable_library.observable.observable import Observable
from observable_library.observable.pack import Pack
from observable_library.runtime.runtime import Runtime
from observable_library.runtime.scheduler import Budget
from observable_library.storage.protocol import ValueSink
from observable_library.tensor.source import TensorSource


@dataclass
class OfflineAnalyzer:
    observables: Pack | Iterable[Observable]
    source: TensorSource
    budget: Budget | None = None
    sink: ValueSink | None = None

    def __post_init__(self) -> None:
        self.observables = tuple(self.observables)

    def analyze(self, step: int, **context: Any) -> dict[str, Any]:
        return Runtime(
            observables=self.observables,
            source=self.source,
            budget=self.budget,
            sink=self.sink,
        ).observe(step=step, **context)
