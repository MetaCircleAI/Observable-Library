from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from observable_library.observable.observable import Observable
from observable_library.runtime.scheduler import (
    Budget,
    estimate_compute_ms,
    should_compute,
)
from observable_library.storage.protocol import ValueSink
from observable_library.tensor.source import TensorSource
from observable_library.tensor.typed_tensor import TypedTensor


@dataclass
class Runtime:
    observables: Iterable[Observable]
    tensors: dict[str, Any] | None = None
    source: TensorSource | None = None
    budget: Budget | None = None
    sink: ValueSink | None = None

    def __post_init__(self) -> None:
        self.observables = tuple(self.observables)
        if self.tensors is not None and self.source is not None:
            raise ValueError("Runtime accepts tensors or a TensorSource, not both.")
        observable_ids = [observable.spec.id for observable in self.observables]
        if len(observable_ids) != len(set(observable_ids)):
            raise ValueError("Runtime requires unique observable ids.")

    def observe(self, step: int, **context: Any) -> dict[str, Any]:
        """Compute scheduled observables for one step and optionally store their values."""
        runtime_context = {"step": step, **context}
        values: dict[str, Any] = {}
        spent_compute_ms = 0.0
        source_cache: dict[str, TypedTensor] = {}
        for observable in self.observables:
            if observable.spec.selector != "all":
                raise NotImplementedError(
                    f"Runtime only supports selector='all', got {observable.spec.selector!r}."
                )
            if not should_compute(observable.spec, step, self.budget, spent_compute_ms):
                continue
            spent_compute_ms += estimate_compute_ms(observable.spec)
            tensors, typed = self._tensors_for(observable, step, source_cache)
            compute_context = dict(runtime_context)
            if typed is not None:
                compute_context["typed_tensor"] = typed
            value = observable.compute(tensors, compute_context)
            observable_id = observable.spec.id
            values[observable_id] = value
            if self.sink is not None:
                meta: dict[str, Any] = {
                    "source": observable.spec.source,
                    "reduction": observable.spec.reduction,
                }
                if typed is not None:
                    meta.update(
                        axes=typed.axes,
                        stage=typed.stage,
                        provenance=typed.provenance,
                    )
                self.sink(
                    observable_id,
                    step,
                    value,
                    meta,
                )
        return values

    def _tensors_for(
        self,
        observable: Observable,
        step: int,
        source_cache: dict[str, TypedTensor],
    ) -> tuple[dict[str, Any], TypedTensor | None]:
        if self.source is not None:
            source_id = observable.spec.source
            typed = source_cache.get(source_id)
            if typed is None:
                typed = self.source.get(source_id, step)
                source_cache[source_id] = typed
            return {source_id: typed.value}, typed
        if self.tensors is None:
            raise ValueError("Runtime requires either tensors or a TensorSource.")
        return self.tensors, None
