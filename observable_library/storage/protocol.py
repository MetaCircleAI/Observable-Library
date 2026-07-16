from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ValueSink(Protocol):
    def __call__(
        self,
        observable_id: str,
        step: int,
        value: Any,
        meta: dict[str, Any],
    ) -> None: ...
