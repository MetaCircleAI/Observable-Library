from __future__ import annotations

from typing import Any

import observable_library as ol


class MemorySink:
    def __init__(self) -> None:
        self.records: list[tuple[str, int, Any, dict[str, Any]]] = []

    def __call__(
        self,
        observable_id: str,
        step: int,
        value: Any,
        meta: dict[str, Any],
    ) -> None:
        self.records.append((observable_id, step, value, meta))


def test_value_sink_protocol_accepts_user_callable_sink() -> None:
    sink = MemorySink()

    assert isinstance(sink, ol.ValueSink)
