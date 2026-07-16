from __future__ import annotations

from typing import Any

from observable_library.storage.local import LocalStorage


def query(storage: LocalStorage, observable_id: str, step: int) -> Any:
    """Read one LocalStorage value by its exact observable id and step."""
    return storage.read(observable_id, step)
