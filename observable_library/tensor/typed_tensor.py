from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TypedTensor:
    value: Any
    axes: tuple[str, ...]
    stage: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
