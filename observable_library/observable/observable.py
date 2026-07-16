from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from observable_library.observable.spec import ObservableSpec


@dataclass
class Observable:
    spec: ObservableSpec
    compute: Callable[[dict[str, Any], dict[str, Any]], Any]
    tags: set[str] = field(default_factory=set)
