from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from observable_library.generator.introspector import TensorMetadata


class IdentityTemplate(TypedDict):
    source: str
    selector: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    reduction: str
    filter: str


def match_identity(
    tensor_metadata: Iterable[TensorMetadata],
    reductions: Iterable[str],
) -> list[IdentityTemplate]:
    templates: list[IdentityTemplate] = []
    for tensor in tensor_metadata:
        for reduction in reductions:
            templates.append(
                {
                    "source": tensor["source"],
                    "selector": tensor["selector"],
                    "shape": tensor["shape"],
                    "axes": tensor["axes"],
                    "reduction": reduction,
                    "filter": "identity",
                }
            )
    return templates
