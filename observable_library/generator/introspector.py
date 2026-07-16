from __future__ import annotations

from typing import Any, TypedDict


class TensorMetadata(TypedDict):
    source: str
    selector: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]


def introspect_parameters(model: Any) -> list[TensorMetadata]:
    return [
        {
            "source": f"param.{name}",
            "selector": "all",
            "shape": tuple(parameter.shape),
            "axes": _parameter_axes(name, tuple(parameter.shape)),
        }
        for name, parameter in model.named_parameters()
    ]


def _parameter_axes(name: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    if name.endswith(".weight") and len(shape) == 2:
        return ("out_feature", "in_feature")
    if name.endswith(".bias") and len(shape) == 1:
        return ("feature",)
    return tuple(f"axis_{index}" for index, _size in enumerate(shape))
