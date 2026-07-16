from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from observable_library.tensor.typed_tensor import TypedTensor


@runtime_checkable
class TensorSource(Protocol):
    def get(self, source_id: str, step: int) -> TypedTensor: ...


class HookSource:
    """Capture model activations, gradients, and recorded losses for Runtime."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._handles: list[Any] = []
        self._activations: dict[str, Any] = {}
        self._gradients: dict[str, Any] = {}
        self._losses: dict[int, Any] = {}

    def attach(self) -> None:
        for name, module in self.model.named_children():
            source_id = f"activation.{name}"
            handle = module.register_forward_hook(self._capture_activation(source_id))
            self._handles.append(handle)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            source_id = f"grad.{name}"
            handle = parameter.register_hook(self._capture_gradient(source_id))
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get(self, source_id: str, step: int) -> TypedTensor:
        if source_id.startswith("activation."):
            value = self._activations[source_id]
            stage = "activation"
            axes = _activation_axes(value)
        elif source_id == "loss":
            value = self._losses[step]
            stage = "loss"
            axes = ()
        elif source_id.startswith("param."):
            parameter_name = source_id.removeprefix("param.")
            value = self.model.get_parameter(parameter_name).detach()
            stage = "parameter"
            axes = _parameter_axes(source_id, tuple(value.shape))
        else:
            value = self._gradients[source_id]
            stage = "gradient"
            axes = _parameter_axes(source_id, tuple(value.shape))
        return TypedTensor(
            value=value,
            axes=axes,
            stage=stage,
            provenance={"source": source_id, "step": step},
        )

    def record_loss(self, loss: Any, step: int) -> None:
        self._losses[step] = loss.detach()

    def _capture_activation(self, source_id: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            value = _first_tensor(output)
            if value is not None:
                self._activations[source_id] = value.detach()

        return hook

    def _capture_gradient(self, source_id: str):
        def hook(gradient: Any) -> None:
            self._gradients[source_id] = gradient.detach()

        return hook


class CheckpointSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state_dict = torch.load(self.path, map_location="cpu", weights_only=True)

    def get(self, source_id: str, step: int) -> TypedTensor:
        if not source_id.startswith("param."):
            raise KeyError(
                f"CheckpointSource only supports param sources, got {source_id!r}."
            )
        parameter_name = source_id.removeprefix("param.")
        value = self.state_dict[parameter_name]
        return TypedTensor(
            value=value,
            axes=_parameter_axes(source_id, tuple(value.shape)),
            stage="offline",
            provenance={"source": source_id, "step": step},
        )


class FileSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get(self, source_id: str, step: int) -> TypedTensor:
        with np.load(self.path) as payload:
            value = torch.as_tensor(payload[source_id])
        return TypedTensor(
            value=value,
            axes=tuple(f"axis_{index}" for index, _size in enumerate(value.shape)),
            stage="offline",
            provenance={"source": source_id, "step": step},
        )


def _activation_axes(value: Any) -> tuple[str, ...]:
    if len(value.shape) == 2:
        return ("sample", "feature")
    return tuple(f"axis_{index}" for index, _size in enumerate(value.shape))


def _first_tensor(value: Any) -> Any | None:
    if torch.is_tensor(value):
        return value
    values: Iterable[Any]
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return None
    for item in values:
        tensor = _first_tensor(item)
        if tensor is not None:
            return tensor
    return None


def _parameter_axes(source_id: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    if source_id.endswith(".weight") and len(shape) == 2:
        return ("out_feature", "in_feature")
    if source_id.endswith(".bias") and len(shape) == 1:
        return ("feature",)
    return tuple(f"axis_{index}" for index, _size in enumerate(shape))
