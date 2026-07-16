from __future__ import annotations

import torch

import observable_library as ol
from observable_library.tensor.source import HookSource


def test_hook_source_satisfies_public_tensor_source_protocol() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))

    source = HookSource(model)

    assert isinstance(source, ol.TensorSource)
