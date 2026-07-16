from __future__ import annotations

import numpy as np
import torch

import observable_library as ol


def test_file_source_reads_npz_tensor_as_typed_tensor(tmp_path) -> None:
    tensor_path = tmp_path / "tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})

    source = ol.FileSource(tensor_path)

    typed = source.get("param.layer.weight", step=7)

    assert isinstance(source, ol.TensorSource)
    assert typed.stage == "offline"
    assert typed.axes == ("axis_0", "axis_1")
    assert typed.provenance == {"source": "param.layer.weight", "step": 7}
    assert torch.equal(typed.value, torch.tensor([[1.0, 2.0]], dtype=torch.float64))


def test_checkpoint_source_reads_state_dict_parameter(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"layer.weight": torch.tensor([[1.0, 2.0]])}, checkpoint_path)

    source = ol.CheckpointSource(checkpoint_path)

    typed = source.get("param.layer.weight", step=11)

    assert isinstance(source, ol.TensorSource)
    assert typed.stage == "offline"
    assert typed.axes == ("out_feature", "in_feature")
    assert typed.provenance == {"source": "param.layer.weight", "step": 11}
    assert torch.equal(typed.value, torch.tensor([[1.0, 2.0]]))


def test_checkpoint_source_loads_weights_only(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "model.pt"
    captured = {}

    def load(path, *, map_location, weights_only):
        captured.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {"layer.weight": torch.ones(1, 1)}

    monkeypatch.setattr("observable_library.tensor.source.torch.load", load)

    ol.CheckpointSource(checkpoint_path)

    assert captured == {
        "path": checkpoint_path,
        "map_location": "cpu",
        "weights_only": True,
    }
