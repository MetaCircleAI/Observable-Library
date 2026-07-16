from __future__ import annotations

import sqlite3

import numpy as np
import torch

import observable_library as ol


def test_local_storage_writes_scalar_metadata_to_sqlite(tmp_path) -> None:
    storage = ol.LocalStorage(tmp_path)

    storage("loss", 3, 1.25, {"source": "loss", "reduction": "mean"})

    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT observable_id, step, value_kind, scalar_value, meta_json, payload_path
            FROM observations
            """
        ).fetchone()

    assert row == (
        "loss",
        3,
        "scalar",
        1.25,
        '{"reduction":"mean","source":"loss"}',
        None,
    )
    assert storage.read("loss", 3) == 1.25


def test_local_storage_writes_array_payload_to_npz_and_reads_by_id_step(
    tmp_path,
) -> None:
    storage = ol.LocalStorage(tmp_path)
    value = np.array([[1.0, 2.0], [3.0, 4.0]])

    storage("activation.layer0", 5, value, {"source": "activation.layer0"})

    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT value_kind, scalar_value, payload_path
            FROM observations
            WHERE observable_id = ? AND step = ?
            """,
            ("activation.layer0", 5),
        ).fetchone()

    assert row[0] == "array"
    assert row[1] is None
    assert row[2].endswith(".npz")
    assert (tmp_path / row[2]).exists()
    np.testing.assert_array_equal(storage.read("activation.layer0", 5), value)


def test_local_storage_normalizes_autograd_torch_tensors(tmp_path) -> None:
    storage = ol.LocalStorage(tmp_path)
    value = torch.tensor([[1.0, 2.0]], requires_grad=True)

    storage("param.layer.weight", 1, value, {"source": "param.layer.weight"})

    stored = storage.read("param.layer.weight", 1)
    assert isinstance(stored, np.ndarray)
    np.testing.assert_array_equal(stored, np.array([[1.0, 2.0]]))


def test_local_storage_normalizes_bfloat16_torch_tensors(tmp_path) -> None:
    storage = ol.LocalStorage(tmp_path)
    value = torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)

    storage("activation.layer0", 1, value, {"source": "activation.layer0"})

    stored = storage.read("activation.layer0", 1)
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, value.float().numpy())
