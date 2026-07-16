from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch


class LocalStorage:
    """Store observations in SQLite metadata and NPZ payloads by exact id and step."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.array_dir = self.root / "arrays"
        self.array_dir.mkdir(exist_ok=True)
        self.database_path = self.root / "metadata.sqlite3"
        self._initialize()

    def __call__(
        self,
        observable_id: str,
        step: int,
        value: Any,
        meta: dict[str, Any],
    ) -> None:
        value = _normalize_value(value)
        meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        if _is_scalar(value):
            self._write_row(
                observable_id, step, "scalar", float(value), meta_json, None
            )
            return

        payload_path = self._payload_path(observable_id, step)
        np.savez_compressed(self.root / payload_path, value=np.asarray(value))
        self._write_row(
            observable_id, step, "array", None, meta_json, str(payload_path)
        )

    def read(self, observable_id: str, step: int) -> Any:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT value_kind, scalar_value, payload_path
                FROM observations
                WHERE observable_id = ? AND step = ?
                """,
                (observable_id, step),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"No stored value for observable_id={observable_id!r}, step={step}."
            )

        value_kind, scalar_value, payload_path = row
        if value_kind == "scalar":
            return scalar_value
        with np.load(self.root / payload_path) as payload:
            return payload["value"]

    def _initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    observable_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    value_kind TEXT NOT NULL,
                    scalar_value REAL,
                    meta_json TEXT NOT NULL,
                    payload_path TEXT,
                    PRIMARY KEY (observable_id, step)
                )
                """
            )

    def _write_row(
        self,
        observable_id: str,
        step: int,
        value_kind: str,
        scalar_value: float | None,
        meta_json: str,
        payload_path: str | None,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observations (
                    observable_id, step, value_kind, scalar_value, meta_json, payload_path
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observable_id,
                    step,
                    value_kind,
                    scalar_value,
                    meta_json,
                    payload_path,
                ),
            )

    def _payload_path(self, observable_id: str, step: int) -> Path:
        digest = hashlib.sha256(f"{observable_id}:{step}".encode("utf-8")).hexdigest()[
            :16
        ]
        return Path("arrays") / f"{digest}.npz"


def _is_scalar(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, np.generic):
        return True
    array = np.asarray(value)
    return array.shape == ()


def _normalize_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return value
