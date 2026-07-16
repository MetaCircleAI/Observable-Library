from __future__ import annotations

import tempfile

import numpy as np

import observable_library as ol


class TinyModel:
    def named_parameters(self):
        return [("layer.weight", np.array([[1.0, 2.0]]))]


with tempfile.TemporaryDirectory() as directory:
    tensor_path = f"{directory}/tensors.npz"
    np.savez(tensor_path, **{"param.layer.weight": np.array([[1.0, 2.0]])})

    observables = ol.generate(TinyModel(), reductions=["sum"])
    runtime = ol.Runtime(observables=observables, source=ol.FileSource(tensor_path))
    RESULTS = runtime.observe(step=0)
