from observable_library.tensor.typed_tensor import TypedTensor


def test_typed_tensor_stores_value_axes_stage_and_provenance():
    tensor = TypedTensor(
        value=[[1.0, 2.0]],
        axes=("sample", "feature"),
        stage="activation",
        provenance={"module": "layers.0", "step": 3},
    )

    assert tensor.value == [[1.0, 2.0]]
    assert tensor.axes == ("sample", "feature")
    assert tensor.stage == "activation"
    assert tensor.provenance == {"module": "layers.0", "step": 3}
