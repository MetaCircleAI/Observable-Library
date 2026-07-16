from observable_library.observable.observable import Observable
from observable_library.observable.spec import ObservableSpec


def test_observable_calls_compute_function_and_carries_tags():
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=[],
        reduction="sum",
    )

    observable = Observable(
        spec=spec,
        compute=lambda tensors, context: (
            tensors["param.layers.0.weight"] + context["bias"]
        ),
        tags={"identity", "parameter"},
    )

    value = observable.compute(
        {"param.layers.0.weight": 3},
        {"bias": 4},
    )

    assert observable.spec is spec
    assert value == 7
    assert observable.tags == {"identity", "parameter"}
