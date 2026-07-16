import torch

from observable_library.generator.introspector import introspect_parameters


def test_introspect_parameters_records_sources_shapes_and_axis_types():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))

    metadata = introspect_parameters(model)

    assert metadata == [
        {
            "source": "param.0.weight",
            "selector": "all",
            "shape": (3, 2),
            "axes": ("out_feature", "in_feature"),
        },
        {
            "source": "param.0.bias",
            "selector": "all",
            "shape": (3,),
            "axes": ("feature",),
        },
    ]
