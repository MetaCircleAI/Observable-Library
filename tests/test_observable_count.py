import torch

import observable_library as ol


def test_public_default_generation_executes_50_plus_observables_for_mlp_and_cnn():
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 6 * 6, 2),
    )

    mlp_observables = ol.generate(mlp)
    cnn_observables = ol.generate(cnn)

    mlp_values = ol.Runtime(mlp_observables, source=ol.HookSource(mlp)).observe(step=0)
    cnn_values = ol.Runtime(cnn_observables, source=ol.HookSource(cnn)).observe(step=0)

    assert len(mlp_observables) >= 50
    assert len(cnn_observables) >= 50
    assert len(mlp_values) == len(mlp_observables)
    assert len(cnn_values) == len(cnn_observables)
    assert len({observable.spec.id for observable in mlp_observables}) == len(
        mlp_observables
    )
    assert len({observable.spec.id for observable in cnn_observables}) == len(
        cnn_observables
    )
