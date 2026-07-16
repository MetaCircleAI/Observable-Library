import torch

from observable_library.tensor.source import HookSource
from observable_library.tensor.typed_tensor import TypedTensor


def test_hook_source_captures_forward_activation_as_typed_tensor():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    source = HookSource(model)

    source.attach()
    model(torch.tensor([[2.0, 3.0]]))
    activation = source.get("activation.0", step=0)
    source.detach()

    assert isinstance(activation, TypedTensor)
    assert activation.stage == "activation"
    assert activation.axes == ("sample", "feature")
    assert activation.provenance == {"source": "activation.0", "step": 0}
    assert activation.value.shape == (1, 1)


def test_hook_source_captures_parameter_gradient_as_typed_tensor():
    model = torch.nn.Linear(2, 1, bias=False)
    source = HookSource(model)

    source.attach()
    output = model(torch.tensor([[2.0, 3.0]]))
    output.sum().backward()
    gradient = source.get("grad.weight", step=0)
    source.detach()

    assert isinstance(gradient, TypedTensor)
    assert gradient.stage == "gradient"
    assert gradient.axes == ("out_feature", "in_feature")
    assert gradient.provenance == {"source": "grad.weight", "step": 0}
    assert torch.equal(gradient.value, torch.tensor([[2.0, 3.0]]))


def test_hook_source_attach_skips_frozen_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = HookSource(model)

    try:
        source.attach()
        model(torch.tensor([[2.0, 3.0]]))
        activation = source.get("activation.0", step=0)
    finally:
        source.detach()

    assert activation.stage == "activation"


def test_hook_source_records_loss_as_typed_tensor():
    source = HookSource(torch.nn.Linear(2, 1))

    source.record_loss(torch.tensor(1.5), step=4)
    loss = source.get("loss", step=4)

    assert isinstance(loss, TypedTensor)
    assert loss.stage == "loss"
    assert loss.axes == ()
    assert loss.provenance == {"source": "loss", "step": 4}
    assert torch.equal(loss.value, torch.tensor(1.5))


def test_hook_source_extracts_tensor_from_structured_module_output():
    model = torch.nn.TransformerEncoderLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        dropout=0.0,
        batch_first=True,
    )
    source = HookSource(model)

    source.attach()
    model(torch.ones(1, 3, 4))
    activation = source.get("activation.self_attn", step=0)
    source.detach()

    assert activation.value.shape == (1, 3, 4)
