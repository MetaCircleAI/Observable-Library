import torch

import observable_library as ol


def test_tiny_training_loop_flow_captures_activation_gradient_and_loss():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    source = ol.HookSource(model)

    source.attach()
    output = model(torch.tensor([[2.0, 3.0]]))
    loss = output.sum()
    source.record_loss(loss, step=0)
    loss.backward()

    activation = source.get("activation.0", step=0)
    gradient = source.get("grad.0.weight", step=0)
    recorded_loss = source.get("loss", step=0)
    source.detach()

    assert activation.stage == "activation"
    assert gradient.stage == "gradient"
    assert recorded_loss.stage == "loss"
    assert activation.value.shape == (1, 1)
    assert gradient.value.shape == (1, 2)
    assert torch.equal(recorded_loss.value, loss.detach())


def test_runtime_observes_a_hook_captured_gradient_before_optimizer_step():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    source = ol.HookSource(model)
    observable = ol.Observable(
        spec=ol.ObservableSpec(
            source="grad.0.weight",
            selector="all",
            reduction="l2_norm",
        ),
        compute=lambda tensors, _context: tensors["grad.0.weight"].norm(),
    )
    runtime = ol.Runtime([observable], source=source)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    source.attach()
    loss = model(torch.tensor([[2.0, 3.0]])).sum()
    loss.backward()
    values = runtime.observe(step=0)
    expected = model[0].weight.grad.detach().norm()
    optimizer.step()
    source.detach()

    assert torch.equal(values[observable.spec.id], expected)
