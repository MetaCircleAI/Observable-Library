import torch

from observable_library.generator.composer import compose_observables
from observable_library.generator.introspector import introspect_parameters
from observable_library.generator.matcher import match_identity
from observable_library.generator.resolver import resolve_templates


def test_generator_pipeline_builds_identity_observables_from_model_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))

    tensors = introspect_parameters(model)
    templates = match_identity(tensors, reductions=["l2_norm"])
    resolved = resolve_templates(templates)
    observables = compose_observables(resolved)

    spec_keys = {
        (observable.spec.source, observable.spec.selector, observable.spec.reduction)
        for observable in observables
    }

    assert spec_keys == {
        ("param.0.weight", "all", "l2_norm"),
        ("param.0.bias", "all", "l2_norm"),
    }
    assert all("identity" in observable.tags for observable in observables)
