from observable_library.filter.identity import identity_observables


def test_identity_observables_generate_specs_from_parameter_metadata():
    observables = identity_observables(
        [
            {"source": "param.layers.0.weight", "selector": "all"},
            {"source": "param.layers.0.bias", "selector": "all"},
        ],
        reductions=["l2_norm", "mean"],
    )

    spec_keys = {
        (observable.spec.source, observable.spec.selector, observable.spec.reduction)
        for observable in observables
    }

    assert spec_keys == {
        ("param.layers.0.weight", "all", "l2_norm"),
        ("param.layers.0.weight", "all", "mean"),
        ("param.layers.0.bias", "all", "l2_norm"),
        ("param.layers.0.bias", "all", "mean"),
    }
    assert all("identity" in observable.tags for observable in observables)
