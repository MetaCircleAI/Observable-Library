import hashlib
import json
import pickle

import pytest

from observable_library.observable.spec import ObservableSpec


def test_observable_spec_stores_basic_fields_and_has_stable_id():
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=["center", "normalize"],
        reduction="l2_norm",
    )
    same_spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=["center", "normalize"],
        reduction="l2_norm",
    )

    assert spec.source == "param.layers.0.weight"
    assert spec.selector == "all"
    assert spec.transforms == ("center", "normalize")
    assert spec.reduction == "l2_norm"
    assert spec.id == same_spec.id


def test_observable_spec_default_scheduling_fields_do_not_change_legacy_id():
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=["center"],
        reduction="l2_norm",
    )
    legacy_payload = {
        "source": spec.source,
        "selector": spec.selector,
        "transforms": spec.transforms,
        "reduction": spec.reduction,
    }
    encoded = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))

    assert spec.temporal is None
    assert spec.frequency == 1
    assert dict(spec.budget_hint) == {}
    assert spec.id == hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def test_observable_spec_explicit_scheduling_fields_participate_in_id():
    base = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        reduction="l2_norm",
    )
    temporal = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        reduction="l2_norm",
        temporal="delta(lag=1)",
        frequency=5,
        budget_hint={"compute_ms": 1.5},
    )

    assert temporal.temporal == "delta(lag=1)"
    assert temporal.frequency == 5
    assert dict(temporal.budget_hint) == {"compute_ms": 1.5}
    assert temporal.id != base.id


def test_observable_spec_captures_immutable_semantic_fields() -> None:
    transforms = ["center"]
    budget_hint = {"compute_ms": 1.5}
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=transforms,
        reduction="l2_norm",
        budget_hint=budget_hint,
    )

    transforms.append("normalize")
    budget_hint["compute_ms"] = 2.0

    assert spec.transforms == ("center",)
    assert dict(spec.budget_hint) == {"compute_ms": 1.5}
    with pytest.raises((AttributeError, TypeError)):
        spec.transforms += ("normalize",)
    with pytest.raises(TypeError):
        spec.budget_hint["compute_ms"] = 2.0


def test_observable_spec_rejects_unbound_dict_mutation() -> None:
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        reduction="l2_norm",
        budget_hint={"compute_ms": 1.5},
    )
    original_id = spec.id

    with pytest.raises(TypeError):
        dict.update(spec.budget_hint, {"compute_ms": 2.0})

    assert spec.budget_hint["compute_ms"] == 1.5
    assert spec.budget_hint.get("missing", 0.0) == 0.0
    assert spec.id == original_id


def test_observable_spec_is_serializable_without_losing_immutability() -> None:
    spec = ObservableSpec(
        source="param.layers.0.weight",
        selector="all",
        transforms=["center"],
        reduction="l2_norm",
        budget_hint={"compute_ms": 1.5},
    )

    assert json.loads(json.dumps(spec.to_dict())) == {
        "source": "param.layers.0.weight",
        "selector": "all",
        "transforms": ["center"],
        "reduction": "l2_norm",
        "temporal": None,
        "frequency": 1,
        "budget_hint": {"compute_ms": 1.5},
    }

    restored = pickle.loads(pickle.dumps(spec))

    assert restored == spec
    assert restored.budget_hint.get("compute_ms") == 1.5
    with pytest.raises(TypeError):
        restored.budget_hint["compute_ms"] = 2.0
