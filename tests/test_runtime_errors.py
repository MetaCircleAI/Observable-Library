import pytest

from observable_library.filter.identity import identity_observables


def test_observable_error_names_missing_tensor_source() -> None:
    observable = identity_observables(
        [{"source": "param.missing.weight", "selector": "all"}],
        reductions=["l2_norm"],
    )[0]

    with pytest.raises(KeyError, match="param.missing.weight.*l2_norm"):
        observable.compute({}, {"step": 0})


def test_observable_error_names_unsupported_reduction_and_source() -> None:
    with pytest.raises(NotImplementedError, match=r"median.*param\.0\.weight"):
        identity_observables(
            [{"source": "param.0.weight", "selector": "all"}],
            reductions=["median"],
        )
