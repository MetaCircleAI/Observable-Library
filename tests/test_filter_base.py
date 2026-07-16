from __future__ import annotations

import observable_library as ol


def _tag_filter(tag: str):
    class TagFilter(ol.Filter):
        def apply(self, observables):
            return [observable for observable in observables if tag in observable.tags]

    return TagFilter()


def _observable(reduction: str, tags: set[str]) -> ol.Observable:
    spec = ol.ObservableSpec(source="param.weight", selector="all", reduction=reduction)
    return ol.Observable(spec=spec, compute=lambda tensors, context: None, tags=tags)


def test_filter_and_composition_preserves_candidate_order() -> None:
    first = _observable("sum", {"spectral", "cheap"})
    second = _observable("mean", {"spectral"})
    third = _observable("max", {"cheap"})

    assert (_tag_filter("spectral") & _tag_filter("cheap")).apply(
        [first, second, third]
    ) == [first]


def test_filter_and_distinguishes_candidates_with_same_spec_id() -> None:
    left_only = _observable("sum", {"left"})
    right_only = _observable("sum", {"right"})

    assert left_only.spec.id == right_only.spec.id
    assert (_tag_filter("left") & _tag_filter("right")).apply(
        [left_only, right_only]
    ) == []


def test_filter_or_composition_is_a_stable_union() -> None:
    first = _observable("sum", {"spectral", "cheap"})
    second = _observable("mean", {"spectral"})
    third = _observable("max", {"cheap"})

    assert (_tag_filter("spectral") | _tag_filter("cheap")).apply(
        [first, second, third]
    ) == [first, second, third]


def test_filter_or_distinguishes_candidates_with_same_spec_id() -> None:
    right_only = _observable("sum", {"right"})
    left_only = _observable("sum", {"left"})

    assert right_only.spec.id == left_only.spec.id
    result = (_tag_filter("left") | _tag_filter("right")).apply([right_only, left_only])

    assert len(result) == 2
    assert result[0] is right_only
    assert result[1] is left_only
