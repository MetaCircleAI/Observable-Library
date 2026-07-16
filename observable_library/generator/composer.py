from __future__ import annotations

from collections.abc import Iterable

from observable_library.filter.identity import identity_observables
from observable_library.generator.matcher import IdentityTemplate


def compose_observables(
    resolved_templates: Iterable[IdentityTemplate],
    transforms: Iterable[str] = (),
):
    grouped: dict[str, dict[str, object]] = {}
    for template in resolved_templates:
        key = f"{template['source']}::{template['selector']}"
        group = grouped.setdefault(
            key,
            {
                "metadata": {
                    "source": template["source"],
                    "selector": template["selector"],
                    "shape": template["shape"],
                    "axes": template["axes"],
                },
                "reductions": [],
            },
        )
        reductions = group["reductions"]
        assert isinstance(reductions, list)
        reductions.append(template["reduction"])

    observables = []
    for group in grouped.values():
        metadata = group["metadata"]
        reductions = group["reductions"]
        assert isinstance(metadata, dict)
        assert isinstance(reductions, list)
        observables.extend(
            identity_observables([metadata], reductions, transforms=transforms)
        )
    return observables
