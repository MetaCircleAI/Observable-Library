from __future__ import annotations

from collections.abc import Iterable

from observable_library.generator.matcher import IdentityTemplate


def resolve_templates(templates: Iterable[IdentityTemplate]) -> list[IdentityTemplate]:
    return list(templates)
