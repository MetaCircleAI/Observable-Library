from __future__ import annotations

import inspect

import observable_library as ol


def test_public_operations_have_concise_docstrings() -> None:
    operations = {
        ol.generate: ("generate", "model"),
        ol.HookSource: ("activations", "gradients"),
        ol.Runtime.observe: ("compute", "step"),
        ol.LocalStorage: ("sqlite", "npz"),
        ol.query: ("read", "id", "step"),
    }

    for public_object, terms in operations.items():
        docstring = inspect.getdoc(public_object)
        assert docstring
        assert len(docstring) <= 160
        assert all(term in docstring.lower() for term in terms)
