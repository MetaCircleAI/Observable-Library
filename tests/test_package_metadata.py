from __future__ import annotations

from pathlib import Path

import observable_library as ol

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_pyproject_declares_installable_observable_library_package() -> None:
    pyproject = Path("pyproject.toml")

    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "observable-library"
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "observable_library*"
    ]
    assert metadata["project"]["requires-python"] == ">=3.10"
    assert metadata["project"]["dependencies"] == [
        "numpy>=1.24",
        "torch>=2.4.1",
    ]
    assert (
        "tomli>=2; python_version < '3.11'"
        in metadata["project"]["optional-dependencies"]["dev"]
    )
    assert not any(
        "pyarrow" in dependency or "parquet" in dependency
        for dependency in metadata["project"]["dependencies"]
    )


def test_pyproject_declares_release_metadata_and_typed_package_data() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["build-system"]["requires"] == ["setuptools>=77.0.3"]
    assert metadata["project"]["readme"] == {
        "file": "README.md",
        "content-type": "text/markdown",
    }
    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert metadata["project"]["authors"] == [{"name": "Jinxin"}]
    assert metadata["tool"]["setuptools"]["package-data"] == {
        "observable_library": ["py.typed"]
    }
    assert Path("LICENSE").is_file()
    assert Path("observable_library/py.typed").is_file()


def test_pyproject_declares_canonical_project_urls() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["urls"] == {
        "Homepage": "https://github.com/MetaCircleAI/Observable-Library",
        "Documentation": "https://metacircleai.github.io/Observable-Library/",
        "Repository": "https://github.com/MetaCircleAI/Observable-Library",
        "Issues": "https://github.com/MetaCircleAI/Observable-Library/issues",
    }


def test_manifest_includes_release_sources_and_excludes_checkout_only_files() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    for line in [
        "include LICENSE",
        "include README.md",
        "include pyproject.toml",
        "recursive-include observable_library *.py py.typed",
        "recursive-include docs *.md",
        "recursive-include examples *.py",
        "prune tests",
        "prune .github",
    ]:
        assert line in manifest


def test_readme_matches_supported_runtime_policy() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in [
        "Python 3.10, 3.11, and 3.12",
        "Python 3.13",
        "advisory",
        "numpy>=1.24",
        "torch>=2.4.1",
    ]:
        assert text in readme


def test_readme_documents_index_and_checkout_install_without_relative_links() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m pip install observable-library" in readme
    assert 'python -m pip install -e ".[dev]"' in readme
    assert "Apache-2.0" in readme
    assert "Jinxin" in readme
    assert "[API reference](docs/api.md)" not in readme


def test_top_level_package_exports_version_and_core_api() -> None:
    assert isinstance(ol.__version__, str)
    assert ol.ObservableSpec.__name__ == "ObservableSpec"
    assert ol.Observable.__name__ == "Observable"
    assert ol.Runtime.__name__ == "Runtime"
    assert ol.TypedTensor.__name__ == "TypedTensor"
