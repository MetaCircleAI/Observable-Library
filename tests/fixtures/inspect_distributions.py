from pathlib import Path
import sys
import tarfile
import zipfile


def inspect_distributions(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert "observable_library/py.typed" in wheel_names
    assert any(name.endswith("/LICENSE") for name in wheel_names)

    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()
    names = {name.split("/", 1)[1] for name in sdist_names if "/" in name}
    for required in [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "observable_library/py.typed",
        "docs/api.md",
        "examples/mlp_online.py",
    ]:
        assert required in names
    assert not any(name.startswith(("tests/", ".github/")) for name in names)


if __name__ == "__main__":
    _, wheel_path, sdist_path = sys.argv
    inspect_distributions(Path(wheel_path), Path(sdist_path))
