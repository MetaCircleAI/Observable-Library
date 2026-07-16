from __future__ import annotations

from pathlib import Path

OPT_IN_TEST_DIRS = {"performance", "integration"}


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    path = Path(collection_path)
    if not OPT_IN_TEST_DIRS.intersection(path.parts):
        return False
    return not any(_is_explicit_check_arg(arg) for arg in config.args)


def _is_explicit_check_arg(arg: str) -> bool:
    path = Path(arg)
    return "tests" in path.parts and bool(OPT_IN_TEST_DIRS.intersection(path.parts))
