(installation)=
# Installation

Observable Library 0.1.0 is published on PyPI.

## Requirements

- Python 3.10, 3.11, or 3.12
- `numpy>=1.24`
- `torch>=2.4.1`

Python 3.13 has an advisory CI lane but is not part of the required support policy for 0.1.0.

## Install from PyPI

```bash
python -m pip install observable-library
```

Check the installed version:

```bash
python -c "import observable_library as ol; print(ol.__version__)"
```

The expected output is `0.1.0`.

## Install from a checkout

```bash
python -m pip install .
```

For development, install the test, formatting, lint, and typing tools:

```bash
python -m pip install -e ".[dev]"
```

The package and its documentation are licensed under Apache-2.0.

## Next step

Continue to the {ref}`quickstart` for a complete CPU-only example with no dataset download.
