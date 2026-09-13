# Project Dependencies

Basic dependencies for an MLOps demonstration project. Add these to `pyproject.toml`
(or `requirements.txt`) as needed.

## Core

| Package | Purpose |
| --- | --- |
| `numpy` | Arrays and numerical operations |
| `pandas` | Data loading and dataframes |
| `scikit-learn` | Model training and evaluation |
| `joblib` | Model serialization (save/load) |
| `matplotlib` | Plotting and visualisation |
| `mlflow` | Experiment tracking and model registry |
| `fastapi` | Model serving API |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Data validation and settings management |
| `requests` | HTTP client for API calls |
| `evidently` | Data and model drift monitoring |

## Development

| Package | Purpose |
| --- | --- |
| `pytest` | Unit and integration testing |
| `pytest-cov` | Test coverage reporting |
| `httpx` | HTTP client for testing FastAPI endpoints |
| `ruff` | Linting and formatting |

## Example `pyproject.toml`

```toml
[project]
name = "poetry-demo"
version = "0.1.0"
description = "Demo MLOps project"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "joblib>=1.4",
    "matplotlib>=3.9",
    "mlflow>=2.15",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "requests>=2.32",
    "evidently>=0.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "httpx>=0.27",
    "ruff>=0.6",
]
```

## What is a wheel?

A **wheel** is a ready-to-install Python package. It is a single `.whl` file that
already contains everything the package needs — the code, plus a small note saying
which other packages it depends on.

Think of it like this: a source package is a flat-pack furniture kit you have to
assemble yourself, while a wheel is the furniture that arrives already built. You
just place it in the room.

Why this matters:

- **Installs are fast.** `pip` can just unpack the wheel into place. There is no
  building or assembling step.
- **No extra tools needed.** Some packages (like `numpy` and `scikit-learn`) contain
  code written in C for speed, which normally needs a compiler to build. A wheel
  ships that code already built, so you do not need a compiler on your machine.
- **Wheels are made for a specific setup.** The filename says which Python version,
  operating system, and CPU the wheel is built for. For example,
  `numpy-1.26.4-cp311-cp311-macosx_11_0_arm64.whl` is for Python 3.11 (`cp311`) on an
  Apple Silicon Mac (`macosx_11_0_arm64`). Your setup must match.
- **When there is no matching wheel**, `pip` downloads the raw source code instead
  (a `.tar.gz` file) and tries to build it on your machine. That is slower and can
  fail if the right tools are missing. This is why brand-new Python versions cannot
  always install a package right away — the wheels for them have not been published
  yet.

Wheel is the modern packaging standard defined by [PEP 427](https://peps.python.org/pep-0427/).

## Notes

- Use `requires-python = ">=3.11"` for the demo. Many of these packages do not yet
  ship wheels for Python 3.14, so `poetry add` will fail on a 3.14-only constraint.
- With Poetry, add dependencies via `poetry add <package>` (and `poetry add --group dev <package>`).
- Poetry-native dev dependencies can alternatively go under
  `[tool.poetry.group.dev.dependencies]` instead of `[project.optional-dependencies]`.



## Commands to run poetry

```poetry install``` to install the dependencies

Can run specific scripts specifying the dependencies ```poetry run python your_script.py```

```poetry.lock``` file will be created when you run poetry install.

This locks in the dependencies and can be committed to Git. This ensures that everyone is working with the same version.