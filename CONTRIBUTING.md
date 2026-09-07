# Contributing Guide

Thank you for contributing to HydroGym. For large changes or new solver backends, open an
[issue](https://github.com/dynamicslab/hydrogym/issues) before implementation so the design can be discussed.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the locked development environment:

```bash
git clone https://github.com/<your-username>/hydrogym.git
cd hydrogym
uv sync --locked
```

Add a solver extra when its external solver is already available:

```bash
uv sync --locked --extra maia
uv sync --locked --extra nek
uv sync --locked --extra jax
uv sync --locked --extra jaxfluids
```

Firedrake is managed by its own environment. Inside the provided Firedrake container, install the locked Python
dependencies additively without pruning Firedrake:

```bash
./scripts/bootstrap_firedrake.sh --dev
```

## Quality checks

Run the same checks used by CI:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check --diff .
uv run isort . --check-only --diff
uv run codespell --toml pyproject.toml README.md CONTRIBUTING.md docs examples test hydrogym
```

Use `uv run ruff format .` and `uv run isort .` to apply formatting locally.

## Tests and documentation

Add focused tests for new behavior and verify scientific results against a cited reference where possible. The legacy
suite under `test/` requires the Firedrake container; backend examples contain additional solver-specific smoke tests.

Build the Python distributions and documentation with:

```bash
uv build --no-sources
python scripts/verify_distribution.py dist
cd docs
npm ci
npm run build
```

## Code of Conduct

All participants must follow the [Python Community Code of Conduct](https://www.python.org/psf/conduct/).
