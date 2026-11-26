# Contributing Guide

Thank you for contributing to HydroGym. For large changes or new solver backends, open an
[issue](https://github.com/dynamicslab/hydrogym/issues) before implementation so the design can be discussed.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the locked development environment:

```bash
git lfs install          # once per machine: meshes and case files are Git LFS files
git clone https://github.com/<your-username>/hydrogym.git
cd hydrogym
uv sync --locked
```

> Git LFS: the Firedrake meshes (`hydrogym/firedrake/envs/*/*.msh`) and the Nek5000 case files (`third_party/nek5000/cases/`) are stored with [Git LFS](https://git-lfs.com). Run `git lfs install` before cloning, or `git lfs pull` in an existing clone; otherwise these files are small pointer files and HydroGym stops with "... is a Git LFS pointer". The wheels on PyPI contain the meshes.

Add a solver extra when its external solver is already available:

```bash
uv sync --locked --extra maia
uv sync --locked --extra nek
uv sync --locked --extra jax
uv sync --locked --extra jaxfluids
uv sync --locked --extra nekrs
```

Firedrake is managed by its own environment. Inside a Firedrake container (the HydroGym CPU image
`clagemann/hydrogym-cpu:1.0.2` after `source /opt/venvs/firedrake/bin/activate`, or
`.devcontainer/firedrake-test.devcontainer.json`), install the locked Python dependencies additively without pruning
Firedrake:

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

Add focused tests for new behavior and verify scientific results against a cited reference where possible. The tests
under `test/` that need Firedrake run in a Firedrake container (see `test/README.md`), the others with the matching
extras; backend examples contain additional solver-specific smoke tests.

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
