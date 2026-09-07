---
sidebar_position: 1
---

# Contributing

Thank you for your interest in contributing to HydroGym. Contributions of all kinds are welcome — bug fixes, new environments, documentation improvements, and new solver backends.

If you have a question or want to discuss a larger change before writing code, open an [issue](https://github.com/dynamicslab/hydrogym/issues) first.

:::note
HydroGym is research software under active development. Breaking changes can occur, but we do our very best to give ample notice beforehand.
:::

## Setting up a development environment

### 1. Fork and clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/hydrogym.git
cd hydrogym
```

### 2. Create the locked development environment

```bash
uv sync --locked
```

Install with a specific solver backend if you plan to work on environment code:

```bash
uv sync --locked --extra maia
uv sync --locked --extra nek
uv sync --locked --extra jax
uv sync --locked --extra jaxfluids
```

Firedrake owns its own Python environment. In the provided Firedrake container, use:

```bash
./scripts/bootstrap_firedrake.sh --dev
```

## Code style

All pull requests are checked by CI. The relevant checks are:

| Tool | What it checks | Run locally |
|------|---------------|-------------|
| `ruff check` | Lint (PEP 8, undefined names) | `uv run ruff check .` |
| `ruff format` | Code formatting | `uv run ruff format .` |
| `isort` | Import ordering | `uv run isort .` |
| `codespell` | Spelling in source and docs | `uv run codespell` |

Run all checks before opening a pull request:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check --diff .
uv run isort . --check-only --diff
uv run codespell --toml pyproject.toml README.md CONTRIBUTING.md docs examples test hydrogym
```

Ruff is configured for a line length of 120 characters with double-quote strings. isort uses the `black` profile with the same line length. Both are set in `pyproject.toml` and do not need extra flags.

## Running tests

If you add a new feature, please include a test that exercises it and verify its correct working with `pytest`. Where a result can be checked against literature, add a comment citing the reference.

## Documentation changes

The documentation lives in the `docs/` subdirectory and is built with Docusaurus. To preview changes locally:

```bash
cd docs
npm ci
npm start       # opens http://localhost:3000/hydrogym/
```

New pages are picked up automatically from the filesystem — place a `.md` file in the appropriate subdirectory under `docs/docs/` and it will appear in the sidebar. Add `sidebar_position: N` frontmatter to control ordering within a section.

## Pull request checklist

Before opening a PR:

- [ ] `uv lock --check` passes
- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format --check --diff .` produces no diff
- [ ] `uv run isort . --check-only --diff` produces no diff
- [ ] `uv run codespell ...` reports no spelling errors
- [ ] New or changed behaviour is covered by a test
- [ ] Docstrings and documentation are updated if the public API changed

## Code of Conduct

All participants are expected to follow the [Python Community Code of Conduct](https://www.python.org/psf/conduct/).
