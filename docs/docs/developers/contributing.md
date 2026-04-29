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

### 2. Install in development mode

```bash
pip install -e .
```

Install with a specific solver backend if you plan to work on environment code:

```bash
pip install -e ".[firedrake]"
pip install -e ".[maia]"
pip install -e ".[nek]"
pip install -e ".[all]"    # all optional extras
```

### 3. Install the linting tools

```bash
pip install ruff isort codespell
```

## Code style

All pull requests are checked by CI. The relevant checks are:

| Tool | What it checks | Run locally |
|------|---------------|-------------|
| `ruff check` | Lint (PEP 8, undefined names) | `ruff check .` |
| `ruff format` | Code formatting | `ruff format .` |
| `isort` | Import ordering | `isort .` |
| `codespell` | Spelling in source and docs | `codespell` |

Run all checks before opening a pull request:

```bash
ruff check .
ruff format .
isort .
codespell
```

Ruff is configured for a line length of 120 characters with double-quote strings. isort uses the `black` profile with the same line length. Both are set in `pyproject.toml` and do not need extra flags.

## Running tests

If you add a new feature, please include a test that exercises it and verify its correct working with `pytest`. Where a result can be checked against literature, add a comment citing the reference.

## Documentation changes

The documentation lives in the `docs/` subdirectory and is built with Docusaurus. To preview changes locally:

```bash
cd docs
npm install
npm start       # opens http://localhost:3000/hydrogym/
```

New pages are picked up automatically from the filesystem — place a `.md` file in the appropriate subdirectory under `docs/docs/` and it will appear in the sidebar. Add `sidebar_position: N` frontmatter to control ordering within a section.

## Pull request checklist

Before opening a PR:

- [ ] `ruff check .` passes with no errors
- [ ] `ruff format .` produces no diff
- [ ] `isort .` produces no diff
- [ ] `codespell` reports no spelling errors
- [ ] New or changed behaviour is covered by a test
- [ ] Docstrings and documentation are updated if the public API changed

## Code of Conduct

All participants are expected to follow the [Python Community Code of Conduct](https://www.python.org/psf/conduct/).
