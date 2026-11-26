# Testing of HydroGym

## Quick-Start

To run HydroGym's tests one best uses a HydroGym release image, which has
Firedrake and all its dependencies pre-installed (the CPU image is enough for
the Firedrake tests):

```bash
docker pull clagemann/hydrogym-cpu:1.0.2
docker run -it --ipc=host -v $(pwd):/workspace -w /workspace clagemann/hydrogym-cpu:1.0.2
```

(or open this checkout in `.devcontainer/release-cpu.devcontainer.json`). Then
activate the Firedrake environment:

```bash
source /opt/venvs/firedrake/bin/activate
```

Install HydroGym and its locked development dependencies additively into the
existing Firedrake environment:

```bash
./scripts/bootstrap_firedrake.sh --dev
```

The environment is then ready to run the tests.

> CI does not use a pre-built image: `.github/workflows/test.yml` builds the
> repo's own `.devcontainer/firedrake-test.devcontainer.json` (petsc +
> firedrake features) on every run, so that definition — not any Docker Hub
> image — is the authoritative test environment.

## Running Tests

```bash
uv run --active --no-sync pytest test/test_pinball.py
```

or to run all tests

```bash
uv run --active --no-sync pytest test
```

> The gradient tests are currently not run, and are to be run at your own risk.
