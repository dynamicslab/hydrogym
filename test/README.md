# Testing of HydroGym

## Quick-Start

To run HydroGym's tests one best pulls a `HydroGym` [docker container](https://hub.docker.com/repository/docker/clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere/general)
(the maintained, regularly-pushed image; the older `lpaehler/hydrogym-env` images
were last updated in March 2024 and no longer match the current dependencies):

```bash
docker pull clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
```

and then launches the VSCode Devcontainer into it. At that point one has Firedrake, and
all its dependencies pre-installed. One then needs to activate the virtualenv at the
command line with

```bash
source /home/firedrake/firedrake/bin/activate
```

Install HydroGym and its locked development dependencies additively into the
existing Firedrake environment:

```bash
./scripts/bootstrap_firedrake.sh --dev
```

And is then set up to run the tests.

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
