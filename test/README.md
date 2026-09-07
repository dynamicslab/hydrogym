# Testing of HydroGym

## Quick-Start

To run HydroGym's tests one best pulls the `HydroGym-Env` [docker container](https://hub.docker.com/repository/docker/lpaehler/hydrogym-env/general):

```bash
docker pull lpaehler/hydrogym-env:stable
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

## Running Tests

```bash
uv run --active --no-sync pytest test/test_pinball.py
```

or to run all tests

```bash
uv run --active --no-sync pytest test
```

> The gradient tests are currently not run, and are to be run at your own risk.
