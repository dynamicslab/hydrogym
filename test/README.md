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

> CI (`.github/workflows/test.yml`) runs the suite inside the published CPU image (`clagemann/hydrogym-cpu:1.0.2`) with
> the checkout mounted over the image's sources, once in the HydroGym environment and once in the Firedrake
> environment. A plain `pip`/`uv` install runs the subset whose backends are installed; the other test modules skip.
> Tests that need `mpirun -np N` (`test_nek_power_rpc.py`, `test_nek_rpc_batching.py`) are launched with
> `mpirun -np 3 python -m pytest <file>`. `test_nek5000_power_kernel.py` starts 3 ranks and allows oversubscription;
> `NEK_TEST_MPIRUN` and `NEK_TEST_MPIRUN_ARGS` override the launcher and its arguments.

## Running Tests

```bash
uv run --active --no-sync pytest test/test_pinball.py
```

or to run all tests

```bash
uv run --active --no-sync pytest test
```

> The gradient tests are currently not run, and are to be run at your own risk.
