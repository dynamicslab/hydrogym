---
sidebar_position: 2
---

# Cavity

:::note[Advanced workflow examples]
The scripts on this page access the Firedrake solver directly — they do not use the standard `env.reset()` / `env.step()` interface. For RL training, see [Getting Started](./getting_started).
:::

The open cavity is a classic benchmark in computational fluid dynamics. Driven by the no-slip condition on the moving top wall, the flow develops a primary recirculation zone inside the cavity and exhibits strong shear-layer instability at high Reynolds numbers. The [`Cavity`](../../api/firedrake/envs/cavity/flow) environment in HydroGym uses this configuration to study turbulence transition and actuation-driven reattachment.

All scripts live in [`examples/firedrake/advanced/cavity/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/firedrake/advanced/cavity).

## Physical setup

- **Geometry:** open square cavity (1 × 1) with a moving top wall
- **Boundary conditions:** uniform lid velocity U = 1.0 on the top wall; no-slip on all other walls
- **Reynolds number:** Re = 7 500

## Workflow overview

The scripts are designed to be run in sequence, with each step building on the outputs of the previous one.

### Step 1 — Compute a steady base flow

Before running any transient simulation, solve for the steady base flow using a Newton iteration with Reynolds ramping. This is necessary because the high-Re cavity flow has a poorly conditioned Jacobian that prevents direct convergence from Re = 7 500.

```bash
python solve-steady.py
```

**Source:** [`solve-steady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cavity/solve-steady.py)

Reynolds ramping schedule: 500 → 1 000 → 2 000 → 4 000 → 7 500. Outputs:

- `output/7500_steady.h5` — checkpoint for use as a restart file
- `output/7500_steady.pvd` — Paraview visualisation of the base flow

This step has no prerequisites and can be run independently.

### Step 2 — Run a transient simulation

With the steady-state checkpoint in hand, perturb the base flow and advance it in time to observe the development of shear-layer instability and the approach to a statistically stationary turbulent state.

```bash
python run-transient.py
```

**Source:** [`run-transient.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cavity/run-transient.py)

**Prerequisite:** `solve-steady.py` must have been run to produce the restart checkpoint.

Outputs written to `output/stats.dat`:
- CFL number
- Kinetic energy (KE)
- Turbulent kinetic energy (TKE)

### Step 3 — Linear stability analysis

With the steady state computed, the linearised Navier-Stokes operator can be analysed to identify the unstable global modes that trigger shear-layer instability.

```bash
python stability.py --Re 7500 --num-eigs 10
```

**Source:** [`stability.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cavity/stability.py)

The script uses Arnoldi iteration to extract the leading eigenvalues (growth rate and frequency) and their associated eigenvectors. The steady-state checkpoint is optional — if omitted, the script computes its own base flow internally.

### Step 4 — End-to-end unsteady demonstration

The `unsteady.py` script reproduces the full transition scenario in a single run: it first solves for the steady state internally using Reynolds ramping, then adds a small perturbation and continues the time integration until a long-time statistically turbulent state is reached.

```bash
python unsteady.py
```

**Source:** [`unsteady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cavity/unsteady.py)

- **Stage 1:** Newton solver with Reynolds ramping (500 → 7 500)
- **Stage 2:** Perturbed transient integration for T_f = 500 time units
- **Outputs:** time series data, Paraview animations, and TKE evolution

This script has no prerequisites and provides a self-contained demonstration of the entire workflow.

## MPI parallelisation

All four scripts support MPI-parallel execution via Firedrake's PETSc backend. Pass the desired process count to `mpirun`:

```bash
mpirun -np 4 python solve-steady.py
mpirun -np 4 python run-transient.py
mpirun -np 4 python stability.py --Re 7500 --num-eigs 10
mpirun -np 4 python unsteady.py
```
