---
sidebar_position: 5
---

# Backward-Facing Step

:::note[Advanced workflow examples]
The scripts on this page access the Firedrake solver directly — they do not use the standard `env.reset()` / `env.step()` interface. For RL training, see [Getting Started](./getting_started).
:::

The backward-facing step is a fundamental benchmark for separated flow control. When the incoming channel flow encounters the step, it separates at the corner, forming a recirculation zone that reattaches some distance downstream. The reattachment length is sensitive to forcing, making this configuration well-suited for studying how actuation can reshape a separated shear layer.

All scripts live in [`examples/firedrake/advanced/step/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/firedrake/advanced/step).

## Physical setup

- **Geometry:** 2-D channel with a sudden downward step
- **Inflow:** uniform velocity from the left boundary
- **Reynolds number:** Re = 600 (default)
- **Observations:** kinetic energy (KE), turbulent kinetic energy (TKE), and reattachment point location

At Re = 600 the uncontrolled flow is unsteady: the separated shear layer undergoes Kelvin-Helmholtz instability and the reattachment point oscillates in time.

## Workflow overview

### Step 1 — Compute the steady base flow

The unsteady step flow has an unstable steady equilibrium that serves as the starting point for stability analysis and for initialising long transient simulations. Solve for it with Reynolds ramping to ensure convergence:

```bash
python solve-steady.py
```

**Source:** [`solve-steady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/step/solve-steady.py)

Reynolds ramping schedule: 100 → 200 → 300 → 400 → 500 → 600. Outputs:

- `output/600_steady.h5` — restart checkpoint
- `output/600_steady.pvd` — Paraview visualisation of the recirculation zone

### Step 2 — Run a long transient simulation

Starting from the perturbed steady state, advance the flow for 1 000 time units to obtain a statistically converged unsteady trajectory:

```bash
python run-transient.py
```

**Source:** [`run-transient.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/step/run-transient.py)

**Prerequisite:** `solve-steady.py` must have been run to provide the restart checkpoint.

The output file `output/stats.dat` records the CFL number, KE, and TKE at every logged timestep. This long integration is used to characterise the baseline uncontrolled flow statistics.

### Step 3 — Observe the transition to unsteadiness

`unsteady.py` combines an internal steady solve with a subsequent transient integration, demonstrating the growth of the separated-flow instability from the equilibrium in a single run:

```bash
python unsteady.py
```

**Source:** [`unsteady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/step/unsteady.py)

- **Stage 1:** Newton solver with Reynolds ramping
- **Stage 2:** Perturbed transient for T_f = 1 000 time units
- **Outputs:** time series data, Paraview animations, TKE evolution

### Step 4 — Measure the step response

Apply a constant actuation after an initial uncontrolled period and measure how the flow responds. This is a standard system-identification experiment used to calibrate linear models for control design:

```bash
python step-control.py
```

**Source:** [`step-control.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/step/step-control.py)

- **Control law:** zero actuation for t < 50, then constant blowing/suction for t ≥ 50
- **Output:** time series showing the flow's response to the step change in forcing
- **Purpose:** system identification — extract the static gain and dominant time constant

This script uses an internal initial condition and has no prerequisites.

## MPI parallelisation

All scripts support MPI-parallel execution:

```bash
mpirun -np 4 python solve-steady.py
mpirun -np 4 python run-transient.py
mpirun -np 4 python unsteady.py
mpirun -np 4 python step-control.py
```
