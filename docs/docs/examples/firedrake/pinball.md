---
sidebar_position: 4
---

# Pinball

:::note[Advanced workflow examples]
The scripts on this page access the Firedrake solver directly — they do not use the standard `env.reset()` / `env.step()` interface. For RL training, see [Getting Started](./getting_started).
:::

The fluidic pinball is a multi-body benchmark consisting of three circular cylinders arranged in an equilateral triangle. The configuration is an influential test case for multi-input flow control: each cylinder can be actuated independently via rotation, giving the controller three degrees of freedom to shape a wake that exhibits rich dynamics including mode switching, asymmetry, and chaos at moderate Reynolds numbers.

All scripts live in [`examples/firedrake/advanced/pinball/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/firedrake/advanced/pinball).

## Physical setup

- **Geometry:** three cylinders of radius 0.5 in an equilateral triangle, separated by one diameter
- **Inflow:** uniform velocity U∞ = 1.0 from the left boundary
- **Reynolds number:** Re = 30 – 150

The flow behaviour changes qualitatively across this range:

- **Re ≈ 30:** near-steady, weakly asymmetric wake
- **Re ≈ 80:** onset of periodic vortex shedding
- **Re ≈ 150:** complex unsteady wake with three-body interactions; mode switching between symmetric and asymmetric states, with chaotic dynamics at the upper end

## Workflow overview

### Step 1 — Compute the steady base flow

Solve for the steady equilibrium at Re = 80 using Newton iteration. Direct convergence from Re = 80 is difficult, so the solver ramps through intermediate Reynolds numbers:

```bash
python solve-steady.py
```

**Source:** [`solve-steady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/pinball/solve-steady.py)

Reynolds ramping schedule: 40 → 60 → 80. Outputs:

- `output/pinball_Re80_steady.h5` — restart checkpoint
- Paraview files for visualisation
- Force coefficients for all three cylinders at the steady state

### Step 2 — Run a transient simulation

Simulate the unsteady three-cylinder wake and observe the inter-body interactions:

```bash
python run-transient.py
```

**Source:** [`run-transient.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/pinball/run-transient.py)

The console prints the lift coefficient time series for each cylinder. The output file `coeffs.dat` records the full CL history for post-processing.

### Step 3 — Observe wake transition

`unsteady.py` chains an internal steady solve with a long transient integration, demonstrating the full sequence from equilibrium to complex unsteady dynamics in one run:

```bash
python unsteady.py
```

**Source:** [`unsteady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/pinball/unsteady.py)

- **Stage 1:** Newton solver with Reynolds ramping (40 → 60 → 80 → 100)
- **Stage 2:** Perturbed transient integration for T_f = 200 time units
- **Outputs:** time series, Paraview animations, force data for all three cylinders

This script has no prerequisites and provides a self-contained end-to-end demonstration.

## MPI parallelisation

All scripts support MPI-parallel execution:

```bash
mpirun -np 4 python solve-steady.py
mpirun -np 4 python run-transient.py
mpirun -np 4 python unsteady.py
```
