---
sidebar_position: 3
---

# Cylinder

:::note[Advanced workflow examples]
The scripts on this page access the Firedrake solver directly — they do not use the standard `env.reset()` / `env.step()` interface. For RL training, see [Getting Started](./getting_started).
:::

Flow past a circular cylinder is the canonical benchmark for flow instability and active flow control. At Re = 100 the steady symmetric flow bifurcates via a supercritical Hopf bifurcation, and the wake develops into a periodic von Kármán vortex street. This makes the cylinder an ideal test case for studying the transition mechanism, validating control strategies, and benchmarking RL agents.

All scripts live in [`examples/firedrake/advanced/cylinder/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/firedrake/advanced/cylinder).

## Physical setup

- **Geometry:** circular cylinder of radius 0.5 in a 2-D domain
- **Inflow:** uniform velocity U∞ = 1.0 from the left boundary
- **Reynolds number:** Re = 100 (default)

Two independent actuation mechanisms are available:

- **Jet blowing/suction** ([`Cylinder`](../../api/firedrake/envs/cylinder/flow)): two 10° tangential jets at ±90° from the stagnation point — used by `solve-steady.py`, `unsteady.py`, `step_input.py`, and `pressure-probes.py`.
- **Rotary control** ([`RotaryCylinder`](../../api/firedrake/envs/cylinder/flow)): a prescribed tangential surface velocity — used by `run-transient.py`, `pd-control.py`, `pd-phase-sweep.py`, and `lti_system.py`.

Both mechanisms are capable of suppressing vortex shedding, but they operate through different physical mechanisms and are suited to different control design approaches.

## Workflow overview

### Step 1 — Observe natural vortex shedding

Run the uncontrolled cylinder flow to generate a time-periodic reference trajectory and a restart checkpoint for subsequent steps:

```bash
python run-transient.py
```

**Source:** [`run-transient.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/run-transient.py)

The console prints the CL/CD time series; checkpoints are written for use in `pd-control.py` and related scripts.

### Step 2 — Compute the unstable steady state

At Re = 100 the symmetric, non-shedding equilibrium exists but is unstable. It can be computed with a Newton solver and used as the base flow for stability analysis and LTI modelling:

```bash
python solve-steady.py
```

**Source:** [`solve-steady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/solve-steady.py)

Outputs: `output/cylinder_Re100_steady.h5` and associated Paraview files.

### Step 3 — Observe the transition to vortex shedding

`unsteady.py` chains the Newton solver and a transient simulation into a single run, demonstrating how a small perturbation to the steady state grows into the limit cycle:

```bash
python unsteady.py
```

**Source:** [`unsteady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/unsteady.py)

- **Stage 1:** Newton solve → unstable equilibrium
- **Stage 2:** Perturbed transient integration → limit cycle

### Step 4 — Linear stability analysis

Extract the leading eigenvalue (growth rate and shedding frequency) from the linearised Navier-Stokes operator around the steady base flow:

```bash
python stability.py
```

**Source:** [`stability.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/stability.py)

The script computes both direct and adjoint global modes via Arnoldi iteration. The optional checkpoint from `solve-steady.py` can be supplied to skip the internal steady solve.

### Step 5 — Apply PD feedback control

With a vortex-shedding trajectory available, demonstrate open-loop-to-closed-loop switching with a proportional-derivative controller:

```bash
python pd-control.py
```

**Source:** [`pd-control.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/pd-control.py)

**Prerequisite:** a checkpoint produced by `run-transient.py`.

The script runs the flow uncontrolled for an initial period, then activates the PD controller and shows the oscillations decaying over time.

## Complete script reference

| Script | Purpose | Key features | Prerequisites |
|--------|---------|--------------|---------------|
| [`solve-steady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/solve-steady.py) | Compute steady-state base flow | Newton solver, Reynolds ramping | None |
| [`unsteady.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/unsteady.py) | Steady → unsteady transition | Two-stage: Newton + transient | None |
| [`run-transient.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/run-transient.py) | Basic time integration | Simple vortex-shedding demo | None |
| [`stability.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/stability.py) | Linear stability analysis | Eigenvalues, eigenmodes (direct/adjoint) | Optional steady checkpoint |
| [`step_input.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/step_input.py) | System identification | Step-response for control design | None |
| [`pressure-probes.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/pressure-probes.py) | Point measurements | Sparse sensing at pressure probes | None |
| [`pd-control.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/pd-control.py) | Feedback control | PD controller with on/off phases | Checkpoint from `run-transient.py` |
| [`pd-phase-sweep.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/pd-phase-sweep.py) | Controller tuning | Sweeps phase angles for optimal gain | Checkpoint from `run-transient.py` |
| [`lti_system.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/advanced/cylinder/lti_system.py) | LTI model extraction | Base flow + actuation influence matrix | None |

## MPI parallelisation

All scripts support MPI-parallel execution:

```bash
mpirun -np 4 python run-transient.py
mpirun -np 4 python solve-steady.py
mpirun -np 4 python stability.py
mpirun -np 4 python pd-control.py
```
