# Backward-Facing Step Flow Examples

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access and specialized workflows. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [../../getting_started/](../../getting_started/) for `env.reset()`/`env.step()` interface.

---

Flow over a backward-facing step demonstrates separated flow and reattachment.

## Physical Description

**Configuration:**
- Channel with sudden expansion (backward-facing step)
- Step height creates separation zone
- Uniform inflow from left
- Reynolds number Re = 600 (default)

**Observations:**
- Kinetic energy (KE)
- Turbulent kinetic energy (TKE)
- Reattachment point location

## Quick Start

### 1. Basic Simulation

Run transient step flow at Re=600:

```bash
python run-transient.py
```

**What it does:** Simulates separated flow from perturbed base state
**Outputs:**
- `output/stats.dat` - Time series of CFL, KE, TKE
- Console shows KE, TKE evolution
- Long time integration (1000 time units)
**Prerequisites:** Requires steady state checkpoint from solve-steady.py

### 2. Find Steady State

Solve for steady flow using Newton iteration with Reynolds ramping:

```bash
python solve-steady.py
```

**What it does:** Computes steady base flow for separated step flow
**Uses ramping:** 100 → 200 → 300 → 400 → 500 → 600 for convergence
**Outputs:**
- `output/600_steady.h5` - Steady checkpoint for restart
- `output/600_steady.pvd` - Paraview visualization of recirculation zone
**Prerequisites:** None

### 3. Observe Instability

Two-stage simulation: steady solve + perturbed transient:

```bash
python unsteady.py
```

**What it does:** Demonstrates transition from steady to unsteady separated flow
**Stage 1:** Solve steady state with Reynolds ramping
**Stage 2:** Add perturbation and run long transient (Tf=1000)
**Outputs:** Time series, Paraview animations, TKE evolution
**Prerequisites:** None (computes steady state internally)

### 4. Test Control Response

Apply step input actuation:

```bash
python step-control.py
```

**What it does:** Applies step change in actuation to measure flow response
**Control:** Off until t=50, then constant actuation
**Purpose:** System identification - measure step response
**Outputs:** Time series showing response to actuation
**Prerequisites:** None (uses internal initial condition)

---

**MPI Parallelization:**
All scripts support parallel execution:
```bash
mpirun -np 4 python <script-name>.py
```


