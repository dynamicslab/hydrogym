# Pinball Flow Examples

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access and specialized workflows. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [../../getting_started/](../../getting_started/) for `env.reset()`/`env.step()` interface.

---

Flow around three cylinders in triangular arrangement - a challenging benchmark for flow control.

## Physical Description

**Configuration:**
- Three cylinders in equilateral triangle arrangement
- Cylinder radius: 0.5
- Uniform inflow from left (U∞ = 1.0)
- Reynolds number Re = 30-150

**Key Phenomena:**
- **Re = 30:** Steady symmetric flow
- **Re = 150:** Complex unsteady wake with three-body interactions
- Wake can exhibit mode switching between symmetric/asymmetric states
- Chaotic dynamics possible at higher Re

## Quick Start

### 1. Basic Simulation

Run unsteady pinball flow at Re=30:

```bash
python run-transient.py
```

**What it does:** Simulates flow around three cylinders showing wake interactions
**Outputs:**
- `coeffs.dat` - Time series of CL for all three cylinders
- Console shows forces on each cylinder
**Prerequisites:** None

### 2. Find Steady State

Solve for steady flow at Re=80 using Newton iteration:

```bash
python solve-steady.py
```

**What it does:** Computes steady state (or unstable equilibrium) for stability analysis
**Uses ramping:** 40 → 60 → 80 for better convergence
**Outputs:**
- `output/pinball_Re80_steady.h5` - Checkpoint for restart
- Paraview files for visualization
- Force coefficients for all three cylinders
**Prerequisites:** None

### 3. Observe Wake Dynamics

Two-stage simulation: steady solve + perturbed transient:

```bash
python unsteady.py
```

**What it does:** Demonstrates transition from steady state to complex wake dynamics
**Stage 1:** Solve steady state with Reynolds ramping (40 → 60 → 80 → 100)
**Stage 2:** Add perturbation and run transient (Tf=200)
**Outputs:** Time series, Paraview animations, force data for all cylinders
**Prerequisites:** None (computes steady state internally)

---

**MPI Parallelization:**
All scripts support parallel execution:
```bash
mpirun -np 4 python <script-name>.py
```

