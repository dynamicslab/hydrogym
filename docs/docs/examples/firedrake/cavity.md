---
sidebar_position: 2
---

# Cavity

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access, stability analysis, and specialized workflows. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [Getting Started](./getting_started) for `env.reset()` / `env.step()` interface.

---

The open cavity is a classic CFD benchmark problem demonstrating recirculating flows and shear-layer instability.

## Physical Description

**Configuration:**

- Open square cavity (1×1) with moving top wall
- Inlet velocity: U = 1.0 
- All other walls: no-slip (U = 0)
- Reynolds number Re = 7500

## Quick Start

### 1. Flow Simulation

Run open cavity at Re=7500:

```bash
python run-transient.py
```

**What it does:** Simulates turbulent cavity flow from perturbed base state
**Outputs:**

- `output/stats.dat` - Time series of CFL, KE, TKE
- Console shows evolution of kinetic energy and turbulent kinetic energy

**Prerequisites:** Requires steady state checkpoint from solve-steady.py

### 2. Find Steady State

Solve for steady flow using Newton iteration with Reynolds ramping:

```bash
python solve-steady.py
```

**What it does:** Computes steady base flow for high-Re cavity
**Uses ramping:** 500 → 1000 → 2000 → 4000 → 7500 for convergence
**Outputs:**

- `output/7500_steady.h5` - Steady flow checkpoint for restart
- `output/7500_steady.pvd` - Paraview visualization
**Prerequisites:** None

### 3. Stability Analysis

Compute eigenvalues of steady flow:

```bash
python stability.py --Re 7500 --num-eigs 10
```

**What it does:** Linear stability analysis using Arnoldi iteration
**Purpose:** Identify unstable modes leading to shear-layer instability
**Outputs:** Eigenvalues, eigenvectors, growth rates
**Prerequisites:** Optional (can compute steady state internally)

### 4. Complete Workflow

Two-stage simulation: steady solve + perturbed transient:

```bash
python unsteady.py
```

**What it does:** Demonstrates transition from steady to unstable flow
**Stage 1:** Solve steady state with Reynolds ramping (500 → ... → 7500)
**Stage 2:** Add perturbation and run long transient (Tf=500)
**Outputs:** Time series, Paraview animations, TKE evolution
**Prerequisites:** None (computes steady state internally)

---

**MPI Parallelization:**
All scripts support parallel execution:

```bash
mpirun -np 4 python <script-name>.py
```

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
