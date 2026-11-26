# Open Cavity Flow Examples

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access, stability analysis, and specialized workflows. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [../../getting_started/](../../getting_started/) for `env.reset()`/`env.step()` interface.

---

The open cavity is a classic CFD benchmark problem demonstrating recirculating flows and shear-layer instability.

## Physical Description

**Configuration:**
- Open cavity (length = depth = 1) below a boundary layer (Barbagallo et al. 2009)
- Inlet velocity: U = 1.0 on the left, free-slip top boundary
- Walls of the plate and the cavity: no-slip (U = 0); a blowing slot at the leading edge of the cavity
- Reynolds number Re = 7500, based on the cavity length

## Quick Start

Run `solve-steady.py` (section 2) before `run-transient.py`, which starts from its checkpoint (`output/*_steady.h5`).
`unsteady.py` computes the steady state itself; its checkpoint is the restart state of `sine-forcing.py`.

### 1. Flow Simulation

Run open cavity at Re=7500:

```bash
python run-transient.py
```

**What it does:** Simulates the unsteady cavity flow from the perturbed base state
**Outputs:**
- `output/stats.dat` - Time series of CFL, KE, TKE
- Console shows the evolution of the kinetic energy and of the fluctuation kinetic energy (TKE)
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
python stability.py --reynolds 7500
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
**Outputs:** Time series (`stats.dat`, with the TKE evolution) and `checkpoint.h5` in `7500_medium_output/`, the
restart state of `sine-forcing.py` (`python sine-forcing.py`: open-loop actuation u(t) = sin(t))
**Prerequisites:** None (computes steady state internally)

---

**MPI Parallelization:**
All scripts support parallel execution:
```bash
mpirun -np 4 python <script-name>.py
```


