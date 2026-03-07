# Cylinder Flow Examples

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access, stability analysis, and specialized control. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [../../getting_started/](../../getting_started/) for `env.reset()`/`env.step()` interface.

---

Flow around a circular cylinder is a canonical benchmark in fluid mechanics and flow control.

## Physical Description

**Configuration:**
- Circular cylinder (radius = 0.5) in 2D
- Uniform inflow from left (U∞ = 1.0)
- Reynolds number Re = 100 (default)

**Actuation:**
- **Jet blowing/suction (Cylinder class):** Two 10° jets at ±90° from stagnation point
  - Used in: `solve-steady.py`, `unsteady.py`, `step_input.py`, `pressure-probes.py`
- **Rotary control (RotaryCylinder class):** Tangential velocity on cylinder surface
  - Used in: `run-transient.py`, `pd-control.py`, `pd-phase-sweep.py`, `lti_system.py`

**Note:** Both actuation types can suppress vortex shedding, but use different physical mechanisms.


## Quick Start

### 1. Basic Vortex Shedding

Run uncontrolled flow at Re=100 to observe natural vortex shedding:

```bash
python run-transient.py
```

**What it does:** Simulates uncontrolled cylinder flow showing oscillating lift/drag from vortex shedding
**Outputs:** Checkpoints for restart, console shows CL/CD time series
**Prerequisites:** None

### 2. Find Steady State

Solve for the unstable steady state at Re=100 using Newton iteration:

```bash
python solve-steady.py
```

**What it does:** Computes unstable equilibrium (saddle point) for stability analysis
**Outputs:** `output/cylinder_Re100_steady.h5`, Paraview files
**Prerequisites:** None

### 3. Observe Instability Growth

Two-stage simulation: steady solve + perturbed transient:

```bash
python unsteady.py
```

**What it does:** Demonstrates transition from steady state to limit cycle (vortex shedding)
**Outputs:** Time series, Paraview animations
**Prerequisites:** None (computes steady state internally)

### 4. Stability Analysis

Compute eigenvalues/eigenmodes using Arnoldi iteration:

```bash
python stability.py
```

**What it does:** Linear stability analysis - finds growth rates and frequencies
**Outputs:** Eigenvalues (growth rate, frequency), eigenvectors
**Prerequisites:** Optional checkpoint from solve-steady.py (or computes internally)

### 5. Apply PD Control

Suppress vortex shedding using feedback control:

```bash
python pd-control.py
```

**What it does:** Demonstrates feedback control (off→on) to stabilize unstable flow
**Outputs:** Time series showing oscillation suppression
**Prerequisites:** **REQUIRED** - Must run `run-transient.py` first for checkpoint

---

**MPI Parallelization:**
All scripts support parallel execution:
```bash
mpirun -np 4 python <script-name>.py
```

---

## Complete Script Reference

| Script | Purpose | Key Features | Prerequisites |
|--------|---------|--------------|---------------|
| **solve-steady.py** | Compute steady-state flow | Newton solver, Reynolds ramping | None |
| **unsteady.py** | Steady→unsteady transition | Two-stage: Newton + transient | None |
| **run-transient.py** | Basic time integration | Simple vortex shedding demo | None |
| **stability.py** | Linear stability analysis | Eigenvalues, eigenmodes (direct/adjoint) | Optional steady checkpoint |
| **step_input.py** | System identification | Step response for control design | None |
| **pressure-probes.py** | Point measurements | Demonstrates sparse sensing | None |
| **pd-control.py** | Feedback control | PD controller with on/off phases | **Requires** run-transient.py checkpoint |
| **pd-phase-sweep.py** | Controller tuning | Sweeps phase angles for optimal gain | **Requires** run-transient.py checkpoint |
| **lti_system.py** | Model linearization | Extracts base flow + control influence | None |


