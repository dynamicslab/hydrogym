# Lid-Driven Cavity Flow Examples

⚠️ **NOTE**: These are **advanced workflow examples** showing direct solver access, stability analysis, and specialized workflows. They do NOT use the standard RL interface.

**Looking for standard RL examples?** See [../../getting_started/](../../getting_started/) for `env.reset()`/`env.step()` interface.

---

The lid-driven cavity is a classic CFD benchmark problem demonstrating recirculating flows and turbulence.

## Physical Description

**Configuration:**
- Square cavity (1×1) with moving top wall
- Top wall velocity: U = 1.0 (left to right)
- All other walls: no-slip (U = 0)
- Reynolds number Re = 1000-7500

**Key Phenomena:**
- **Re < 5000:** Steady recirculating flow with corner vortices
- **Re > 5000:** Turbulent flow with fluctuating kinetic energy
- **Re = 7500:** Benchmark case for turbulent cavity flow

## Examples Overview

| Script | Type | Purpose | Runtime |
|--------|------|---------|---------|
| **run-transient.py** | Simulation | Turbulent flow evolution | ~3 min |
| **solve-steady.py** | Solver | Steady state at high Re | ~2 min |
| **unsteady.py** | Workflow | Steady → turbulent transition | ~5 min |
| **stability.py** | Analysis | Eigenvalue/stability analysis | ~10 min |
| **sine-forcing.py** | Control | Sinusoidal forcing demo | ~3 min |

## Quick Start

### 1. Turbulent Flow Simulation

Run turbulent cavity at Re=7500:

```bash
python run-transient.py
```

**Outputs:**
- Console shows evolution of kinetic energy (KE) and turbulent kinetic energy (TKE)
- Flow develops from random initial condition

### 2. Find Steady State

Solve for steady flow using Reynolds ramping:

```bash
python solve-steady.py
```

**Uses ramping:** 500 → 1000 → 2000 → 4000 → 7500

**Outputs:**
- `output/7500_steady.h5` - Steady flow checkpoint
- `output/7500_steady.pvd` - Paraview visualization

### 3. Stability Analysis

Compute eigenvalues of steady flow:

```bash
python stability.py --Re 5000 --num-eigs 10
```

**Purpose:** Identify unstable modes leading to turbulence

### 4. Complete Workflow

Run full steady → unsteady transition:

```bash
python unsteady.py
```

**Stage 1:** Solve steady state with ramping
**Stage 2:** Add perturbation and watch turbulence develop

## Detailed Example Descriptions

### run-transient.py
- Starts from random initial condition
- Evolves turbulent flow at Re=7500
- Monitors kinetic energy (KE) and turbulent kinetic energy (TKE)
- Uses BDF time-stepping with GLS stabilization

### solve-steady.py
- Newton solver with Reynolds ramping for convergence
- High-Re flows need gradual ramping (500 → 7500)
- Saves checkpoint for restart or perturbation studies

### unsteady.py
- Two-stage workflow: (1) steady solve, (2) transient with perturbation
- Demonstrates transition to turbulence
- Compares steady vs unsteady kinetic energy

### stability.py
- Linear stability analysis using Arnoldi iteration
- Computes eigenvalues and eigenmodes
- Command-line arguments for Re, number of eigenvalues, shift
- Includes adjoint mode computation

### sine-forcing.py
- Demonstrates sinusoidal forcing on top wall
- Uses FlowEnv wrapper (different pattern from others)
- Shows time-varying boundary conditions

## Physical Validation

Benchmark data from Ghia et al. (1982):

| Re | u-velocity at cavity centerline | v-velocity at cavity centerline |
|----|----------------------------------|----------------------------------|
| 1000 | Validated ✓ | Validated ✓ |
| 5000 | See validation.ipynb | See validation.ipynb |

See `validation.ipynb` for detailed comparison plots.

## Usage Tips

**Reynolds number:**
- Re = 1000: Steady flow, good for validation
- Re = 5000: Transitional regime
- Re = 7500: Fully turbulent (default)

**Mesh resolution:**
- For Re=7500, use `fine` mesh (80k+ elements)
- For Re=1000, `medium` mesh sufficient

**Time step:**
- High Re needs small dt: 1e-4 to 5e-4
- CFL should stay < 1 for stability

## References

1. Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using Navier-Stokes equations. *J. Comp. Phys.*, 48, 387-411.
2. See `validation.ipynb` for detailed comparisons

## Next Steps

- Validate against Ghia et al. using `validation.ipynb`
- Try different Reynolds numbers
- Apply control/forcing with `sine-forcing.py`
- Perform stability analysis at transitional Re
