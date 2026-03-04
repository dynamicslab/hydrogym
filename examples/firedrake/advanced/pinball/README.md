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
- Reynolds number Re = 30-100

**Key Phenomena:**
- **Re = 30:** Steady symmetric flow
- **Re = 100:** Complex unsteady wake with three-body interactions
- Wake can exhibit mode switching between symmetric/asymmetric states
- Chaotic dynamics possible at higher Re

**Observations:**
- Lift coefficient on each of the three cylinders: CL[0], CL[1], CL[2]
- Complex force interactions due to wake interference

## Examples Overview

| Script | Type | Purpose | Runtime |
|--------|------|---------|---------|
| **run-transient.py** | Simulation | Basic three-cylinder flow | ~5 min |
| **solve-steady.py** | Solver | Find steady/unstable equilibrium | ~3 min |
| **unsteady.py** | Workflow | Steady → complex wake transition | ~10 min |

## Quick Start

### 1. Basic Simulation

Run unsteady pinball flow:

```bash
python run-transient.py
```

**Outputs:**
- `coeffs.dat` - Time series of CL for all three cylinders
- Console shows forces on each cylinder

### 2. Find Steady State

Solve for steady flow at Re=80:

```bash
python solve-steady.py
```

**Uses ramping:** 40 → 60 → 80

**Outputs:**
- `output/pinball_Re80_steady.h5` - Checkpoint
- Force coefficients for all three cylinders

### 3. Observe Wake Dynamics

Start from steady state, watch complex wake develop:

```bash
python unsteady.py
```

**Stage 1:** Solve steady state with Reynolds ramping
**Stage 2:** Add perturbation and run transient
**Result:** Complex three-body wake interactions

## Detailed Example Descriptions

### run-transient.py
- Simulates unsteady flow at Re=30 (default)
- Monitors lift on all three cylinders
- Shows wake interactions and force coupling
- Uses BDF time-stepping with GLS stabilization

### solve-steady.py
- Newton solver with Reynolds ramping (40 → 60 → 80)
- Finds steady or unstable equilibrium state
- Useful as base flow for perturbation studies
- Saves checkpoint and visualization

### unsteady.py
- Complete workflow: steady solve → transient with perturbation
- Demonstrates transition from steady to complex dynamics
- Long time integration (Tf=200) to capture wake evolution
- Outputs force time series for all cylinders

## Physical Insights

**Wake Modes:**
- **Symmetric mode:** Equal forces on all cylinders
- **Asymmetric mode:** Wake favors one side
- **Mode switching:** Flow can switch between modes
- **Chaotic regime:** Complex, aperiodic dynamics at high Re

**Three-Body Interactions:**
- Cylinder wakes interfere with each other
- Force coupling between cylinders
- Rich dynamics even at moderate Reynolds numbers

## Usage Tips

**Reynolds number:**
- Re < 50: Mostly steady, good for testing
- Re = 100: Complex unsteady dynamics (challenging)
- Use lower Re for initial testing

**Mesh resolution:**
- Pinball geometry is complex → needs fine mesh
- Default `medium` mesh okay for Re < 50
- Use `fine` mesh for Re > 50

**Integration time:**
- Need long integration (Tf > 100) to see mode switching
- Complex dynamics take time to develop

**Observations:**
- Watch all three CL values - they're coupled
- Look for symmetry breaking in force time series

## Future Examples (TODO)

- [ ] **stability.py** - Eigenvalue analysis of three-body wake
- [ ] **control-example.py** - Independent control of each cylinder
- [ ] **validation.py** - Compare against literature benchmarks

## References

1. Deng, N. et al. (2020). Low-order model for successive bifurcations of the fluidic pinball. *J. Fluid Mech.*, 884, A37.
2. Cornejo Maceda, G.Y. et al. (2021). Stabilization of the fluidic pinball with gradient-enriched machine learning control. *J. Fluid Mech.*, 917, A42.

## Next Steps

- Vary Reynolds number to explore different regimes
- Implement control for wake stabilization
- Compare with published pinball control studies
- Add stability analysis to find unstable modes
