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

**Key Phenomena:**
- **Low Re:** Steady recirculation zone behind step
- **Re ~ 600:** Transitional flow, periodic vortex shedding
- **High Re:** Turbulent separation and reattachment
- Reattachment length is key metric

**Observations:**
- Kinetic energy (KE)
- Turbulent kinetic energy (TKE)
- Reattachment point location

## Examples Overview

| Script | Type | Purpose | Runtime |
|--------|------|---------|---------|
| **run-transient.py** | Simulation | Basic step flow simulation | ~5 min |
| **solve-steady.py** | Solver | Find steady/unstable state | ~2 min |
| **unsteady.py** | Workflow | Steady → unsteady transition | ~8 min |
| **step-control.py** | Control | Step input control test | ~5 min |

## Quick Start

### 1. Basic Simulation

Run transient step flow at Re=600:

```bash
python run-transient.py
```

**Outputs:**
- Console shows KE, TKE evolution
- Long time integration (1000 time units)

### 2. Find Steady State

Solve for steady flow with ramping:

```bash
python solve-steady.py
```

**Uses ramping:** 100 → 300 → 500 → 600

**Outputs:**
- `output/600_steady.h5` - Steady checkpoint
- Visualization of recirculation zone

### 3. Observe Instability

Start from steady, watch flow destabilize:

```bash
python unsteady.py
```

**Stage 1:** Solve steady with ramping
**Stage 2:** Add perturbation, run transient
**Result:** Vortex shedding in wake

### 4. Test Control Response

Apply step input actuation:

```bash
python step-control.py
```

**Control:** Off until t=50, then constant actuation
**Purpose:** Measure system step response

## Detailed Example Descriptions

### run-transient.py
- Long time integration (1000 time units)
- Monitors kinetic energy and turbulence
- Shows development from initial condition
- High CFL monitoring for stability

### solve-steady.py
- Newton solver with Reynolds ramping (100 → 600)
- Finds steady separation zone
- At Re=600, may find unstable equilibrium
- Good initial condition for unsteady simulations

### unsteady.py
- Two-stage workflow: steady → transient
- Demonstrates transition to periodic shedding
- Compares steady vs unsteady TKE
- Long integration to see flow development

### step-control.py
- Open-loop step input control
- Control switches on at t=50
- Demonstrates actuation response
- Useful for system identification

## Physical Insights

**Separation Zone:**
- Flow separates at step corner
- Recirculation region behind step
- Reattachment point moves with Re

**Reattachment Length:**
- Key metric: distance from step to reattachment
- Increases with Reynolds number
- Target for control: reduce reattachment length

**Vortex Shedding:**
- At Re ~ 600, wake becomes unstable
- Periodic vortex shedding from separation
- TKE indicates unsteady fluctuations

## Usage Tips

**Reynolds number:**
- Re < 400: Steady flow, good for testing
- Re = 600: Transitional, periodic shedding
- Re > 800: Turbulent regime

**Mesh resolution:**
- Step geometry needs refinement at corner
- Use `medium` or `fine` mesh
- Check CFL number stays < 1

**Time step:**
- Default dt = 0.01 usually okay
- For high Re, may need dt < 0.005
- Monitor CFL number in output

**Long time integration:**
- Step flow needs long time to develop (Tf > 500)
- Statistics require time-averaging over many cycles

## Future Examples (TODO)

- [ ] **stability.py** - Eigenvalue analysis of separated flow
- [ ] **pd-control.py** - Feedback control to reduce separation

## References

1. Armaly, B.F. et al. (1983). Experimental and theoretical investigation of backward-facing step flow. *J. Fluid Mech.*, 127, 473-496.
2. Gartling, D.K. (1990). A test problem for outflow boundary conditions—flow over a backward-facing step. *Int. J. Num. Meth. Fluids*, 11, 953-967.

## Next Steps

- Validate reattachment length against Armaly et al.
- Implement control to reduce separation zone
- Add stability analysis
- Study Reynolds number dependence
