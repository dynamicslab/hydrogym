# Advanced Firedrake Examples

⚠️ **NOTE**: These examples demonstrate **advanced workflows and direct solver access**, not the standard RL interface.

**If you're looking for standard RL examples** (reset, step, action_space.sample), please see [`../getting_started/`](../getting_started/).

## What's in This Directory

These examples show specialized research and development workflows:

- **Direct solver manipulation** - Newton solvers, steady-state computation
- **Linear stability analysis** - Eigenvalue computation, mode analysis
- **Specialized control** - PD control, phase-swept control (not through RL interface)
- **Advanced workflows** - Multi-stage simulations, perturbation studies
- **Research tools** - Pressure probes, linearization, system identification

## Who Should Use These

These examples are intended for:
- Researchers developing new control strategies
- Users who need direct access to Firedrake solvers
- Advanced users exploring flow physics and stability
- Developers extending HydroGym capabilities

## Example Types

### Steady-State Solvers (`solve-steady.py`)
Find equilibrium solutions using Newton iteration:
```python
from hydrogym.firedrake import Cylinder, IPCS
from hydrogym.firedrake.utils import solve_steady_state_stokes

flow = Cylinder(Re=100, mesh='medium')
solver_parameters = {"snes_monitor": None}
solver = hgym.NewtonSolver(
    flow,
    stabilization="gls",  
    solver_parameters=solver_parameters,
)

solver.solve()
```

### Direct Control (`pd-control.py`)
Apply feedback control directly (not through RL interface):
```python
# Direct PD control loop
for i in range(num_steps):
    CL, CD = flow.get_observations()
    action = -Kp * CL - Kd * (CL - CL_prev) / dt
    solver.step(action)
```

### Workflow Scripts (`unsteady.py`)
Multi-stage research workflows:
1. Solve steady state at low Re
2. Ramp Reynolds number
3. Add perturbation
4. Run transient simulation

## Comparison with Standard RL Interface

| Feature | Standard RL (getting_started/) | Advanced (this directory) |
|---------|-------------------------------|---------------------------|
| **Interface** | `env.reset()`, `env.step()` | Direct solver calls |
| **Purpose** | Reinforcement learning training | Research, analysis, development |
| **Complexity** | Simple, standardized | Flexible, specialized |
| **Use case** | Training RL agents | Flow physics, control design |

## Getting Started with Advanced Examples

1. **Ensure you understand the standard RL interface first** - See [`../getting_started/`](../getting_started/)

2. **Choose an environment** - Each has its own README with detailed descriptions

3. **Start simple** - Begin with `run-transient.py` in any environment

4. **Progress to specialized scripts** - Try steady solvers, stability analysis, control

## Need Standard RL Examples?

👉 **Go to [`../getting_started/`](../getting_started/)** for standard RL interface examples with:
- Complete configuration reference
- Interactive test scripts
- Standard `reset()`/`step()` interface
- Copy-paste ready templates

---
