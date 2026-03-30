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

## Environments

Each environment folder contains advanced workflow scripts:

| Environment | Description | Advanced Examples |
|-------------|-------------|-------------------|
| [**cavity/**](cavity/) | Lid-driven cavity | Steady solver, stability analysis, sine forcing |
| [**cylinder/**](cylinder/) | Circular cylinder | PD control, stability, phase-swept control, probes |
| [**pinball/**](pinball/) | Three cylinders | Steady → transient workflow, complex wake |
| [**step/**](step/) | Backward-facing step | Separation control, steady solver |

## Example Types

### Steady-State Solvers (`solve-steady.py`)
Find equilibrium solutions using Newton iteration:
```python
from hydrogym.firedrake import Cylinder, IPCS
from hydrogym.firedrake.utils import solve_steady_state_stokes

flow = Cylinder(Re=100, mesh='medium')
solver = IPCS(flow, dt=1e-2)
solve_steady_state_stokes(flow, solver, max_iter=10)
```

### Linear Stability Analysis (`stability.py`)
Compute eigenvalues and unstable modes:
```python
from hydrogym.firedrake.utils import compute_stability

eigenvalues, eigenmodes = compute_stability(
    flow, solver, n_eigs=10, target=1j
)
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

## Running Examples

### Basic Execution
```bash
cd advanced/cylinder
python run-transient.py
```

### With MPI Parallelism
```bash
mpirun -np 4 python stability.py
```

### Typical Workflow
```bash
# 1. Understand the flow
python run-transient.py

# 2. Find base state
python solve-steady.py

# 3. Analyze stability
python stability.py

# 4. Test control
python pd-control.py
```

## Documentation

Each environment has detailed documentation:
- [cavity/README.md](cavity/README.md) - Lid-driven cavity
- [cylinder/README.md](cylinder/README.md) - Circular cylinder
- [pinball/README.md](pinball/README.md) - Three cylinders
- [step/README.md](step/README.md) - Backward-facing step

## Need Standard RL Examples?

👉 **Go to [`../getting_started/`](../getting_started/)** for standard RL interface examples with:
- Complete configuration reference
- Interactive test scripts
- Standard `reset()`/`step()` interface
- Copy-paste ready templates

---

**Remember**: These are advanced examples for specialized workflows. For standard RL training, use [`../getting_started/`](../getting_started/).
