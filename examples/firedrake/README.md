# Firedrake Examples

Comprehensive examples demonstrating Hydrogym's Firedrake backend for CFD and flow control.

## Directory Structure

```
firedrake/
├── README.md                        # This file
├── getting_started/                 # START HERE - Standard RL interface
│   ├── README.md                    # Complete getting started guide
│   ├── test_firedrake_env.py        # Interactive test script
│   ├── config_reference.py          # Copy-paste configuration examples
│   └── run_example_docker.sh        # Docker runner script
└── advanced/                        # Advanced workflows (not standard RL)
    ├── README.md                    # Advanced examples overview
    ├── cavity/                      # Open cavity flow
    │   ├── README.md                # Cavity-specific documentation
    │   ├── run-transient.py         # Direct simulation (not RL interface)
    │   ├── solve-steady.py          # Steady-state solver
    │   ├── unsteady.py              # Steady → turbulent workflow
    │   ├── stability.py             # Eigenvalue analysis
    │   └── sine-forcing.py          # Sinusoidal forcing
    ├── cylinder/                    # Flow around circular cylinder
    │   ├── README.md                # Cylinder-specific documentation
    │   ├── run-transient.py         # Direct simulation (not RL interface)
    │   ├── solve-steady.py          # Steady/unstable equilibrium
    │   ├── unsteady.py              # Steady → shedding workflow
    │   ├── stability.py             # Wake stability analysis
    │   ├── pd-control.py            # PD feedback control (not RL)
    │   ├── step_input.py            # Step response test
    │   ├── pd-phase-sweep.py        # Phase-swept control
    │   ├── pressure-probes.py       # Multi-point sensing
    │   └── lti_system.py            # Linearization (WIP)
    ├── pinball/                     # Three-cylinder configuration
    │   ├── README.md                # Pinball-specific documentation
    │   ├── run-transient.py         # Direct simulation (not RL interface)
    │   ├── solve-steady.py          # Steady equilibrium
    │   └── unsteady.py              # Wake dynamics workflow
    └── step/                        # Backward-facing step
        ├── README.md                # Step-specific documentation
        ├── run-transient.py         # Direct simulation (not RL interface)
        ├── solve-steady.py          # Steady recirculation zone
        ├── unsteady.py              # Separation → shedding
        └── step-control.py          # Step input control
```

## Quick Start

### **New Users: Start with the RL Interface**

If you want to use HydroGym for **reinforcement learning** (the standard use case):

```bash
cd getting_started
python test_firedrake_env.py --environment cylinder --num-steps 10
```

This demonstrates the standard `env.reset()` / `env.step()` interface.

**Next steps:**
1. Read [getting_started/README.md](getting_started/README.md) for complete documentation
2. Explore [getting_started/config_reference.py](getting_started/config_reference.py) for copy-paste examples
3. Start training your RL agent!

### **Advanced Users: Direct Solver Access**

If you need **specialized workflows** (steady solvers, stability analysis, direct control):

```bash
cd advanced/cylinder
python run-transient.py
```

**Note:** These examples do NOT use the standard RL interface. See [advanced/README.md](advanced/README.md) for details.

## Example Categories

### Standard RL Interface (`getting_started/`)

**Purpose:** Train reinforcement learning agents

**Key files:**
- `test_firedrake_env.py` - Interactive testing with standard Gym API
- `config_reference.py` - 10 copy-paste configuration examples
- Full documentation in [getting_started/README.md](getting_started/README.md)

**Typical usage:**
```python
from hydrogym import FlowEnv
import hydrogym.firedrake as hgym

env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {'mesh': 'medium', 'Re': 100},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Advanced Workflows (`advanced/`)

**Purpose:** Research, development, and specialized analysis

**Example types:**
- **Direct simulations** (`run-transient.py`) - Time-stepping without RL interface
- **Steady solvers** (`solve-steady.py`) - Newton iteration for equilibria
- **Stability analysis** (`stability.py`) - Eigenvalue computation
- **Direct control** (`pd-control.py`) - Feedback control without RL
- **Specialized workflows** (`unsteady.py`) - Multi-stage simulations

See [advanced/README.md](advanced/README.md) for complete documentation.

## Available Environments

| Environment | Description | Re (default) | Key Phenomena |
|-------------|-------------|--------------|---------------|
| **Cavity** | Open cavity | 7500 | Recirculation, turbulence |
| **Cylinder** | Circular cylinder | 100 | Vortex shedding |
| **Pinball** | Three cylinders | 100 | Complex wake, mode switching |
| **Step** | Backward step | 600 | Separation, reattachment |

All environments are available in both `getting_started/` (RL interface) and `advanced/` (specialized workflows).

## Typical Workflows

### For RL Training (`getting_started/`)

```bash
# 1. Test the environment
cd getting_started
python test_firedrake_env.py --environment cylinder --num-steps 10

# 2. Explore configuration options
python config_reference.py

# 3. Copy a template and start training
# (use your favorite RL library: Stable-Baselines3, RLlib, etc.)
```

### For Research/Analysis (`advanced/`)

```bash
cd advanced/cylinder

# 1. Understand the flow
python run-transient.py

# 2. Find steady state
python solve-steady.py

# 3. Analyze stability
python stability.py

# 4. Test control strategies
python pd-control.py
```

## Running Examples

### Standard RL Interface

```bash
cd getting_started

# Single process
python test_firedrake_env.py --environment cylinder --num-steps 10

# MPI parallel
mpirun -np 4 python test_firedrake_env.py --environment cylinder --num-steps 50

# In Docker
bash run_example_docker.sh
```

### Advanced Workflows

```bash
cd advanced/cylinder

# Basic usage
python run-transient.py

# With MPI parallelism
mpirun -np 4 python stability.py
```

## Output Files

Examples typically generate:

- **`.pvd` files** - Paraview visualization
  - Open with: `paraview solution.pvd`
- **`.h5` files** - Checkpoint files for restart
  - Use with: `flow = hgym.Cylinder(restart="checkpoint.h5")`
- **`.dat` files** - Time series data (lift, drag, energy)
  - Plot with: `np.loadtxt("stats.dat")`

