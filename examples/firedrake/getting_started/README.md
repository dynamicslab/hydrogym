# Getting Started with Firedrake Environments

✅ **START HERE** for standard RL interface examples using `env.reset()` and `env.step()`.

This directory contains comprehensive configuration examples and testing utilities for HydroGym's Firedrake-based flow environments using the **standard RL interface**.

> **Looking for advanced workflows?** (steady solvers, stability analysis, direct control)
> See [`../advanced/`](../advanced/) for specialized research and development examples.

## 📁 Files

### [`config_reference.py`](config_reference.py)
**Comprehensive configuration examples** - Copy-pasteable configurations for all use cases.

Run to see all examples:
```bash
python config_reference.py
```

Contains 10 detailed examples:
1. **Minimal Configuration** - Simplest setup using defaults
2. **Cylinder with Velocity Probes** - Probe-based observations
3. **Rotary Cylinder** - Rotation actuation
4. **Cavity with Multi-Substep** - Multi-substep simulation with callbacks
5. **Pinball with Multiple Checkpoints** - Curriculum learning
6. **Step with Noise Forcing** - Random forcing for exploration
7. **Cylinder with Restart** - Load from checkpoint
8. **Advanced Multi-Substep** - All aggregation strategies
9. **All Observation Types** - Comparing observation modes
10. **Production RL Setup** - Recommended training configuration

### [`test_firedrake_env.py`](test_firedrake_env.py)
**Interactive test script** - Test environments with command-line arguments.

Usage:
```bash
# Single process
python test_firedrake_env.py --environment cylinder --num-steps 10

# MPI parallel
mpirun -np 4 python test_firedrake_env.py --environment cylinder --num-steps 50
```

Contains **inline configuration documentation** showing all available options.

## 🎯 Quick Start

### 1. View All Configuration Options
```bash
python config_reference.py
```

### 2. Test an Environment
```bash
python test_firedrake_env.py --environment cylinder --num-steps 10 --verbose
```

### 3. Copy a Configuration Template
Open `config_reference.py` and copy the example that matches your use case.

## 📚 Configuration Categories

### **Flow Configuration** (`flow_config`)
| Parameter | Description | Options/Examples |
|-----------|-------------|------------------|
| `mesh` | Mesh resolution | `'coarse'`, `'medium'`, `'fine'` |
| `Re` | Reynolds number | Flow-dependent (e.g., 100 for cylinder) |
| `observation_type` | Observation method | `'lift_drag'`, `'stress_sensor'`, `'velocity_probes'`, `'pressure_probes'`, `'vorticity_probes'` |
| `probes` | Probe locations | `[(x1, y1), (x2, y2), ...]` |
| `restart` | Checkpoint file(s) | `None` (auto), `'file.h5'`, `'Cylinder_2D_Re100_medium_FD'` (env name), or `['file1.h5', 'file2.h5']` (multiple) |
| `local_dir` | Local checkpoint directory | `'/path/to/checkpoints'` (for offline/testing) |
| `cache_dir` | Custom cache directory | `'/path/to/cache'` (where HF downloads are stored) |
| `velocity_order` | FEM element order | `2` (default, P2-P1 Taylor-Hood) |

### **Solver Configuration** (`solver_config`)
| Parameter | Description | Default/Options |
|-----------|-------------|-----------------|
| `dt` | Time step | **REQUIRED** (e.g., `1e-2` for cylinder, `1e-4` for cavity) |
| `order` | BDF order | `3` (options: 1, 2, 3) |
| `stabilization` | Stabilization type | `'supg'`, `'gls'`, `'none'` |
| `rtol` | Krylov tolerance | `1e-6` |

### **Actuation Configuration** (`actuation_config`)
| Parameter | Description | Default/Options |
|-----------|-------------|-----------------|
| `num_substeps` | Solver steps per action | `1` (default) |
| `reward_aggregation` | Aggregation method | `'mean'`, `'sum'`, `'median'` |

### **Environment Settings**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_steps` | Episode length | `1e6` |
| `callbacks` | Callback list | `[]` |

## 🔬 Available Environments

| Environment | Inputs | Control Type | Default Obs | Meshes |
|-------------|--------|--------------|-------------|--------|
| **Cylinder** | 1 | Blowing/suction (±0.1) | lift_drag | medium, fine |
| **RotaryCylinder** | 1 | Rotation (±0.5π rad) | lift_drag | medium, fine |
| **Pinball** | 3 | Rotation (±10.0) | lift_drag | medium, fine |
| **Cavity** | 1 | Blowing/suction (±0.1) | stress_sensor | medium, fine |
| **Step** | 1 | Blowing/suction (±0.1) | stress_sensor | coarse, medium, fine |

## 📊 Observation Types

### 1. **Force-Based Observations**
- `'lift_drag'` → Returns `(CL, CD)` for cylinder/rotary, `(CL1, CD1, CL2, CD2, CL3, CD3)` for pinball

### 2. **Sensor-Based Observations**
- `'stress_sensor'` → Returns wall shear stress (scalar)

### 3. **Probe-Based Observations**
- `'velocity_probes'` → Returns `[u1, u2, ..., v1, v2, ...]` at probe locations
- `'pressure_probes'` → Returns `[p1, p2, ...]` at probe locations
- `'vorticity_probes'` → Returns `[ω1, ω2, ...]` at probe locations

**Note:** For probe-based observations, you must specify `probes` in `flow_config`.

## 🎓 Usage Examples

### Example 1: Basic Cylinder Environment
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

### Example 2: Multi-Substep Simulation
```python
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {'mesh': 'medium', 'Re': 100},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
    'actuation_config': {
        'num_substeps': 5,              # Run 5 solver steps per action
        'reward_aggregation': 'mean',   # Average rewards over substeps
    },
}

env = FlowEnv(env_config)
# Each env.step() now runs 5 simulation steps internally
```

### Example 3: Automatic Checkpoint Loading (NEW!)
```python
# Checkpoints are automatically inferred from flow config and downloaded from HF Hub
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {
        'mesh': 'medium',
        'Re': 100,
        # No 'restart' specified - automatically loads 'Cylinder_2D_Re100_medium_FD'
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
# Checkpoint auto-downloaded from HF Hub and loaded!
print(f"Loaded checkpoint: {env.flow.checkpoint_path}")
```

### Example 4: Local Checkpoint Directory
```python
# Use local checkpoints without HF Hub (for offline/testing)
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {
        'mesh': 'medium',
        'Re': 100,
        'local_dir': '/workspace/my_checkpoints',  # Local directory
        # Automatically loads: /workspace/my_checkpoints/Cylinder_2D_Re100_medium_FD/*.ckpt
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
```

### Example 5: Multiple Checkpoints for Curriculum Learning
```python
env_config = {
    'flow': hgym.Pinball,
    'flow_config': {
        'mesh': 'fine',
        'Re': 30,
        'restart': [
            'checkpoint_early.h5',
            'checkpoint_mid.h5',
            'checkpoint_late.h5',
        ],
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
# Each reset() randomly selects one of the three initial conditions
obs, info = env.reset()
print(f"Started from checkpoint index: {info.get('checkpoint_index')}")
```

### Example 6: Probe-Based Observations
```python
import numpy as np

# Define wake probes
wake_probes = [(x, 0.0) for x in np.linspace(1.0, 10.0, 20)]

env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {
        'mesh': 'medium',
        'Re': 100,
        'observation_type': 'velocity_probes',
        'probes': wake_probes,
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # (40,) for 20 probes × 2 velocity components
```

### Example 7: Using Callbacks
```python
from hydrogym.firedrake.io import CheckpointCallback, LogCallback

env_config = {
    'flow': hgym.Cavity,
    'flow_config': {'mesh': 'fine', 'Re': 7500},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-4},
    'callbacks': [
        CheckpointCallback(
            interval=1000,
            filename='cavity_checkpoint.h5',
        ),
        LogCallback(
            postprocess=lambda flow: flow.get_observations(),
            nvals=1,
            interval=10,
            filename='cavity_log.txt',
        ),
    ],
}

env = FlowEnv(env_config)
```

## 💾 Checkpoint Management

HydroGym provides flexible checkpoint management with automatic inference and HuggingFace Hub integration.

### **Checkpoint Loading Methods**

| Method | Example | Use Case |
|--------|---------|----------|
| **Automatic** | No `restart` specified | Auto-loads from HF Hub based on flow config |
| **Environment Name** | `restart='Cylinder_2D_Re100_medium_FD'` | Load specific HF Hub environment |
| **Explicit Path** | `restart='/path/to/checkpoint.h5'` | Use local checkpoint file |
| **Multiple Checkpoints** | `restart=['ckpt1.h5', 'ckpt2.h5']` | Random selection for curriculum learning |

### **Configuration Parameters**

```python
flow_config = {
    # Checkpoint configuration
    'restart': None,  # or path, environment name, or list
    'local_dir': '/path/to/local/checkpoints',  # For offline/testing
    'cache_dir': '/path/to/custom/cache',  # Custom HF cache location
}
```

### **Automatic Checkpoint Naming**

Checkpoints follow the pattern: `{FlowClass}_2D_Re{Reynolds}_{mesh}_FD`

Examples:
- `Cylinder_2D_Re100_medium_FD` - Cylinder at Re=100 on medium mesh
- `Pinball_2D_Re30_fine_FD` - Pinball at Re=30 on fine mesh
- `Cavity_2D_Re7500_medium_FD` - Cavity at Re=7500 on medium mesh

### **How It Works**

1. **No restart specified** → Auto-constructs environment name → Downloads from HF Hub → Loads first checkpoint
2. **Environment name given** → Downloads from HF Hub → Loads first checkpoint
3. **Explicit path** → Uses path directly
4. **Local directory** → Searches local directory → Uses symlinks (no duplication)

### **Verification**

After loading, check the checkpoint:
```python
env = FlowEnv(env_config)
if env.flow.checkpoint_path:
    print(f"Loaded: {env.flow.checkpoint_path}")
else:
    print("Starting from zeros")
```

## 🎛️ Available Callbacks

Import from `hydrogym.firedrake.io`:

| Callback | Purpose | Key Parameters |
|----------|---------|----------------|
| `CheckpointCallback` | Save HDF5 checkpoints | `interval`, `filename`, `write_mesh` |
| `ParaviewCallback` | Export for visualization | `interval`, `filename`, `postprocess` |
| `LogCallback` | Log to text file | `interval`, `filename`, `postprocess`, `nvals` |
| `SnapshotCallback` | Save for modal analysis | `interval`, `filename` |
| `GenericCallback` | Custom function | `callback`, `interval` |

## 📖 Additional Resources

- **HydroGym Documentation**: [https://hydrogym.readthedocs.io](https://hydrogym.readthedocs.io)
- **Firedrake Documentation**: [https://www.firedrakeproject.org](https://www.firedrakeproject.org)
- **Gymnasium API**: [https://gymnasium.farama.org](https://gymnasium.farama.org)

## 💡 Tips

1. **Start with defaults** - Use `example_minimal()` and add options incrementally
2. **Check timestep stability** - If simulation diverges, reduce `dt`
3. **Use multi-substep for RL** - Improves sample efficiency by running multiple simulation steps per policy action
4. **Multiple checkpoints for exploration** - Provides diverse initial conditions
5. **Monitor with callbacks** - Use `LogCallback` to track training progress
6. **Cavity is stiff** - Requires very small timestep (`dt=1e-4`)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Simulation diverges | Reduce `dt` or add stabilization |
| Low observation values | Check `observation_type` and `probes` configuration |
| Slow performance | Use coarser mesh or increase `num_substeps` |
| Checkpoint load fails | Verify file exists and mesh compatibility |
| Import errors | Ensure Firedrake is installed and activated |

## 📝 Contributing

To add new examples:
1. Add a new `example_*()` function in `config_reference.py`
2. Document the configuration purpose and parameters
3. Include in the summary table and main execution block

---

**Last Updated**: March 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
