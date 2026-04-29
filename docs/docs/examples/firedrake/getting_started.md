---
sidebar_position: 1
---

# Getting Started with Firedrake

This section covers the standard reinforcement learning interface for HydroGym's Firedrake-based flow environments, using the familiar `env.reset()` and `env.step()` API from [Gymnasium](https://gymnasium.farama.org/).

:::tip[Looking for advanced workflows?]
The pages that follow cover direct solver access, Newton steady solvers, stability analysis, and classical feedback control. Start here if your goal is training an RL agent; jump to [Cavity](./cavity), [Cylinder](./cylinder), [Pinball](./pinball), or [Step](./step) for lower-level simulation scripts.
:::

The [`examples/firedrake/getting_started/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/firedrake/getting_started) directory contains a configuration reference, an interactive test script, and a complete Stable-Baselines3 training script — everything you need to get an agent running on any of the five 2-D Firedrake environments.

## Example scripts

### [`config_reference.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/getting_started/config_reference.py)

A catalogue of copy-pasteable configurations covering all common use cases. Running it prints annotated output for each example:

```bash
python config_reference.py
```

The file walks through ten progressively more complex setups:

1. Minimal configuration — simplest possible environment
2. Cylinder with velocity probes
3. Rotary cylinder with rotation actuation
4. Cavity with multi-substep simulation and callbacks
5. Pinball with multiple checkpoints for curriculum learning
6. Step flow with random noise forcing for exploration
7. Cylinder loaded from a saved checkpoint
8. Advanced multi-substep with all reward-aggregation strategies
9. All observation types compared side-by-side
10. Recommended production configuration for RL training

### [`test_firedrake_env.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/getting_started/test_firedrake_env.py)

An interactive script for smoke-testing any environment from the command line. It prints observations and rewards at each step and accepts both single-process and MPI execution:

```bash
# Single process
python test_firedrake_env.py --environment cylinder --num-steps 10

# MPI parallel
mpirun -np 4 python test_firedrake_env.py --environment cylinder --num-steps 50
```

The source file also serves as inline configuration documentation: every available option is annotated directly in the argument parser.

### [`train_sb3_firedrake.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/getting_started/train_sb3_firedrake.py)

A self-contained training script using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). It wraps the environment with `Monitor` and `VecNormalize`, saves checkpoints and normalization statistics alongside the model, and logs metrics to TensorBoard. PPO, TD3, and SAC are all supported:

```bash
# Basic training run
python train_sb3_firedrake.py --env cylinder --algo PPO --total-timesteps 100000

# High-Reynolds cavity with SAC
python train_sb3_firedrake.py --env cavity --reynolds 7500 --mesh fine --algo SAC

# Monitor training progress
tensorboard --logdir logs/
```

Pure Python execution — no MPMD launch required for Firedrake environments.

### [`run_example_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/getting_started/run_example_docker.sh)

A convenience wrapper that handles Docker image selection and volume mounting. Pass `train` to run the SB3 training script instead of the environment test:

```bash
./run_example_docker.sh          # test environment
./run_example_docker.sh train    # train SB3 agent
```

## Quick start

```bash
# 1. Browse all configuration options
python config_reference.py

# 2. Verify the environment runs correctly
python test_firedrake_env.py --environment cylinder --num-steps 10 --verbose

# 3. Copy a configuration template from config_reference.py and start training
python train_sb3_firedrake.py --env cylinder --algo PPO --total-timesteps 100000
```

## Configuration reference

### Flow configuration (`flow_config`)

| Parameter | Description | Options / examples |
|-----------|-------------|-------------------|
| `mesh` | Mesh resolution | `'coarse'`, `'medium'`, `'fine'` |
| `Re` | Reynolds number | Flow-dependent (e.g. `100` for cylinder) |
| `observation_type` | Observation method | `'lift_drag'`, `'stress_sensor'`, `'velocity_probes'`, `'pressure_probes'`, `'vorticity_probes'` |
| `probes` | Probe coordinates | `[(x1, y1), (x2, y2), ...]` |
| `restart` | Checkpoint source | `None` (auto-infer), `'Cylinder_2D_Re100_medium_FD'`, `'/path/to/file.h5'`, or a list for curriculum |
| `local_dir` | Local checkpoint directory | `'/path/to/checkpoints'` — bypasses HuggingFace Hub |
| `cache_dir` | HuggingFace cache directory | `'/path/to/cache'` |
| `velocity_order` | FEM element order | `2` (default, P2-P1 Taylor-Hood) |

### Solver configuration (`solver_config`)

| Parameter | Description | Default / options |
|-----------|-------------|------------------|
| `dt` | Time step | **Required** — `1e-2` for cylinder, `1e-4` for cavity |
| `order` | BDF order | `3` (options: 1, 2, 3) |
| `stabilization` | Stabilization scheme | `'supg'`, `'gls'`, `'none'` |
| `rtol` | Krylov solver tolerance | `1e-6` |

### Actuation configuration (`actuation_config`)

| Parameter | Description | Default / options |
|-----------|-------------|------------------|
| `num_substeps` | Solver steps per `env.step()` call | `1` |
| `reward_aggregation` | How rewards are combined across substeps | `'mean'`, `'sum'`, `'median'` |

### Environment settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_steps` | Episode length in environment steps | `1e6` |
| `callbacks` | List of callbacks to attach to the solver | `[]` |

## Available environments

| Environment | Inputs | Control type | Default observation | Meshes |
|-------------|--------|--------------|---------------------|--------|
| [`Cylinder`](../../api/firedrake/envs/cylinder/flow) | 1 | Blowing / suction (±0.1) | `lift_drag` | medium, fine |
| [`RotaryCylinder`](../../api/firedrake/envs/cylinder/flow) | 1 | Rotation (±0.5π rad) | `lift_drag` | medium, fine |
| [`Pinball`](../../api/firedrake/envs/pinball/flow) | 3 | Rotation (±10.0) | `lift_drag` | medium, fine |
| [`Cavity`](../../api/firedrake/envs/cavity/flow) | 1 | Blowing / suction (±0.1) | `stress_sensor` | medium, fine |
| [`Step`](../../api/firedrake/envs/step/flow) | 1 | Blowing / suction (±0.1) | `stress_sensor` | coarse, medium, fine |

## Observation types

**Force-based**
- `'lift_drag'` — returns `(CL, CD)` for single-cylinder cases; `(CL1, CD1, CL2, CD2, CL3, CD3)` for Pinball.

**Sensor-based**
- `'stress_sensor'` — returns wall shear stress as a scalar.

**Probe-based**
- `'velocity_probes'` — returns `[u1, u2, …, v1, v2, …]` at the specified probe locations.
- `'pressure_probes'` — returns `[p1, p2, …]` at the specified probe locations.
- `'vorticity_probes'` — returns `[ω1, ω2, …]` at the specified probe locations.

:::note
Probe-based observations require a `probes` list in `flow_config`.
:::

## Usage examples

### Basic cylinder environment

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

### Multi-substep simulation

Running multiple solver steps per `env.step()` call increases the physical time elapsed per agent decision and is often necessary to give the flow time to respond to an actuation change:

```python
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {'mesh': 'medium', 'Re': 100},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
    'actuation_config': {
        'num_substeps': 5,
        'reward_aggregation': 'mean',
    },
}

env = FlowEnv(env_config)
# Each env.step() now advances the simulation by 5 × dt = 0.05 time units
```

### Training with Stable-Baselines3

```python
from hydrogym import FlowEnv
import hydrogym.firedrake as hgym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    env_config = {
        'flow': hgym.Cylinder,
        'flow_config': {'mesh': 'medium', 'Re': 100},
        'solver': hgym.SemiImplicitBDF,
        'solver_config': {'dt': 1e-2},
        'actuation_config': {'num_substeps': 2},
    }
    return Monitor(FlowEnv(env_config))

env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=100_000)

model.save("ppo_cylinder")
env.save("vec_normalize.pkl")
```

See [`train_sb3_firedrake.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/firedrake/getting_started/train_sb3_firedrake.py) for the full command-line version.

### Automatic checkpoint loading

When no `restart` key is provided, HydroGym constructs an environment name from the flow class, Reynolds number, and mesh resolution and downloads the corresponding checkpoint from the HuggingFace Hub automatically:

```python
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {
        'mesh': 'medium',
        'Re': 100,
        # No 'restart' key — loads 'Cylinder_2D_Re100_medium_FD' from HF Hub
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
print(f"Loaded: {env.flow.checkpoint_path}")
```

### Local checkpoint directory

For offline environments or HPC clusters without internet access, point `local_dir` at a directory containing pre-downloaded checkpoint folders:

```python
env_config = {
    'flow': hgym.Cylinder,
    'flow_config': {
        'mesh': 'medium',
        'Re': 100,
        'local_dir': '/workspace/my_checkpoints',
        # Loads: /workspace/my_checkpoints/Cylinder_2D_Re100_medium_FD/*.ckpt
    },
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-2},
}

env = FlowEnv(env_config)
```

### Curriculum learning with multiple checkpoints

Passing a list of checkpoint paths causes each `env.reset()` to randomly select one, enabling a simple form of curriculum learning:

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
obs, info = env.reset()
print(f"Started from checkpoint index: {info.get('checkpoint_index')}")
```

### Probe-based observations

```python
import numpy as np

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
print(f"Observation shape: {obs.shape}")  # (40,) — 20 probes × 2 velocity components
```

### Attaching callbacks

[`CheckpointCallback`](../../api/firedrake/utils/io) and [`LogCallback`](../../api/firedrake/utils/io) let you save restart files and log observables without modifying the training loop:

```python
from hydrogym.firedrake.utils.io import CheckpointCallback, LogCallback

env_config = {
    'flow': hgym.Cavity,
    'flow_config': {'mesh': 'fine', 'Re': 7500},
    'solver': hgym.SemiImplicitBDF,
    'solver_config': {'dt': 1e-4},
    'callbacks': [
        CheckpointCallback(interval=1000, filename='cavity_checkpoint.h5'),
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

## Checkpoint management

HydroGym resolves checkpoint sources in the following order of precedence:

| Method | `restart` value | Behaviour |
|--------|----------------|-----------|
| Automatic | `None` (omitted) | Constructs the canonical environment name and downloads from HF Hub |
| Environment name | `'Cylinder_2D_Re100_medium_FD'` | Downloads the named environment from HF Hub |
| Explicit path | `'/path/to/checkpoint.h5'` | Loads the file directly |
| Multiple checkpoints | `['ckpt1.h5', 'ckpt2.h5']` | Randomly selects one on each `reset()` |

The canonical name follows the pattern `{FlowClass}_2D_Re{Reynolds}_{mesh}_FD`, for example `Cylinder_2D_Re100_medium_FD` or `Cavity_2D_Re7500_medium_FD`.

After environment creation you can verify which checkpoint was loaded:

```python
env = FlowEnv(env_config)
if env.flow.checkpoint_path:
    print(f"Loaded: {env.flow.checkpoint_path}")
else:
    print("Starting from zero initial condition")
```

## Available callbacks

All callbacks are importable from [`hydrogym.firedrake.utils.io`](../../api/firedrake/utils/io):

| Callback | Purpose | Key parameters |
|----------|---------|----------------|
| [`CheckpointCallback`](../../api/firedrake/utils/io) | Save HDF5 restart files | `interval`, `filename`, `write_mesh` |
| [`ParaviewCallback`](../../api/firedrake/utils/io) | Export PVD files for visualisation | `interval`, `filename`, `postprocess` |
| [`LogCallback`](../../api/firedrake/utils/io) | Write scalar time series to a text file | `interval`, `filename`, `postprocess`, `nvals` |
| [`SnapshotCallback`](../../api/firedrake/utils/io) | Save snapshots for modal analysis | `interval`, `filename` |
| [`GenericCallback`](../../api/firedrake/utils/io) | Arbitrary Python callback | `callback`, `interval` |
