---
sidebar_position: 1
---

# Getting Started

✅ **START HERE** for standard RL interface examples using MAIA solver with `env.reset()` and `env.step()`.

This directory contains examples and utilities for HydroGym's MAIA-based flow environments using the **standard RL interface** with **MPMD coupling**.

## Files

### [`test_maia_env.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/maia/getting_started/test_maia_env.py)
**Interactive test script** - Test MAIA environments with command-line arguments via MPMD execution.

Usage:
```bash
# From a prepared workspace (m-AIA reads properties_run.toml from the working directory)
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run_000
cd test_run_000

# Basic usage (1 Python + 1 MAIA process)
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties_run.toml

# Parallel MAIA (1 Python + 4 MAIA processes)
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 4 maia properties_run.toml
```

### [`train_sb3_maia.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/maia/getting_started/train_sb3_maia.py)
**SB3 training script** - Train reinforcement learning agents (PPO/TD3/SAC) with Stable-Baselines3.

Features:
- Monitor wrapper for episode statistics
- VecNormalize for observation/reward normalization
- Checkpoint saving with normalization stats
- TensorBoard logging

Usage:
```bash
# First, prepare workspace
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./train_run

# Then train with MPMD execution
cd train_run
mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO --total-timesteps 100000 : -np 1 maia properties_run.toml

# Monitor training
tensorboard --logdir logs/
```

### [`prepare_workspace.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/maia/getting_started/prepare_workspace.py)
**Workspace setup utility** - Downloads environment data and creates workspace for HPC jobs.

Usage:
```bash
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./my_workspace
```

### [`run_example_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/maia/getting_started/run_example_docker.sh)
**Launch script** - Prepares the workspace and starts the MPMD run (1 Python + 1 m-AIA rank). Runs as is in the HydroGym images, where `maia` starts the m-AIA build for the visible GPU (as a non-root user, from a copy of `/opt/hydrogym/examples`, which is not writable); elsewhere, activate your HydroGym environment and set `MAIA=/path/to/maia`.

Usage:
```bash
# Test environment
./run_example_docker.sh

# Train SB3 agent
./run_example_docker.sh train
```

## Quick Start

**⚠️ Internet Required:** Environment files are downloaded from Hugging Face Hub. For offline/HPC use, see [Offline Usage](#offline-usage-no-internet-on-compute-nodes) below.

### Basic Test Workflow (Online)

**Step 1:** Prepare the workspace (downloads data from Hugging Face Hub, **requires internet**, no MPI needed):

```bash
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run_000
```

This creates:
- `test_run_000/` - working directory
- `test_run_000/properties_run.toml` - MAIA run configuration (symlink)
- `test_run_000/geometry.toml` - geometry configuration (symlink)
- `test_run_000/environment_config.yaml` - environment/actuation configuration (symlink)
- `test_run_000/out_lb/` - solver output directory (grid + restart files symlinked inside)
- `test_run_000/stl/` - STL geometry files (symlink)

**Step 2:** Run the test with MPMD execution:

```bash
cd test_run_000
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 --num-steps 10 : -np 1 maia properties_run.toml
```

This runs:
- 1 Python process (RL environment)
- 1 MAIA process (CFD solver)
- Communication via MPI

### Parallel MAIA

To run with more MAIA processes for larger meshes:

```bash
cd test_run_000
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 4 maia properties_run.toml
```

### Explore Options

The test script supports many configuration options:

```bash
python test_maia_env.py --help
```

## Usage Examples

### Example 1: Basic RL Loop

```python
import hydrogym.maia as maia
import numpy as np

# Create MAIA environment from Hugging Face Hub
# Probe locations are flattened: [x0, y0, x1, y1, ...]
probe_locations = []
for x in np.linspace(1.0, 8.0, 8):
    probe_locations.extend([x, 0.0])

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=probe_locations,
    obs_normalization_strategy="U_inf",
)

obs, info = env.reset()

# Run standard RL loop
for step in range(100):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**Important:** This script must be run with MPMD:
```bash
mpirun -np 1 python your_script.py : -np 4 maia properties_run.toml
```

### Example 2: Training with Stable-Baselines3

```python
# See train_sb3_maia.py for the full implementation (there --obs-norm defaults to none, VecNormalize
# normalizes the observations)
import hydrogym.maia as maia
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Define probes
probe_locations = []
for x in np.linspace(1.0, 8.0, 8):
    for y in np.linspace(-1.0, 1.0, 5):
        probe_locations.extend([x, y])


def make_env():
    env = maia.from_hf(
        "Cylinder_2D_Re200",
        use_clean_cache=False,
        probe_locations=probe_locations,
        obs_normalization_strategy="U_inf",
    )
    return Monitor(env)


env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=100000)

model.save("ppo_cylinder")
env.save("vec_normalize.pkl")
```

**Run with MPMD:**
```bash
cd work_dir
mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO : -np 1 maia properties_run.toml
```

### Example 3: Custom Probe Configuration

```python
import hydrogym.maia as maia
import numpy as np

# Define wake probes (flattened format)
wake_probes = []
for x in np.linspace(1.0, 8.0, 10):
    for y in np.linspace(-1.0, 1.0, 5):
        wake_probes.extend([x, y])  # 50 probes total

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=wake_probes,
    obs_normalization_strategy="U_inf",
)

obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # (100,) for 50 probes × 2 velocity components
```

### Example 4: Custom Normalization

```python
import hydrogym.maia as maia
import numpy as np

# Define wake probes (flattened format)
wake_probes = []
for x in np.linspace(1.0, 8.0, 8):
    for y in np.linspace(-1.0, 1.0, 5):
        wake_probes.extend([x, y])  # 40 probes total

# Define custom normalization (location and scale for each probe component)
# For N probes with nDim=2 velocity, you need nDim*N location and scale values
custom_loc = [0.0] * 80  # Here 40 probes × 2 components
custom_scale = [1.0] * 80

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=[2.0, 0.0, 4.0, 0.5, 4.0, -0.5, 6.0, 0.0],  # x, y of each probe  # Your probe locations
    obs_normalization_strategy="customized",
    obs_loc=custom_loc,
    obs_scale=custom_scale,
)
```

### Example 5: Offline Usage with Local Files

```python
import hydrogym.maia as maia

# Use pre-downloaded environment files (no internet required)
env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=[2.0, 0.0, 4.0, 0.5, 4.0, -0.5, 6.0, 0.0],  # x, y of each probe
    local_fallback_dir="/scratch/<project>/hf_environments",  # holds Cylinder_2D_Re200/, see Offline Usage
)
```

## Offline Usage (No Internet on Compute Nodes)

**Important:** Environment files are stored on Hugging Face Hub and require internet access. HPC compute nodes often lack internet connectivity.

### HPC Offline Workflow

**Step 1:** Download the environments into a directory on a filesystem that the compute nodes can read:

```bash
# on a machine with internet access (login node, workstation)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='dynamicslab/HydroGym-environments', repo_type='dataset',
                  allow_patterns='Cylinder_2D_Re200/**', local_dir='/scratch/<project>/hf_environments')
"
```

This writes `/scratch/<project>/hf_environments/Cylinder_2D_Re200/`; further environments go into the same directory.

**Step 2:** Create a workspace preparation script that uses the local files:

```python
# prepare_offline_workspace.py
from hydrogym.maia.workspace import prepare_maia_workspace

work_dir, props_file = prepare_maia_workspace(
    environment_name="Cylinder_2D_Re200",
    work_dir="./my_run",
    local_fallback_dir="/scratch/<project>/hf_environments",  # used directly, without contacting the Hub
)

print(f"Workspace: {work_dir}")
print(f"Properties: {props_file}")
```

**Step 3:** In your RL script, use the same `local_fallback_dir`:

```python
import hydrogym.maia as maia

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=[2.0, 0.0, 4.0, 0.5, 4.0, -0.5, 6.0, 0.0],  # x, y of each probe
    local_fallback_dir="/scratch/<project>/hf_environments",
)
```

**Step 4:** Run on compute node (offline):

```bash
cd my_run
mpirun -np 1 python ../my_rl_script.py : -np 4 maia properties_run.toml
```

