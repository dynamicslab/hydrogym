# Getting Started with MAIA Environments

✅ **START HERE** for standard RL interface examples using MAIA solver with `env.reset()` and `env.step()`.

This directory contains examples and utilities for HydroGym's MAIA-based flow environments using the **standard RL interface** with **MPMD coupling**.

## Files

### [`test_maia_env.py`](test_maia_env.py)
**Interactive test script** - Test MAIA environments with command-line arguments via MPMD execution.

Usage:
```bash
# Basic usage (1 Python + 1 MAIA process)
mpirun -np 1 python test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml

# Parallel MAIA (1 Python + 4 MAIA processes)
mpirun -np 1 python test_maia_env.py --environment Cylinder_2D_Re200 : -np 4 maia properties.toml
```

### [`train_sb3_maia.py`](train_sb3_maia.py)
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
mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO --total-timesteps 100000 : -np 1 maia properties.toml

# Monitor training
tensorboard --logdir logs/
```

### [`prepare_workspace.py`](prepare_workspace.py)
**Workspace setup utility** - Downloads environment data and creates workspace for HPC jobs.

Usage:
```bash
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./my_workspace
```

### [`run_example_docker.sh`](run_example_docker.sh)
**Docker runner script** - Run MAIA examples in Docker with automatic setup.

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
- `test_run_000/properties.toml` - MAIA configuration file (symlink)
- `test_run_000/grid` - mesh file (symlink)
- `test_run_000/out/` - output directory

**Step 2:** Run the test with MPMD execution:

```bash
cd test_run_000
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 --num-steps 10 : -np 1 maia properties.toml
```

This runs:
- 1 Python process (RL environment)
- 1 MAIA process (CFD solver)
- Communication via MPI

### Parallel MAIA

To run with more MAIA processes for larger meshes:

```bash
cd test_run_000
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 4 maia properties.toml
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
    'Cylinder_2D_Re200',
    probe_locations=probe_locations,
    obs_normalization_strategy='U_inf',
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
mpirun -np 1 python your_script.py : -np 4 maia properties.toml
```

### Example 2: Training with Stable-Baselines3

```python
# See train_sb3_maia.py for full implementation
import hydrogym.maia as maia
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
        'Cylinder_2D_Re200',
        use_clean_cache=False,
        probe_locations=probe_locations,
        obs_normalization_strategy='U_inf',
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
mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO : -np 1 maia properties.toml
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
    'Cylinder_2D_Re200',
    probe_locations=wake_probes,
    obs_normalization_strategy='U_inf',
)

obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # (100,) for 50 probes × 2 velocity components
```

### Example 4: Custom Normalization

```python
import hydrogym.maia as maia

# Define wake probes (flattened format)
wake_probes = []
for x in np.linspace(1.0, 8.0, 8):
    for y in np.linspace(-1.0, 1.0, 5):
        wake_probes.extend([x, y])  # 40 probes total

# Define custom normalization (location and scale for each probe component)
# For N probes with nDim=2 velocity, you need nDim*N location and scale values
custom_loc = [0.0] * 80    # Here 40 probes × 2 components
custom_scale = [1.0] * 80

env = maia.from_hf(
    'Cylinder_2D_Re200',
    probe_locations=[...],  # Your probe locations
    obs_normalization_strategy='customized',
    obs_loc=custom_loc,
    obs_scale=custom_scale,
)
```

### Example 5: Offline Usage with Local Files

```python
import hydrogym.maia as maia

# Use pre-downloaded environment files (no internet required)
env = maia.from_hf(
    'Cylinder_2D_Re200',
    probe_locations=[...],
    local_fallback_dir='/scratch/my_project/hf_environments/models--dynamicslab--HydroGym-environments/snapshots/main',
    use_clean_cache=False,  # Don't try to download from HF Hub
)
```

## Offline Usage (No Internet on Compute Nodes)

**Important:** Environment files are stored on Hugging Face Hub and require internet access. HPC compute nodes often lack internet connectivity.

### HPC Offline Workflow

**Step 1:** On a machine with internet (login node or workstation), download environment data:

```bash
# Download to Hugging Face cache
python -c "
from hydrogym.data_manager import HFDataManager
dm = HFDataManager(repo_id='dynamicslab/HydroGym-environments', use_clean_cache=False)
env_path = dm.get_environment_path('Cylinder_2D_Re200')
print(f'Environment downloaded to: {env_path}')
"
```

**Step 2:** Copy to shared HPC filesystem (if needed):

```bash
# Copy from HF cache to shared storage accessible from compute nodes
cp -r ~/.cache/huggingface/hub/models--dynamicslab--HydroGym-environments \
      /scratch/my_project/hf_environments/
```

**Step 3:** Create a workspace preparation script that uses local files:

```python
# prepare_offline_workspace.py
from hydrogym.maia.workspace import prepare_maia_workspace

work_dir, props_file = prepare_maia_workspace(
    environment_name='Cylinder_2D_Re200',
    work_dir='./my_run',
    local_fallback_dir='/scratch/my_project/hf_environments/models--dynamicslab--HydroGym-environments/snapshots/main',
    use_clean_cache=False,  # Use existing cache
    force_download=False,   # Don't try to download
)

print(f"Workspace: {work_dir}")
print(f"Properties: {props_file}")
```

**Step 4:** In your RL script, use the same `local_fallback_dir`:

```python
import hydrogym.maia as maia

env = maia.from_hf(
    'Cylinder_2D_Re200',
    probe_locations=[...],
    local_fallback_dir='/scratch/my_project/hf_environments/models--dynamicslab--HydroGym-environments/snapshots/main',
    use_clean_cache=False,
)
```

**Step 5:** Run on compute node (offline):

```bash
cd my_run
mpirun -np 1 python ../my_rl_script.py : -np 4 maia properties.toml
```

---

**Last Updated**: March 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
