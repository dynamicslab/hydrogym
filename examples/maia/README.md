# MAIA Examples

Examples demonstrating HydroGym's MAIA backend for high-performance CFD with MPMD coupling.

## What is MAIA?

MAIA is a high-performance CFD solver designed for massively parallel simulations by the Aerodynamic Institute Aachen. HydroGym's MAIA backend enables RL training on large-scale flow control problems with more than 1B cells using MPI-based coupling.

## Directory Structure

```
maia/
├── README.md              # This file
└── getting_started/       # ⭐ START HERE - Standard RL interface
    ├── test_maia_env.py   # Interactive test script with MPMD
    ├── prepare_workspace.py  # Workspace setup utility
    └── run_example_docker.sh # Docker runner script
```

## Quick Start

### Prerequisites

1. **MAIA installation** - MAIA solver must be compiled and available
2. **MPI environment** - OpenMPI or MPICH for MPMD execution
3. **HydroGym with MAIA support** - Install with MAIA backend enabled
4. **Internet access** - Required to download environment data from Hugging Face Hub
   - For offline/HPC use, download environments beforehand and use `local_fallback_dir` (see below)

### Running Your First Example

```bash
cd getting_started

# Step 1: Prepare workspace (downloads from HF Hub, no MPI needed)
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run

# Step 2: Run with MPMD execution (1 Python process + 1 MAIA process)
cd test_run
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml
```

**Note:** MAIA uses MPMD (Multiple Program Multiple Data) execution where Python and MAIA run as separate MPI programs that communicate.

## Available Examples

### Standard RL Interface (`getting_started/`)

**Purpose:** Train reinforcement learning agents using MAIA solver

**Key files:**
- `test_maia_env.py` - Interactive testing with standard Gym API via MPMD
- `prepare_workspace.py` - Utility to set up MAIA workspace directories
- `run_example_docker.sh` - Docker execution script

**Typical usage:**
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

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

**Note:** MAIA environments must be run with MPMD execution - see [getting_started/README.md](getting_started/README.md) for details.

## Configuration Options

MAIA environments are configured via keyword arguments to `maia.from_hf()`:

```python
env = maia.from_hf(
    'Cylinder_2D_Re200',                      # Environment name (required)
    hf_repo_id='dynamicslab/HydroGym-environments',  # HF Hub repository (optional)
    probe_locations=[x0, y0, x1, y1, ...],    # Flattened probe coordinates (required) - for 3D: [x0, y0, z0, x1, y1, z1, ...]
    obs_normalization_strategy='U_inf',       # Normalization: 'U_inf', 'probewise_mean_std', 'none', 'customized'
    obs_loc=[...],                            # Custom normalization location (if strategy='customized')
    obs_scale=[...],                          # Custom normalization scale (if strategy='customized')
    use_clean_cache=False,                    # Whether to use clean cache directory

    # For offline/HPC use (no internet on compute nodes):
    local_fallback_dir='/path/to/local/environments',  # Use pre-downloaded files
)
```

**Key Parameters for Offline/HPC Use:**
- `local_fallback_dir`: Path to pre-downloaded environment files (required if no internet)
- `use_clean_cache`: Set to `False` to avoid downloading (default: `True`)
- `hf_repo_id`: HF repository to download from when online (default: `'dynamicslab/HydroGym-environments'`)

**Available Environments in 2D and 3D:**
- `Cylinder_2D_Re200` - 2D cylinder flow at Re=200
- `RotaryCylinder_2D_Re1000` - 2D rotating cylinder
- `Cavity_2D_Re4140` - 2D open cavity
- `Pinball_2D_Re130` - 2D pinball configuration
- And many more... (see `maia.list_available_environments()`)

## Workspace Setup

### Online Mode (with Internet Access)

For testing, MAIA environments automatically download and set up workspaces from Hugging Face Hub when you call `maia.from_hf()`.

For HPC jobs, you can pre-download the environment on a login node (no MPI needed):

```bash
# Download and prepare workspace (run on login node with internet)
python getting_started/prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./my_run
```

This downloads the environment data from Hugging Face Hub and creates symlinks to:
```
my_run/
├── properties.toml    # MAIA configuration
├── grid               # Mesh files (symlinked)
└── out/               # Solution output directory
```

Then in your SLURM/PBS job script, the MAIA solver will find the workspace via the `properties.toml` path.

### Offline Mode (No Internet on Compute Nodes)

**Important:** Environment files are stored on Hugging Face Hub and require internet access to download. Many HPC compute nodes do not have internet access.

For offline HPC use:

**Step 1:** Download environments on a machine with internet (e.g., login node or local workstation):

```bash
# Download environment data to a local directory
python -c "
from hydrogym.data_manager import HFDataManager
dm = HFDataManager(repo_id='dynamicslab/HydroGym-environments', use_clean_cache=False)
env_path = dm.get_environment_path('Cylinder_2D_Re200')
print(f'Downloaded to: {env_path}')
"
```

**Step 2:** Copy the downloaded data to your HPC filesystem (if needed):

```bash
# The environment data is in the Hugging Face cache
# Copy it to a shared location accessible from compute nodes
cp -r ~/.cache/huggingface/hub/models--dynamicslab--HydroGym-environments /scratch/my_project/hf_environments/
```

**Step 3:** Use `local_fallback_dir` when creating environments on compute nodes:

```python
import hydrogym.maia as maia

# Point to your local copy of the environments
env = maia.from_hf(
    'Cylinder_2D_Re200',
    probe_locations=[...],
    local_fallback_dir='/scratch/my_project/hf_environments/,
    use_clean_cache=False,  # Use existing cache, don't try to download
)
```

Or with the workspace preparation script:

```bash
# On compute node (offline) - uses local copy
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./my_run
# (Set LOCAL_FALLBACK_DIR environment variable or modify script)
```
