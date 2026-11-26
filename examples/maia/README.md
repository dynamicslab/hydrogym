# MAIA Examples

Examples demonstrating HydroGym's MAIA backend for high-performance CFD with MPMD coupling.

## What is MAIA?

MAIA is a high-performance CFD solver designed for massively parallel simulations by the Aerodynamic Institute Aachen. HydroGym's MAIA backend enables RL training on large-scale flow control problems with more than 1B cells using MPI-based coupling.

## Directory Structure

```
maia/
├── README.md              # This file
├── getting_started/       # START HERE - Standard RL interface
│   ├── gym_make_demo.py   # gym.make() -- the standard entry point (MPMD underneath)
│   ├── test_maia_env.py   # Interactive test script (lower-level, direct from_hf)
│   ├── prepare_workspace.py  # Workspace setup utility
│   └── run_example_docker.sh # MPMD launch script (runs as is in the HydroGym images and on cluster nodes)
└── released_policies/     # Run a released policy against the uncontrolled flow, plot it in the paper style
```

## Quick Start

### Prerequisites

1. **MAIA installation** - MAIA solver must be compiled and available
2. **MPI environment** - OpenMPI or MPICH for MPMD execution
3. **HydroGym with MAIA support** - Install with MAIA backend enabled
4. **Internet access** - Required to download environment data from Hugging Face Hub
   - For offline/HPC use, download environments beforehand and use `local_fallback_dir` (see below)

### Running Your First Example

MAIA is still reached through the standard `gym.make()` entry point --
the only difference from Firedrake/JAX-Fluids is that it must be launched
as an MPMD job, since the real solver runs as a separate MPI application
that Python talks to over MPI, not a library call:

```bash
cd getting_started

# Step 1: Prepare workspace (downloads from HF Hub, no MPI needed)
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run

# Step 2: Run with MPMD execution (1 Python process + 1 MAIA process)
cd test_run
mpirun -np 1 python ../gym_make_demo.py : -np 1 maia properties_run.toml
```

```python
# gym_make_demo.py's core, once inside the MPMD-launched process:
import gymnasium as gym
import hydrogym.registration

env = gym.make("hydrogym-maia/Cylinder_2D_Re200-v0")  # default probe grid, zero extra args
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Other MAIA ids (e.g. `hydrogym-maia/RotaryCylinder_2D_Re1000-v0`) have no
universal default probe grid and need `probe_locations=[...]` passed to
`gym.make()` explicitly -- it raises a clear error naming this if it's
missing, rather than guessing at a flow-geometry-specific default. See
`test_maia_env.py` for the lower-level `hydrogym.maia.from_hf(...)` path
and worked probe-grid examples.

**Note:** MAIA uses MPMD (Multiple Program Multiple Data) execution where Python and MAIA run as separate MPI programs that communicate.

## Available Examples

### Standard RL Interface (`getting_started/`)

**Purpose:** Train reinforcement learning agents using MAIA solver

**Key files:**
- `test_maia_env.py` - Interactive testing with standard Gym API via MPMD
- `prepare_workspace.py` - Utility to set up MAIA workspace directories
- `run_example_docker.sh` - MPMD launch script: prepares the workspace and starts 1 Python + 1 m-AIA rank. Runs as is in the HydroGym images (`maia` starts the m-AIA build for the visible GPU); elsewhere, activate your HydroGym environment and set `MAIA=/path/to/maia`. It does not start a container.

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
    "Cylinder_2D_Re200",
    probe_locations=probe_locations,
    obs_normalization_strategy="U_inf",
)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

**Note:** MAIA environments must be run with MPMD execution - see [getting_started/README.md](getting_started/README.md) for details.

### Released policies (`released_policies/`)

Runs one released policy of `policies/` (by its `MANIFEST.yaml` id) and the uncontrolled flow from the same start
state, and plots the paper quantity (drag, lift, or cavity TKE) in the style of the paper figures:

```bash
cd released_policies
./run_example.sh cylinder_2d_re200/ppo/seed1 runs/cylinder_re200 1
```

See [released_policies/README.md](released_policies/README.md) for the command of every case.

## Configuration Options

MAIA environments are configured via keyword arguments to `maia.from_hf()`:

```python
env = maia.from_hf(
    "Cylinder_2D_Re200",  # Environment name (required)
    hf_repo_id="dynamicslab/HydroGym-environments",  # HF Hub repository (optional)
    probe_locations=[
        x0,
        y0,
        x1,
        y1,
        ...,
    ],  # Flattened probe coordinates (required) - for 3D: [x0, y0, z0, x1, y1, z1, ...]
    obs_normalization_strategy="U_inf",  # Normalization: 'U_inf', 'probewise_mean_std', 'none', 'customized'
    obs_loc=[...],  # Custom normalization location (if strategy='customized')
    obs_scale=[...],  # Custom normalization scale (if strategy='customized')
    # For offline/HPC use (no internet on compute nodes):
    local_fallback_dir="/path/to/local/environments",  # pre-downloaded environments, used without contacting the Hub
)
```

**Key Parameters for Offline/HPC Use:**
- `local_fallback_dir`: directory that holds pre-downloaded environment folders (`<dir>/<environment>/`); an
  environment found there is used directly, without contacting the Hub (required without internet)
- `use_clean_cache`: how downloaded files are cached (`True`, the default: links in `~/.cache/hydrogym`; `"copy"`:
  copies; `False`: the Hugging Face cache paths); it does not prevent a download
- `hf_repo_id`: HF repository to download from when online (default: `'dynamicslab/HydroGym-environments'`)

**Available Environments in 2D and 3D:**
- `Cylinder_2D_Re200` - 2D cylinder flow at Re=200
- `RotaryCylinder_2D_Re1000` - 2D rotating cylinder
- `Cavity_2D_Re4200`, `Cavity_2D_Re7500` - 2D open cavity, one leading-edge jet (weakly unstable / limit cycle)
- `Cavity3Jet_2D_Re4200`, `Cavity3Jet_2D_Re7500` - 2D open cavity, three jets
- `Cavity_3D_Re4200_lz024`, `Cavity_3D_Re7500_lz024`, `Cavity3Jet_3D_Re4200_lz024`, `Cavity3Jet_3D_Re7500_lz024` -
  3D open cavity: the 2D literature cavity extruded over a periodic width of 0.24 D (D/200, 7.4 M cells, D3Q27
  cumulant), one or three jets. Coordinates
  (cavity [0, 1] x [-1, 0]; the 2D cases are shifted by +1.2 in x). Pass `probe_locations` (3D triples;
  `probes_8planes.txt` in the case folder = the 1760 cells observed by the 3D cavity policies in `policies/`). Launch on 4 GPUs as an
  MPMD job (1 Python rank + 4 m-AIA ranks; with `srun --multi-prog` set `MAIA_MPMD_APPNUM`). `max_episode_steps` is
  200 as in 2D; set it for longer roll-outs (60 D/U are 3156 actions). Needs an m-AIA version with the D3Q27
  characteristic outflow and `MAIA_MPMD_APPNUM` support (the pinned `third_party/m-AIA`). See the docstring of `hydrogym.maia.envs.cavity`.
- `Pinball_2D_Re{30,75,100,150}` - 2D fluidic pinball
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
├── properties_run.toml      # MAIA run configuration (symlink)
├── geometry.toml            # Geometry configuration (symlink)
├── environment_config.yaml  # Environment/actuation configuration (symlink)
├── out_lb/                  # Solver output dir (grid + restart files symlinked inside)
└── stl/                     # STL geometry files (symlink)
```

Then in your SLURM/PBS job script, the MAIA solver will find the workspace via the `properties_run.toml` path.

### Offline Mode (No Internet on Compute Nodes)

**Important:** Environment files are stored on Hugging Face Hub and require internet access to download. Many HPC compute nodes do not have internet access.

For offline HPC use:

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

**Step 2:** Pass this directory as `local_fallback_dir` on the compute nodes:

```python
import hydrogym.maia as maia

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=[2.0, 0.0, 4.0, 0.5, 4.0, -0.5, 6.0, 0.0],  # x, y of each probe
    local_fallback_dir="/scratch/<project>/hf_environments",
)
```

`use_clean_cache` only chooses how downloaded files are cached (`True`, the default: links in `~/.cache/hydrogym`;
`"copy"`: copies; `False`: the paths of the Hugging Face cache); it does not prevent a download.

Or with the workspace preparation script:

```bash
# On compute node (offline) - uses local copy
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./my_run
# (with the Hugging Face cache of the download step: HF_HUB_OFFLINE=1)
```
