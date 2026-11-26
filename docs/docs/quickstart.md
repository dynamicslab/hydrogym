---
sidebar_position: 2
---

# Quickstart

## Start with Docker (Recommended)

Pre-built images have the HydroGym backends built in, with MPI, CUDA and the
Python environment already installed and on the `PATH`; see
[Container images](./installation/containers) for the details.

```bash
# NVIDIA GPUs (x86_64, cc75-cc120: T4, RTX 20-50, A100, A6000, L40, H100, H200, B200, ...;
# NVIDIA driver >= 525): m-AIA, NekRS, Nek5000, JAX, JAX-Fluids
docker pull clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
docker run -it --gpus all --ipc=host clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2

# CPU only (x86_64): all backends, including Firedrake
docker pull clagemann/hydrogym-cpu:1.0.2
docker run -it --ipc=host clagemann/hydrogym-cpu:1.0.2
```

Inside the images, `python` is the HydroGym environment (`/opt/venvs/hydrogym`,
with JAX and JAX-Fluids), `maia`, `nekrs`, `nekrs-mpmd` and `mpirun` are on the
`PATH`, the Nek5000 case executables are in `/opt/nek5000/cases/<case>/nek5000`,
and in the CPU image Firedrake has its own environment
(`source /opt/venvs/firedrake/bin/activate`). The HydroGym sources and examples
are in `/opt/hydrogym`. In the GPU image, `maia` starts the m-AIA build for the
visible GPU (`maia --list-builds` lists them, `MAIA_GPU_CC=cc80` selects one).

For NVIDIA GH200 nodes (aarch64) there is an Apptainer image built from the
recipe in `.devcontainer/release/apptainer/`, and for AMD GPUs a ROCm image
built with `.devcontainer/scripts/build_rocm_image.sh`; see
[Container images](./installation/containers).

## Start with Apptainer

Most shared clusters run containers with Apptainer. Convert the Docker image with `apptainer pull`:

```bash
apptainer pull docker://clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
```

This writes `hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif`. Open a shell in it with GPU access (`--nv`) and your working directory bound to `/workspace`:

```bash
apptainer shell --nv --bind $(pwd):/workspace hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif
```

On NVIDIA GH200 (aarch64) nodes pull the GH200 image instead: `apptainer pull hydrogym-gh200.sif oras://docker.io/clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2-aarch64`.

The image is read-only under Apptainer: run cases from `/workspace` (or another bound directory), and point NekRS's kernel cache there (`export NEKRS_CACHE_DIR=/workspace/.cache/nekrs`). Run the container with the resources allocated (in a Slurm environment, inside an interactive session or an `sbatch` script). Compute nodes often have no internet access: download the case data first (see [HPC offline usage](./installation/maia#hpc-offline-usage)). [Container images](./installation/containers#apptainer) lists what else to watch for (`--cleanenv`, GPU lists, `srun`).

### Post-processing with ParaView

The images do not include ParaView. To post-process on the cluster, start a `pvserver` from a ParaView installation there (e.g. `mpirun -np 32 pvserver`), which prints a socket such as `node042:11111`, and open an SSH tunnel from your desktop:

```bash
ssh USER@REMOTE.WORKSTATION.ADDRESS.EDU -L 11111:node042:11111
```

You can then connect your local ParaView client (the same version as the server) to `localhost:11111`, using the remote workstation's compute with all your MAIA save files.

## Run environments

HydroGym provides **89 environments** across 7 solver backends (counts are
the environment configurations currently published on the
[Hugging Face Hub](https://huggingface.co/datasets/dynamicslab/HydroGym-environments)):

| Solver backend | Count | Description | Dimensions |
| --- | ---: | --- | --- |
| **Firedrake** `FEM` | 20 | Canonical flow control benchmarks | 2D |
| **MAIA** `LBM` | 55 | Lattice Boltzmann method environments | 2D, 3D |
| **MAIA** `STRUCTURED FV` | 4 | High-Reynolds turbulent boundary layers | 3D |
| **NEK5000** `SEM` | 4 | Spectral element turbulent channel/airfoil flow | 3D |
| **NekRS** `SEM`, GPU | 2 | The channel and the wing of NEK5000, on GPUs | 3D |
| **JAX** `SEM, FD` | 2 | Differentiable fluid dynamics | 2D, 3D |
| **JAX-Fluids** `FVM` | 4 | Compressible shock vector control | 2D, 3D |

### The standard entry point: `gym.make()`

The recommended way to start any environment is gymnasium's `gym.make()`
— one call, the same shape regardless of backend:

```python
import gymnasium as gym
import hydrogym.registration

env = gym.make("hydrogym/Cylinder-v0")  # Firedrake
env = gym.make("hydrogym-maia/Cylinder_2D_Re200-v0")  # MAIA (real MPMD underneath)
env = gym.make("hydrogym-nek/TCFmini_3D_Re180-v0", nproc=10)  # Nek5000 (real MPMD underneath)
env = gym.make("hydrogym-jaxfluids/Nozzle2D-v0")  # JAX-Fluids

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

MAIA and Nek5000 must still be launched as MPMD jobs (`mpirun -np 1
python script.py : -np N maia/nek5000 ...`) — `gym.make()` constructs and
wires up the coupling once inside that job, it does not remove the need
for it. Two backends are not reachable through `gym.make()`. JAX
implements a functional/JIT (`gymnax`-style) contract for
`jax.jit`/`jax.vmap`/`jax.lax.scan` compatibility, incompatible with
`gymnasium.Env`'s `self`-mutating shape — see `examples/jax/`. NekRS
environments (`hydrogym.nekrs.NekRSFlowEnv`, a `gymnasium.Env`) are set up
from a YAML file and launched as MPMD jobs by the scripts of
`examples/nekrs/` — see [NekRS](./installation/nekrs).

HydroGym interfaces with [Hugging Face](https://huggingface.co/datasets/dynamicslab/HydroGym-environments) to easily set up fluid environments; `gym.make()` resolves environment data from there automatically. Every backend's lower-level, native construction path (used internally by `gym.make()`, and fully supported directly) also remains available — for instance, MAIA environments can be loaded via `.from_hf()`:

```python
import hydrogym.maia as maia

env = maia.from_hf("Cylinder_2D_Re200")
```

Several pre-configured environments are available to load from Hugging Face. The full list of environments and their exact naming conventions is the set of top-level folders on [the Hub dataset](https://huggingface.co/datasets/dynamicslab/HydroGym-environments) — each top-level directory is one environment, named `{Flow}_{dim}_{Re}[_{variant}]` (with a `_FD` suffix for the Firedrake environments). For further detail on customizing environments and running RL scripts, jump to the [RL training](#rl-training) section.

### MAIA

MAIA: high-performance CFD for large-scale simulations. Built on RWTH Aachen’s m-AIA framework, this backend enables massive parallel simulations with efficient CPU/GPU acceleration using the Lattice Boltzmann method.

- [Setup and test MAIA environment](installation/maia)
- [Train PPO on a cylinder with MAIA](examples/maia/getting_started)
- [Offline usage and Hugging Face interface](installation/maia#hpc-offline-usage)

### Firedrake

Firedrake is an automated system for the solution of partial differential equations using the finite element method (FEM).

- [Configuration details for Firedrake environments](examples/firedrake/getting_started#configuration-reference)
- [Run a 2D flow around a cylinder with Firedrake](examples/firedrake/cylinder)
- [Checkpointing with Firedrake](examples/firedrake/getting_started#automatic-checkpoint-loading)

### JAX

Fully differentiable spectral solvers for turbulent flows in JAX. Easy to set up and run examples — no HPC cluster needed. The simulations and training scripts benefit from GPU acceleration with JAX functionality. Examples are also feasible to run on a personal laptop with no GPU access.

- [Setup and test Kolmogorov environment](examples/jax/getting_started)
- [Visualize results for Kolmogorov flow](examples/jax/kolmogorov)
- [Set up and run 3D turbulent channel flow environment with basic actuation](examples/jax/channel)

### NEK5000

Nek5000 is a computational fluid dynamics code that employs the spectral-element method (SEM) to simulate unsteady incompressible fluid flow. It can handle general two- and three-dimensional domains described by isoparametric quad or hex elements.

- [Load pre-configured environments with Hugging Face interface](examples/nek5000/from_hf)
- [Run 3D channel flow with a single agent](examples/nek5000/single_environment)
- [Run basic control with environment](examples/nek5000/hydrogym_control)

### NekRS

NekRS is the GPU version of Nek5000. HydroGym runs the turbulent channel and the NACA0012 wing of the Nek5000 backend in it,
with probes, actuators and reward defined in a YAML file.

- [Install and run the NekRS channel and wing](installation/nekrs)
- [Validation of the NekRS NACA0012 wing](examples/nekrs/wing_validation)

## Interactive Test Scripts

HydroGym also contains a number of interactive test scripts that can easily be run. For instance, this is a script run with 1 Python + 1 MAIA process:

```bash
cd examples/maia/getting_started
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run   # workspace with properties_run.toml
cd test_run
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties_run.toml
```

## RL training

The MAIA environments can be customised by passing arguments to the `.from_hf()` function. For instance, an SB3 script with the Cylinder 2D environment and custom probe locations:

### Basic training loop

The basic training loop takes the following shape:

```python
# See examples/maia/getting_started/train_sb3_maia.py for full implementation
import numpy as np
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

