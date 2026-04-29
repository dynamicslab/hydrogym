---
sidebar_position: 2
---

# Quickstart

## Start with Docker (Recommended)

For machines with NVIDIA GPUs (CUDA):

```bash
docker pull clagemann/maia-cuda-12.8.1:latest
```

For AMD GPUs (ROCm):

```bash
docker pull clagemann/maia-rocm-6.3.3:latest
```

Run container:

```bash
docker run -it --gpus all clagemann/maia-cuda-12.8.1:latest
```

## Start with Apptainer

Note that in order to interface with a shared cluster, it is necessary to convert the available MAIA docker container to the apptainer format, using the `apptainer pull` command followed by the host of the docker container:

```bash
apptainer pull docker://clagemann/maia-cuda-12.8.1:latest
```

This will convert the docker build file into the necessary `.sif` format for running with apptainer. To run the MAIA apptainer, run `apptainer run` such as below with CUDA enabled and bound to the workspace in the `.sif` environment:

```bash
apptainer run --nv --bind $(pwd):/workspace maia-cuda-12.8.1_latest.sif
```

Note that the above likely should be run with the additional necessary resources allocated (i.e. in a Slurm environment, launch the apptainer within an interactive session or within an `sbatch` file). Once the apptainer is launched, you can run HydroGym training with access to the full MAIA backend.

### Port-forwarding with Apptainer

Some packages will not be automatically added to the path due to the Apptainer process of generating the `.sif` file. However, functionality such as launching a `pvserver` can be done simply by launching the source file:

```bash
bash /home/easybuild/paraview-plugins/pvServerLaunch2024.sh 32
```

On a Slurm environment, this will give a machine socket (e.g. `xxxxx:11111`) that can be connected to by SSH-ing into the remote workstation and specifying a local port to attach to the given remote socket. Run the following on your local desktop with an install of ParaView (must be ParaView v5.13 for current container):

```bash
ssh USER@REMOTE.WORKSTATION.ADDRESS.EDU -L 11111:XXXXX:11111
```

You should now be able to connect from your local ParaView client — the local port will be connected to the remote workstation port, allowing post-processing using the remote workstation’s compute with all your MAIA save files.

## Run environments

HydroGym provides **88 environments** across 6 solver backends:

| Solver backend | Count | Description | Dimensions |
| --- | ---: | --- | --- |
| **Firedrake** `FEM` | 20 | Canonical flow control benchmarks | 2D |
| **MAIA** `LBM` | 55 | Lattice Boltzmann method environments | 2D, 3D |
| **MAIA** `STRUCTURED FV` | 8 | High-Reynolds turbulent boundary layers | 3D |
| **NEK5000** `SEM` | 1 | Spectral element turbulent channel flow | 3D |
| **JAX** `SEM, FD` | 2 | Differentiable fluid dynamics | 2D, 3D |
| **JAX-Fluids** `FVM` | 2 | Compressible jet engine control | 2D, 3D |

HydroGym interfaces with [Hugging Face](https://huggingface.co/datasets/dynamicslab/HydroGym-environments) to easily set up fluid environments. Hugging Face currently contains several pre-configured environments that can be loaded via `.from_hf()`. For instance, MAIA environments can be loaded as follows:

```python
import hydrogym.maia as maia

env = maia.from_hf("Cylinder_2D_Re200")
```

Several pre-configured environments are available to load from Hugging Face. A list of environments and their exact naming conventions can be found [in this breakdown](https://github.com/dynamicslab/hydrogym/blob/main/existing_environments.yaml). For further detail on customizing environments and running RL scripts, jump to the [RL training](#rl-training) section.

### MAIA

MAIA: high-performance CFD for large-scale simulations. Built on RWTH Aachen’s m-AIA framework, this backend enables massive parallel simulations with efficient CPU/GPU acceleration using the Lattice Boltzmann method.

- [Setup and test MAIA environment](installation/maia)
- [Train PPO on a cylinder with MAIA](installation/maia)
- [Offline usage and Hugging Face interface](installation/maia)

### Firedrake

Firedrake is an automated system for the solution of partial differential equations using the finite element method (FEM).

- [Configuration details for Firedrake environments](installation/firedrake)
- [Run a 2D flow around a cylinder with Firedrake](installation/firedrake)
- [Checkpointing and using callbacks with Firedrake](installation/firedrake)

### JAX

Fully differentiable spectral solvers for turbulent flows in JAX. Easy to set up and run examples — no HPC cluster needed. The simulations and training scripts benefit from GPU acceleration with JAX functionality. Examples are also feasible to run on a personal laptop with no GPU access.

- [Setup and test Kolmogorov environment](installation/jax)
- [Visualize results for Kolmogorov flow with Jupyter notebook](installation/jax)
- [Set up and run 3D turbulent channel flow environment with basic actuation](installation/jax)

### NEK5000

Nek5000 is a computational fluid dynamics code that employs the spectral-element method (SEM) to simulate unsteady incompressible fluid flow. It can handle general two- and three-dimensional domains described by isoparametric quad or hex elements.

- [Load pre-configured environments with Hugging Face interface](installation/nek5000)
- [Run 3D channel flow with single PPO agent](installation/nek5000)
- [Run basic control with environment](installation/nek5000)

## Interactive Test Scripts

HydroGym also contains a number of interactive test scripts that can easily be run. For instance, this is a script run with 1 Python + 1 MAIA process:

```bash
mpirun -np 1 python test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml
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

