<p align="center">
<a rel="nofollow"><img alt="HydroGym Logo" src="docs/_static/imgs/logo.svg"></a>
</p>

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://python.org/"><img alt="Language: Python" src="https://img.shields.io/badge/language-Python-orange.svg"></a>
<a href="https://spdx.org/licenses/MIT.html"><img alt="License MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://join.slack.com/t/hydrogym/shared_invite/zt-27u914dfn-UFq3CkaxiLs8dwZ_fDkBuA"><img alt="Slack" src="https://img.shields.io/badge/slack-hydrogym-brightgreen.svg?logo=slack"></a>
<a href="https://github.com/google/yapf"><img alt="Code style: yapf" src="https://img.shields.io/badge/code%20style-yapf-000000.svg"></a>
</p>

# HydroGym: Reinforcement Learning for Fluid Dynamics

**88 environments | 6 solver backends | 2D & 3D | Ready for RL training**

HydroGym is a comprehensive platform for applying reinforcement learning to fluid dynamics and flow control. With environments ranging from canonical benchmarks to turbulent flows, HydroGym provides a standardized Gymnasium-compatible interface for training RL agents on challenging CFD problems.

> **Paper**: Lagemann, C., et al. (2025). *HydroGym: A reinforcement learning platform for fluid dynamics.* arXiv:2512.17534 [[arxiv]](https://arxiv.org/abs/2512.17534)

## Key Features

- **Diverse Environments**: 88 pre-configured environments across 6 CFD solvers
- **Standard RL Interface**: Gymnasium-compatible API works with Stable-Baselines3, RLlib, and other RL libraries
- **Compute Efficient**: Highly optimized GPU & CPU backends for efficient RL deployment ranging from local workstations to exascale HPC systems
- **Scalable**: MPI-parallelized solvers with distributed RL training support
- **Multiple Backends**: Finite Element (Firedrake), Lattice Boltzmann (MAIA LBM), Finite Volume (MAIA FV), Spectral Element (NEK5000), Fully Differentiable solvers (JAX-Fluids)
- **2D & 3D**: From simple 2D benchmarks to complex 3D turbulent flows (Re up to 400,000)
- **Research-Ready**: Includes checkpoints, observation strategies, and reward formulations managed by a complementary HuggingFace repository

## Quick Start with Docker (Recommended)

**We strongly recommend using our pre-configured Docker containers** for hassle-free setup:

```bash
# For NVIDIA GPUs (CUDA)
docker pull clagemann/maia-cuda-12.8.1:latest

# For AMD GPUs (ROCm)
docker pull clagemann/maia-rocm-6.3.3:latest

# Run container
docker run -it --gpus all clagemann/maia-cuda-12.8.1:latest
```

## Apptainer Usage
Note that in order to interface with a shared cluster, it is necessary to convert the available MAIA docker container to the apptainer format as docker typically isn’t allowed in these shared compute contexts due to its root access requirements. This is done simply by performing an `apptainer pull` command followed by the host of the docker container. For example:

```bash
apptainer pull docker://clagemann/maia-cuda-12.8.1:latest
```

This will convert the docker build file into the necessary .sif format for running with apptainer. To run the maia apptainer (note that the specific command may change with different versions of the container), run `apptainer run` such as below with cuda enabled and bound to the workspace in the .sif environment:

```bash
apptainer pull docker://clagemann/maia-cuda-12.8.1:latest
```

Note that the above likely should be run with the additional necessary resources allocated (i.e. in a slurm environment, running within an interactive session or one can prep an sbatch file to launch the environment and run predefined scripts in the container). In this environment, you can run HydroGym training with access to the full MAIA backend.

### Port-Forwarding with Apptainer
Another important note is that some packages will not be automatically added to the path due to the Apptainer process of generating the .sif file which doesn’t honor the docker Entrypoint setup in the same way. After launching the apptainer environment, it is important to enable the following packages to retain full functionality, particularly when attempting to view results locally by port-forwarding from a remote workstation to a local install of paraview.

```bash
module load GCC/13.3.0 gompi/2024a-SystemCUDA Python/3.12.3-GCCcore-13.3.0 HDF5/1.14.5-gompi-2024a-SystemCUDA OpenMPI/5.0.5-GCC-13.3.0-SystemCUDA FFTW.MPI/3.3.10-gompi-2024a-SystemCUDA PnetCDF/1.12.3-gompi-2024a
-SystemCUDA mpi4py/4.0.1-gompi-2024a-SystemCUDA Eigen/3.4.0-GCCcore-13.3.0

export PATH=$PATH:/home/easybuild/paraview-install/bin/
export LD_LIBRARY_PATH=$EBROOTGCC/lib64:$EBROOTPNETCDF/lib:$EBROOTNETCDF/lib64:$EBROOTHDF5/lib:$LD_LIBRARY_PATH

export PV_PLUGIN_PATH=/home/easybuild/paraview-plugins/build/lib64
export PSP_SCHED_YIELD=1
```
This allows for all post-processing tools to be available when launching a pvserver on a given port (say 11111, for example), which can be run by running a script such as:
```bash
srun -c 24 apptainer exec --nv --bind $(pwd):/workspace  maia-cuda-12.8.1_latest.sif /home/easybuild/paraview-install/bin/pvserver --server-port=11111
```

On a slurm environment, this will give a machine port (say XXXXX) that can be connected to by ssh-ing into the remote workstation and specifying a local port (say 11111) to attach to the given remote port, such as running the below on your local desktop with an install of paraview (must be Paraview v5.13 for current container)
```bash
ssh USER@REMOTE.WORKSTATION.ADDRESS.EDU -L 11111:XXXXX:11111
```

You should now be able to follow the instructions to Connect from your local Paraview Client as the local port will be connected to the remote workstation port which will contain all your maia save files and run the post-processing using the remote workstation’s compute. 

## Available Environments

HydroGym provides **88 environments** across 6 solver backends:

| Solver Backend | Count | Description | Dimensions |
|----------------|-------|-------------|------------|
| **Firedrake** (FEM) | 20 | Canonical flow control benchmarks | 2D |
| **MAIA LBM** | 55 | Lattice Boltzmann method environments | 2D, 3D |
| **MAIA Structured FV** | 8 | High-Reynolds turbulent boundary layers | 3D |
| **NEK5000** | 1 | Spectral element turbulent channel flow | 3D |
| **JAX** | 2 | Differentiable fluid dynamics | 2D, 3D |
| **JAX-Fluids** | 2 | Compressible jet engine control | 2D, 3D |

### Environment Categories

**Canonical Benchmarks** (Low-Mid Re):
- Cylinder wake (Re=100-3900, 2D/3D)
- Rotating cylinder (Re=100-3900, 2D/3D)
- Pinball (Re=30-150, 2D/3D)
- Cavity flow (Re=4140-7500, 2D/3D)
- Backward-facing step (Re=600, 2D)
- Square cylinder (Re=200-3900, 2D/3D)
- Sphere (Re=300-3700, 3D)
- Cube (Re=300-3700, 3D)
- Turbulent channel flow (Re_tau=180, 3D)

**Airfoil Control**:
- NACA0012 steady (Re=100-50000, AOA=12-40°, 2D/3D)
- NACA0012 with gust disturbance (Re=100-50000, 2D/3D)

**High Reynolds Number Flows**:
- Zero-pressure-gradient turbulent boundary layer with jet/surface wave actuation (Re_Tau=1000-5000, 3D)
- DRA2303 airfoil with jet/surface wave actuation (Re=400000, Ma=0.2-0.7, 3D)

**Fully Differentiable Flows**:
- Jet engine thrust vectoring (TVC/TVD, Ma=2.2, 2D/3D)
- Kolmogorov flow (Re=1000, 2D)

See [`existing_environments.yaml`](existing_environments.yaml) for complete list with exact naming conventions.

## Examples

HydroGym includes comprehensive examples for each solver backend:

### Firedrake Examples

See [examples/firedrake/getting_started/](examples/firedrake/getting_started/) for detailed documentation.

```bash
cd examples/firedrake/getting_started

# Test environment interactively
python test_firedrake_env.py --environment cylinder --num-steps 10

# Train with Stable-Baselines3
python train_sb3_firedrake.py --env cylinder --algo PPO --total-timesteps 100000
```

### MAIA Examples

See [examples/maia/getting_started/](examples/maia/getting_started/) for MPMD coupling details.

```bash
cd examples/maia/getting_started

# Prepare workspace (downloads from Hugging Face Hub)
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run

# Run with MPMD execution (1 Python + 1 MAIA process)
cd test_run
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml
```

### NEK5000 Examples

See [examples/nek/getting_started/](examples/nek/getting_started/) for interface patterns.

```bash
cd examples/nek/getting_started/1_nekenv_single

# Test single-agent environment
mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000

# Train with SB3
mpirun -np 1 python train_sb3_nek_direct.py --env TCFmini_3D_Re180 --algo PPO : -np 10 nek5000
```

## Training RL Agents

HydroGym works with standard RL libraries. Example with Stable-Baselines3:

```python
from hydrogym import FlowEnv
import hydrogym.firedrake as hgym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create environment
def make_env():
    env_config = {
        'flow': hgym.Cylinder,
        'flow_config': {'mesh': 'medium', 'Re': 100},
        'solver': hgym.SemiImplicitBDF,
        'solver_config': {'dt': 1e-2},
        'actuation_config': {'num_substeps': 2},
    }
    return FlowEnv(env_config)

# Vectorize and normalize
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Advanced Features

- **Checkpoint management**: Automatic loading from Hugging Face Hub
- **Custom observations**: Force sensors, velocity/pressure/vorticity probes, stress sensors
- **Callbacks**: Checkpointing, logging, Paraview export
- **Stability analysis**: Global stability analysis via SLEPc (Firedrake)
- **Modal decompositions**: DMD, POD via modred (Firedrake)
- **Multi-agent RL**: PettingZoo interface for distributed actuation (NEK5000)

## Documentation

- **Getting Started Guides**: See `examples/[backend]/getting_started/README.md`
- **API Documentation**: [https://hydrogym.readthedocs.io](https://hydrogym.readthedocs.io)
- **Flow Configurations**: [docs/FlowConfigurations.md](docs/FlowConfigurations.md)
- **Paper**: [arXiv:2512.17534](https://arxiv.org/abs/2512.17534)

## Citation

If you use HydroGym in your research, please cite:

```bibtex
@article{lagemann2025hydrogym,
  title={Hydrogym: A reinforcement learning platform for fluid dynamics},
  author={Lagemann, Christian and Mokbel, Sajeda and Gondrum, Miro and R{\"u}ttgers, Mario and Callaham, Jared and Paehler, Ludger and Ahnert, Samuel and Zolman, Nicholas and Lagemann, Kai and Adams, Nikolaus and others},
  journal={arXiv preprint arXiv:2512.17534},
  year={2025}
}
```

## License

HydroGym is released under the MIT License. See [LICENSE](LICENSE) for details.
