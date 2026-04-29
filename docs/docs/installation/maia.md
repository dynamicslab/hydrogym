---
sidebar_position: 2
---

# MAIA

[m-AIA](https://git.rwth-aachen.de/aia/m-AIA/m-AIA) (Multi-physics Aachen code — formerly ZFS) is a high-performance CFD framework developed at the Institute of Aerodynamics (AIA) at RWTH Aachen University. It couples finite-volume, discontinuous Galerkin, Lattice Boltzmann (LBM), and level-set methods on Cartesian meshes and targets massively parallel simulations on systems ranging from single workstations to leadership-class GPU clusters. HydroGym's MAIA backend supports both the LBM and structured finite-volume solvers and provides over 60 flow-control environments in 2-D and 3-D.

## Option 1: Docker (recommended)

Pre-built images ship m-AIA together with all MPI, CUDA/ROCm, and HydroGym dependencies:

```bash
# NVIDIA GPU (CUDA 12)
docker pull clagemann/maia-cuda-12.8.1:latest
docker run -it --gpus all clagemann/maia-cuda-12.8.1:latest

# AMD GPU (ROCm 6.3)
docker pull clagemann/maia-rocm-6.3.3:latest
docker run -it clagemann/maia-rocm-6.3.3:latest
```

Inside the container HydroGym and m-AIA are already installed and on the PATH.

## Option 2: Apptainer / Singularity (HPC clusters)

Most HPC clusters do not allow Docker. The MAIA Docker images can be converted to the Apptainer SIF format on any machine that has Docker and Apptainer available:

```bash
apptainer pull docker://clagemann/maia-cuda-12.8.1:latest
```

This produces `maia-cuda-12.8.1_latest.sif`. To run the container with GPU access and the current directory mounted as `/workspace`:

```bash
apptainer run --nv --bind $(pwd):/workspace maia-cuda-12.8.1_latest.sif
```

:::note
On a Slurm cluster, run `apptainer run` inside an interactive session (`srun --pty bash`) or within an `sbatch` script so that the correct resources (GPUs, memory) are allocated.
:::

### Port-forwarding for ParaView post-processing

The container includes a ParaView server launcher. Start it inside the Apptainer environment:

```bash
bash /home/easybuild/paraview-plugins/pvServerLaunch2024.sh 32
```

This prints a socket address such as `node042:11111`. On your local workstation, open an SSH tunnel to that port:

```bash
ssh USER@REMOTE.WORKSTATION.ADDRESS -L 11111:node042:11111
```

You can then connect your local ParaView client (version 5.13) to `localhost:11111` and access the full compute power of the cluster for post-processing. 

## Installing the HydroGym Python package

Once m-AIA is available in your environment (either via Docker or a native build), install the HydroGym MAIA extras:

```bash
pip install hydrogym[maia]
```

This adds `mpi4py`, `omegaconf`, `einops`, and `toml` alongside the core package.

## Option 3: Building m-AIA from source

:::warning
Access to the m-AIA source repository requires registration with the AIA group at RWTH Aachen. Contact the maintainers via the [GitLab repository](https://git.rwth-aachen.de/aia/m-AIA/m-AIA) for access.
:::

m-AIA is written in C++ and built with CMake. A representative set of prerequisites includes:

| Prerequisite | Notes |
|---|---|
| C++17-capable compiler | GCC ≥ 9 or Clang ≥ 10 recommended |
| CMake ≥ 3.18 | |
| MPI | OpenMPI ≥ 4 or MPICH ≥ 3 |
| CUDA Toolkit ≥ 12 | GPU builds only |
| ROCm ≥ 6 | AMD GPU builds only |

Refer to the [m-AIA documentation](https://git.rwth-aachen.de/aia/m-AIA/m-AIA) and the `CONTRIBUTING.md` file in the repository for the full build procedure.

## Quick functional test

After starting the container and installing HydroGym, run a two-process smoke test (one Python controller + one m-AIA solver process):

```bash
# Download environment data (login node or machine with internet)
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./test_run

# Run the test (requires MPI)
cd test_run
mpirun -np 1 python ../test_maia_env.py --environment Cylinder_2D_Re200 \
    : -np 1 maia properties.toml
```

## HPC offline usage

Compute nodes on most HPC clusters do not have outbound internet access. Download environment data on a login node before submitting a job:

```bash
# On the login node (internet access available)
python -c "
from hydrogym.data_manager import HFDataManager
dm = HFDataManager(repo_id='dynamicslab/HydroGym-environments', use_clean_cache=False)
env_path = dm.get_environment_path('Cylinder_2D_Re200')
print(f'Downloaded to: {env_path}')
"

# Copy to shared storage accessible from compute nodes
cp -r ~/.cache/huggingface/hub/models--dynamicslab--HydroGym-environments \
    /scratch/$USER/hf_environments/
```

Then on the compute node, point HydroGym at the local copy:

```python
import hydrogym.maia as maia

env = maia.from_hf(
    'Cylinder_2D_Re200',
    probe_locations=[...],
    local_fallback_dir='/scratch/$USER/hf_environments',
    use_clean_cache=False,
)
```
