---
sidebar_position: 2
---

# MAIA

[m-AIA](https://git.rwth-aachen.de/aia/m-AIA/m-AIA) (Multi-physics Aachen code) is a high-performance CFD framework developed at the Institute of Aerodynamics (AIA) at RWTH Aachen University. It couples finite-volume, discontinuous Galerkin, Lattice Boltzmann (LBM), and level-set methods on Cartesian meshes and targets massively parallel simulations on systems ranging from single workstations to leadership-class GPU clusters. HydroGym's MAIA backend supports both the LBM and structured finite-volume solvers and provides 59 flow-control environments in 2-D and 3-D (55 lattice Boltzmann, 4 structured finite volume).

## Option 1: Docker (recommended)

Pre-built images ship m-AIA together with NekRS, Nek5000, JAX, JAX-Fluids,
MPI, CUDA and HydroGym (Firedrake: CPU image); see
[Container images](./containers) for all details:

| Image | GPUs | Compute capability |
|---|---|---|
| `clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` | T4, RTX 20-50, A100, A30, A10, A4000-A6000, L4, L40, H100, H200, B100, B200 | 7.5-12.0 |
| `clagemann/hydrogym-cpu:1.0.2` | none (CPU only) | - |

```bash
# NVIDIA GPU (x86_64)
docker pull clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
docker run -it --gpus all --ipc=host clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2

# CPU only
docker pull clagemann/hydrogym-cpu:1.0.2
docker run -it --ipc=host clagemann/hydrogym-cpu:1.0.2
```

Inside the container HydroGym and m-AIA are already installed and on the PATH.
The GPU image contains one m-AIA build per GPU architecture
(`/opt/maia/cc*/bin/maia`); `maia` (`/opt/maia/bin/maia`) starts the one that
matches the visible GPU (`maia --list-builds` lists them, `MAIA_GPU_CC=cc80`
selects one by hand). The image uses CUDA 12.9 and needs an NVIDIA driver with
CUDA 12 support, ≥ 525 (tested with drivers 570 and 595). The HydroGym sources
and examples are in `/opt/hydrogym`. The images are x86_64; for NVIDIA GH200
nodes (aarch64) build the Apptainer image from `.devcontainer/release/apptainer/`
(see [Container images](./containers#nvidia-gh200-nodes-aarch64)), and for AMD
GPUs the ROCm image with `.devcontainer/scripts/build_rocm_image.sh`.

## Option 2: Apptainer / Singularity (HPC clusters)

Most HPC clusters do not allow Docker. The Docker images can be converted to the Apptainer SIF format on any machine that has Apptainer available:

```bash
apptainer pull docker://clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
```

This produces `hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif`. To open a shell in the container with GPU access and the current directory mounted as `/workspace`:

```bash
apptainer shell --nv --bind $(pwd):/workspace hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif
```

On NVIDIA GH200 (aarch64) nodes use the GH200 SIF instead: `apptainer pull hydrogym-gh200.sif oras://docker.io/clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2-aarch64` (see [Container images](./containers#nvidia-gh200-nodes-aarch64)).

The image is read-only under Apptainer, so run cases from a bound directory such as `/workspace`.

:::note
On a Slurm cluster, run `apptainer shell` / `apptainer exec` inside an interactive session (`srun --pty bash`) or within an `sbatch` script so that the correct resources (GPUs, memory) are allocated.
:::

m-AIA uses the first GPU it sees: with several m-AIA ranks on one node, give each rank its own GPU through
`CUDA_VISIBLE_DEVICES`. With `mpirun` (Python is local rank 0), launch each m-AIA rank as
`bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK-1)) exec maia properties_run.toml'`. With
`apptainer exec --cleanenv` (recommended when the image's `mpirun` starts all ranks, so that host modules do not
leak in), pass the GPU list as `APPTAINERENV_CUDA_VISIBLE_DEVICES`. See
[Container images](./containers#apptainer) for `srun` launches.

### Port-forwarding for ParaView post-processing

The images do not include ParaView. Start a `pvserver` from a ParaView installation on the cluster (for example `mpirun -np 32 pvserver`); it prints a socket address such as `node042:11111`. On your local workstation, open an SSH tunnel to that port:

```bash
ssh USER@REMOTE.WORKSTATION.ADDRESS -L 11111:node042:11111
```

You can then connect your local ParaView client (the same version as the server) to `localhost:11111` and access the full compute power of the cluster for post-processing.

## Installing the HydroGym Python package

**Using one of the pre-built Docker/Apptainer images from Option 1 or 2?**
HydroGym and its MAIA extras are already installed — skip this section.

**Building m-AIA yourself (Option 3 below), or adding the MAIA extras to
an existing environment that already has a local m-AIA build?** Install
the HydroGym MAIA extras:

```bash
pip install hydrogym[maia]
```

This adds `mpi4py`, `omegaconf`, `einops`, `toml`, `pyarrow` (the feather files of some environment baselines), `stable-baselines3` and `tensorboard` alongside the core package.

`mpi4py` builds against an MPI installation: install the system MPI (e.g. `apt install libopenmpi-dev` or the
cluster's MPI module) before `pip install`, or use the MPI wheels from PyPI (`pip install mpich` or `pip install openmpi`).
m-AIA and `mpi4py` must use the same MPI.

## Option 3: Building m-AIA from source

:::note
HydroGym pins its m-AIA version as the submodule `third_party/m-AIA` (`git.rwth-aachen.de/aia/MAIA/Solver.git`), which
clones without credentials: `git submodule update --init third_party/m-AIA` in a HydroGym clone (see
`.devcontainer/README.md` for the nested submodules m-AIA needs).
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
    : -np 1 maia properties_run.toml
```

## HPC offline usage

Compute nodes on most HPC clusters do not have outbound internet access. Download environment data on a login node before submitting a job:

```bash
# on a machine with internet access (login node, workstation)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='dynamicslab/HydroGym-environments', repo_type='dataset',
                  allow_patterns='Cylinder_2D_Re200/**', local_dir='/scratch/<project>/hf_environments')
"
```

This writes `/scratch/<project>/hf_environments/Cylinder_2D_Re200/`; further environments go into the same directory.

Then on the compute node, point HydroGym at the local copy:

```python
import hydrogym.maia as maia

env = maia.from_hf(
    "Cylinder_2D_Re200",
    probe_locations=[2.0, 0.0, 4.0, 0.5, 4.0, -0.5, 6.0, 0.0],  # x, y of each probe
    local_fallback_dir="/scratch/<project>/hf_environments",
)
```

An environment found in `local_fallback_dir` is used directly, without contacting the Hub. `use_clean_cache` only chooses how downloaded files are cached (`True`, the default: links in `~/.cache/hydrogym`;
`"copy"`: copies; `False`: the paths of the Hugging Face cache); it does not prevent a download.
