---
sidebar_position: 0
---

# Container images

The HydroGym 1.0.2 images have the solver backends built in, together with MPI, CUDA and the Python environment.
The solvers are built into the images and their sources are not needed at run time. The only compilation at run time is
NekRS's just-in-time build of its device kernels and of the case's `udf` on first use (cached afterwards).

| Image | Platform | Backends | Requirements |
|---|---|---|---|
| `clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` | x86_64, NVIDIA GPUs cc75-cc120 (T4, RTX 20-50, A100, A30, A10, A4000-A6000, L4, L40, H100, H200, B100, B200) | m-AIA, NekRS, Nek5000, JAX, JAX-Fluids (no Firedrake) | NVIDIA driver ≥ 525 (CUDA 12) |
| `clagemann/hydrogym-cpu:1.0.2` | x86_64, CPU only | m-AIA, NekRS, Nek5000, JAX, JAX-Fluids, Firedrake | - |
| GH200 Apptainer image (`hydrogym-gh200.sif`) | aarch64, NVIDIA GH200 (cc90) | m-AIA, NekRS, Nek5000, JAX, JAX-Fluids (no Firedrake) | Apptainer ≥ 1.3, NVIDIA driver ≥ 525 |

The GPU image uses CUDA 12.9 for everything (m-AIA, NekRS, the OCCA kernel compiler, JAX), so any driver with CUDA 12
support runs it (CUDA minor-version compatibility; tested with drivers 570 and 595). Firedrake runs on the CPU and is
only in the CPU image: its build needs the MPI Fortran modules, which the GPU image's HPC-X MPI provides for nvfortran
only. For GH200 nodes, pull the Apptainer SIF or build it from its recipe (see
[GH200 nodes](#nvidia-gh200-nodes-aarch64) below).

## Start a container

```bash
# NVIDIA GPU
docker pull clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
docker run -it --gpus all --ipc=host clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2

# CPU only (also the image for Firedrake)
docker pull clagemann/hydrogym-cpu:1.0.2
docker run -it --ipc=host clagemann/hydrogym-cpu:1.0.2
```

To work in your own directory, mount it, e.g. `-v $(pwd):/workspace -w /workspace`. The `release-gpu` and
`release-cpu` devcontainer configurations in `.devcontainer/` open a HydroGym checkout (at `/workspace`) in these
images.

## What is inside

| Path | Contents |
|---|---|
| `/opt/venvs/hydrogym` | The HydroGym Python environment, on the `PATH` (`python` is this environment): `hydrogym[maia,nek,nekrs,jax,jaxfluids]`, torch (CPU build), JAX (CUDA 12 build in the GPU images), JAX-Fluids, at the versions of `uv.lock`. Also reachable as `/opt/venvs/nekrs`. |
| `/opt/venvs/firedrake` | CPU image only: Firedrake and PETSc with `hydrogym[firedrake]`, Stable-Baselines3, TensorBoard and torch (CPU), at the versions of `/opt/venvs/hydrogym` (`source /opt/venvs/firedrake/bin/activate`). Firedrake's kernel caches: in the environment for root, in `~/.cache/firedrake` for other users (`/tmp/firedrake-cache-uid<uid>` without a writable home directory), or where `PYOP2_CACHE_DIR` and `FIREDRAKE_TSFC_KERNEL_CACHE_DIR` point. |
| `/opt/hydrogym` | The HydroGym sources and examples, installed editable in both environments. `/opt/hydrogym/IMAGE_VERSION` holds the HydroGym version and revision of the image. |
| `/opt/maia/bin/maia` | `maia` on the `PATH`. In the GPU images a wrapper that starts the m-AIA build for the visible GPU, `/opt/maia/cc*/bin/maia` (see below); the CPU image has a GCC build. |
| `/opt/nekrs` | NekRS 26.0 (`NEKRS_HOME`), with `nekrs` and `nekrs-mpmd` on the `PATH`. GPU images: OCCA CUDA, GPU HYPRE, GPU-aware MPI; CPU image: the serial OCCA backend (`NEKRS_BACKEND=SERIAL`). |
| `/opt/nek5000/cases` | The Nek5000 case executables, `/opt/nek5000/cases/<case>/nek5000`. |

One MPI serves the Python controller and every solver of an MPMD launch: HPC-X Open MPI 5 (CUDA-aware) in the GPU
images, the Ubuntu Open MPI in the CPU image. `mpirun` is on the `PATH`. In the GPU images
`XLA_PYTHON_CLIENT_PREALLOCATE=false` is set, so a JAX process does not reserve most of a GPU that a solver shares.

### m-AIA builds

The GPU image holds one single-architecture m-AIA build per GPU generation (cc75, cc80, cc89, cc90, cc100, cc120).
The `maia` wrapper starts the one for the visible GPU: the exact compute capability, else the build of the same major
version below it (e.g. cc80 for a cc86 GPU).

```bash
maia --list-builds            # builds in this image
MAIA_GPU_CC=cc80 maia ...     # use that build, without detection
```

The wrapper looks at the GPUs in `CUDA_VISIBLE_DEVICES`. If they have different compute capabilities, restrict
`CUDA_VISIBLE_DEVICES` or set `MAIA_GPU_CC`.

## Running the examples

The examples are in `/opt/hydrogym/examples`. The `run_*_docker.sh` scripts in the `getting_started/` directories
are plain launch scripts: they activate the image's environment (`/opt/venvs/hydrogym`, or `/opt/venvs/firedrake`
for the Firedrake examples, which need the CPU image), use the installed solvers and need no module system; outside
the images, activate your own environment first. The examples write into their own directories, and
`/opt/hydrogym/examples` is writable only for root (`docker run` without `--user`); as any other user, and always
under Apptainer (read-only image), copy the examples first and run them from the copy:
`cp -r /opt/hydrogym/examples ~/hydrogym-examples`.

## Operational notes

**Case data.** Environments download their configuration and initial fields from the
[Hugging Face Hub](https://huggingface.co/datasets/dynamicslab/HydroGym-environments) into `~/.cache` on first use.
Compute nodes without internet access need the data downloaded first, on a machine with internet (e.g. the login
node), into a cache directory the compute nodes can read. Then set `HF_HUB_OFFLINE=1` for the run. With Docker, mount a
host directory as the cache (`-v $HOME/.cache/huggingface:/root/.cache/huggingface`) so it survives the container;
with Apptainer, `--home <dir>` puts the cache in `<dir>/.cache`. See also [HPC offline usage](./maia#hpc-offline-usage).

**Non-root users and GPU access.** When you start the container as another user (`docker run --user ...`) on a host
whose `/dev/nvidia*` devices belong to a group (e.g. `root:video`, mode 0660), add that group, or the container sees
no GPU (`maia: no NVIDIA GPU visible`):

```bash
docker run -it --gpus all --ipc=host --user $(id -u):$(id -g) \
    --group-add $(getent group video | cut -d: -f3) clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
```

**Nek5000: more ranks than cores.** The Nek5000 examples start 1 Python and 10 Nek5000 ranks. Open MPI 5 counts slots
by physical core, so on a machine with 10 cores (or fewer) pass `--oversubscribe` to `mpirun`, e.g.
`mpirun --oversubscribe -np 1 python ... : -np 10 nek5000`. The Nek5000 example scripts take extra `mpirun` options
from `MPIRUN_ARGS` (`MPIRUN_ARGS=--oversubscribe ./run_nekenv_docker.sh`).

**NekRS: several ranks on one GPU.** NekRS gives each local rank its own GPU. To run several NekRS ranks on a single
GPU, add `--device-id 0` to the `nekrs`/`nekrs-mpmd` command line. For the NekRS `run.sh` scripts, point
`NEKRS_MPMD` at a two-line wrapper:

```bash
printf '#!/bin/sh\nexec nekrs-mpmd "$@" --device-id 0\n' > /tmp/nekrs-mpmd-dev0 && chmod +x /tmp/nekrs-mpmd-dev0
NEKRS_MPMD=/tmp/nekrs-mpmd-dev0 NP=2 RUN=/tmp/run ./run.sh --policy oppo --steps 3
```

With several GPUs and more NekRS ranks than GPUs per node, `examples/nekrs/gpu_share.sh` spreads the ranks over the
GPUs (rank mod number of GPUs, through `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`) and adds `--device-id 0`;
the wing's `run.sh` uses it with `GPU_SHARE=1`.

**NekRS: GPU-aware MPI and UCX.** GPU-aware MPI needs UCX's CUDA transports. If you restrict `UCX_TLS`, keep
`cuda_copy` (and `cuda_ipc` for several GPUs), e.g. `UCX_TLS=shm,self,tcp,cuda_copy`.

**NekRS kernel cache.** The first NekRS run JIT-compiles the solver kernels for the GPU (several minutes) into
`NEKRS_CACHE_DIR`; keep that directory between runs, and in a writable place under Apptainer.

## Apptainer

Most shared clusters run containers with Apptainer. The x86_64 Docker images convert with `apptainer pull`:

```bash
apptainer pull docker://clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2   # -> hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif
apptainer exec --nv hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif maia --list-builds
```

`--nv` brings in the host's GPU driver. Run inside a Slurm allocation (`srun --pty bash` or an `sbatch` script), so
that the GPUs are yours. The image is read-only: run cases from a bound, writable directory
(`--bind $(pwd):/workspace`), and point the NekRS kernel cache there
(`export NEKRS_CACHE_DIR=/workspace/.cache/nekrs`).

- **Host environment.** `apptainer exec` passes the host environment in. Variables of loaded host modules
  (`PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, compiler settings) then reach the container; a host compiler can, for
  example, build the NekRS udf against the host libraries, which fails. Use `--cleanenv` when the image's own
  `mpirun` starts all ranks, and pass what the run needs with `APPTAINERENV_<NAME>` variables, e.g.
  `APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"` for the GPU list Slurm sets
  (`--env CUDA_VISIBLE_DEVICES=0,1,2,3` would split the comma-separated list). Do not use `--cleanenv` when `srun`
  starts every rank in its own container: those ranks need the `PMIX_*`/`SLURM_*` variables.
- **srun.** `srun` can start every rank in its own container, also across nodes. The image's Open MPI 5 needs a PMIx
  plugin of the cluster's Slurm: check `srun --mpi=list` (e.g. `pmix`, or `pspmix` with ParaStation PMIx). Tested on
  NVIDIA GH200 (ARM, aarch64) nodes with a PMIx plugin.
- **m-AIA ranks and GPUs.** m-AIA uses the first GPU it sees, so give each m-AIA rank its own GPU through
  `CUDA_VISIBLE_DEVICES`. With `mpirun` (Python is local rank 0), launch each m-AIA rank as
  `bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK-1)) exec maia properties_run.toml'`; with
  `srun --multi-prog`, use `%o` in the configuration file (see [`.devcontainer/README.md`](https://github.com/dynamicslab/hydrogym/blob/main/.devcontainer/README.md#nvidia-gh200-aarch64-apptainer-image)).
- **NekRS case directories.** NekRS writes into the case directory. The channel example includes HydroGym's udf
  header by a relative path, so bind a writable copy over it:
  `--bind <copy>:/opt/hydrogym/examples/nekrs/channel`.

## NVIDIA GH200 nodes (aarch64)

The Docker images are x86_64 only. For GH200 nodes (Grace CPU, Hopper GPU) the image is published as an Apptainer SIF
(an OCI artifact on Docker Hub):

```bash
apptainer pull hydrogym-gh200.sif oras://docker.io/clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2-aarch64
apptainer exec --nv hydrogym-gh200.sif maia --list-builds
```

To build it yourself: `.devcontainer/release/apptainer/*.def`
build the same stack as an Apptainer SIF: m-AIA for cc90 behind the same `maia` wrapper, NekRS 26.0 (CUDA, GPU HYPRE,
GPU-aware MPI), the Nek5000 case executables and `/opt/venvs/hydrogym` with JAX (CUDA 12) and JAX-Fluids, at the same
paths as in the x86_64 images. It has no Firedrake.

The build runs on an aarch64 machine with Apptainer ≥ 1.3 and `--fakeroot` (e.g. a GH200 cluster's login node); the
solver compiles can run as batch jobs:

```bash
# On the machine with the HydroGym checkout (solver submodules initialised): the build context
STAGE_DIR=ctx STAGE_ONLY=1 HYDROGYM_REF=v1.0.2 .devcontainer/scripts/build_release_images.sh
git rev-parse --short=9 v1.0.2 > ctx/HYDROGYM_REVISION
# On the aarch64 machine, with ctx copied there
CTX=$PWD/ctx WORK=$PWD BUILD_ROOT=/tmp/hg-build ctx/build/scripts/build_apptainer_arm64.sh all
ctx/build/scripts/test_apptainer_image.sh hydrogym-gh200.sif prepare    # case data (needs internet)
ctx/build/scripts/test_apptainer_image.sh hydrogym-gh200.sif            # on a GH200 node, in an allocation
```

The Apptainer notes above apply. In addition, the NekRS NACA0012 wing needs at least 8 NekRS ranks on aarch64 nodes:
with fewer ranks the static Nek5000 arrays exceed the aarch64 relocation range. With fewer GPUs per node, use
`examples/nekrs/gpu_share.sh` (`GPU_SHARE=1` in the wing's `run.sh`; `run_slurm.sbatch` sets it when `NP` exceeds the
GPUs of the node), e.g. 8 ranks on the 4 GPUs of one node.

## AMD GPUs

`.devcontainer/scripts/build_rocm_image.sh` builds a self-contained ROCm image (default target gfx942, MI300A/MI300X,
ROCm 7.2.4) with NekRS (HIP), m-AIA (hipstdpar), Nek5000, Firedrake, JAX and JAX-Fluids. See
[NekRS](./nekrs) and [`.devcontainer/README.md`](https://github.com/dynamicslab/hydrogym/blob/main/.devcontainer/README.md#amd-gpus-self-contained-rocm-image).

## Building the images

`.devcontainer/scripts/build_release_images.sh` builds the x86_64 images from a committed revision (the solver
submodules must be initialised), and `.devcontainer/scripts/test_release_image.sh <image>` runs real solver steps of
every backend in an image (`GPU=1` for the GPU tests). Details, build options and the GH200 build steps are in
[`.devcontainer/README.md`](https://github.com/dynamicslab/hydrogym/blob/main/.devcontainer/README.md#release-images-self-contained).
