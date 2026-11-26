# HydroGym devcontainer

Modular devcontainer setup: a shared base image (NVHPC 26.5 + CUDA 13.2,
apt OpenMPI, the parallel HDF5/netCDF/FFTW stacks) plus per-solver
"features" you compose per config (`maia-gpu`, `maia-cpu`, `nek5000`,
`petsc`, `firedrake`, `python-ml`, `hydrogym-gpu`, `jax-fluids`). Start
from `devcontainer-template.json` to see all available features and
example combinations, or open one of the pinned `*.devcontainer.json`
configs directly (`devcontainer.json` is the default: MAIA GPU only).

## You must set your GPU architecture before building

The `maia-gpu` feature compiles MAIA with NVHPC's stdpar for a **specific**
GPU architecture (`-gpu=ccXX`) — it is not a portable binary. Every config
in this directory ships with `pstlPreset` hardcoded to `"ada"` (an RTX
40-series / Ada Lovelace card) as an example default. **If your GPU isn't Ada Lovelace, you
must change this before building**, or MAIA will compile for the wrong
architecture and fail to run (or run without GPU acceleration) on your
hardware.

The one place that matters is the `pstlPreset` option under
`features["./features/maia-gpu"]` in whichever `*.devcontainer.json` you're
using, e.g.:

```jsonc
"features": {
  "./features/maia-gpu": {
    "pstlPreset": "ampere",   // <- change this to match your GPU
    "buildType": "production"
  }
}
```

`build.args.PSTL_PRESET` (in the same files, under `"build"`) is **not**
load-bearing — `Dockerfile.base` declares no matching `ARG`, so it's
silently ignored by Docker. Don't rely on editing it; only the feature
option above actually reaches `configure.py --enable-pstl`.

Valid `pstlPreset` values and the GPUs they correspond to:

| Preset               | Compute capability | Example GPUs                  |
|-----------------------|--------------------|--------------------------------|
| `volta`               | cc70               | V100, Titan V                  |
| `turing`               | cc75               | RTX 20-series, GTX 16-series, T4 |
| `ampere`               | cc80 / cc86        | A100, RTX 30-series             |
| `ada`                  | cc89               | RTX 40-series, L4, L40          |
| `hopper`               | cc90               | H100, H200                      |
| `blackwell`            | cc100              | B100, B200 (datacenter)         |
| `blackwell_consumer`   | cc120              | RTX 50-series                   |
| `multicore`            | —                  | CPU-only stdpar fallback, no GPU |
| `HOST`                 | —                  | let `nvc++` auto-detect the build machine's GPU at compile time |

Not sure which one you have? Run `nvidia-smi --query-gpu=name,compute_cap
--format=csv` on the host before building.

## `third_party/m-AIA` submodule

`third_party/m-AIA` is the m-AIA solver (`git.rwth-aachen.de/aia/MAIA/Solver.git`,
branch `wipmaiaml`) with the RL features HydroGym's flow control cases depend
on: the LB jet-actuation boundary conditions (BC `2007`/`2008`) and the
MPMD-based external flow-control channel. The branch is public: a plain
`git clone --recurse-submodules` (or `git submodule update --init
--recursive` after a plain clone) needs no credentials.

One nested submodule of m-AIA, `include/PyJacReactionMechanisms`, is not
publicly accessible and is **not** needed unless
`ENABLE_PYJAC` is on — `maia-gpu`/`maia-cpu`'s `postCreateCommand.sh`
deliberately excludes it from MAIA's own submodule init (see
`include/Eigen include/cantera include/doctest include/hypre include/mfem
include/sundials` in that script), so a blanket
`git submodule update --init --recursive` from the top of this repo will
fail at that one path — expected, not a sign anything else is broken; use
the scoped commands `postCreateCommand.sh` itself uses instead if you're
initializing submodules by hand.

The submodule also doesn't ship `auxiliary/hosts/DEVCONTAINER.cmake` (the
NVHPC/CUDA/library-path host config this container needs), since a host
config is environment-specific and is not part of m-AIA. `maia-gpu/postCreateCommand.sh` materializes
it at container-start from `.devcontainer/base/DEVCONTAINER.cmake` if the
submodule doesn't already have one, and syncs its `PSTL` value to whatever
`pstlPreset` you set — a no-op if the submodule already contains the file.

## Workspace layout

`workspaceMount` binds the repo itself (`${localWorkspaceFolder}`) to
`/workspace` in the container — no host-specific paths to edit. Every
solver's source lives inside this one clone: `third_party/m-AIA` (MAIA,
see caveat above), `third_party/nek5000` (Nek5000 core + KTH Toolbox +
case sources), `third_party/firedrake`.

## Release images: self-contained

`release/Dockerfile.hydrogym-cuda` and `release/Dockerfile.hydrogym-cpu` are
standalone images (not devcontainers) with the HydroGym backends built in:
m-AIA, NekRS, Nek5000, JAX and JAX-Fluids in both, Firedrake in the CPU image
only (see below). Nothing is built at container start and no solver source is
needed at run time. The released images (x86_64):

| Image | GPUs | Compute capability |
|---|---|---|
| `clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` | T4, RTX 20-50, A100, A30, A10, A4000-A6000, L4, L40, H100, H200, B100, B200 | cc75, cc80, cc86, cc89, cc90, cc100, cc120 |
| `clagemann/hydrogym-cpu:1.0.2` | none (CPU only) | - |

```bash
docker run -it --gpus all --ipc=host clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
docker run -it --ipc=host clagemann/hydrogym-cpu:1.0.2
```

For NVIDIA GH200 nodes (aarch64) there is an Apptainer image, built from the
recipe in `release/apptainer/` (see the next section). The user
documentation of all three is the docs page
[Container images](../docs/docs/installation/containers.md).

`release-gpu.devcontainer.json` and `release-cpu.devcontainer.json` open this
checkout (at `/workspace`) in these images.

What is inside (x86_64):

- **m-AIA**: `maia` on PATH (`/opt/maia/bin/maia`). The CUDA image holds one
  single-architecture NVHPC stdpar build per GPU generation
  (`/opt/maia/<ccXY>/bin/maia`: cc75, cc80, cc89, cc90, cc100, cc120); `maia`
  starts the one for the visible GPU (the exact compute capability, else the
  build of the same major version below it, e.g. cc80 for cc86).
  `MAIA_GPU_CC=cc80` overrides the detection, `maia --list-builds` lists the
  builds. The CPU image has a GCC build.
- **NekRS 26.0**: `nekrs`, `nekrs-mpmd` on PATH (`NEKRS_HOME=/opt/nekrs`).
  CUDA image: OCCA CUDA, GPU HYPRE and GPU-aware MPI; HYPRE is compiled for
  the image's architectures, the solver kernels are JIT-compiled for the GPU
  at the first run (cached in `NEKRS_CACHE_DIR`). CPU image: OCCA serial
  backend and CPU HYPRE (`NEKRS_BACKEND=SERIAL`, used by
  `examples/nekrs/channel/run.sh`). The channel case files are included.
  Several NekRS ranks on one GPU: NekRS gives each local rank its own GPU
  by default; add `--device-id 0` to the `nekrs`/`nekrs-mpmd` command line
  (for `run.sh`: `NEKRS_MPMD` pointing at a two-line script that does it).
  GPU-aware MPI needs UCX's CUDA transports: if you restrict `UCX_TLS`, keep
  `cuda_copy` (and `cuda_ipc` for several GPUs), e.g.
  `UCX_TLS=shm,self,tcp,cuda_copy`.
- **GPU access as a non-root user** (`docker run --user ...`): on a host
  whose `/dev/nvidia*` are group-restricted (e.g. `root:video`, mode 0660),
  add that group, `--group-add $(getent group video | cut -d: -f3)`, or the
  container sees no GPU (`maia: no NVIDIA GPU visible`).
- **Nek5000**: the case binaries in `/opt/nek5000/cases/<case>/nek5000`.
  Open MPI 5 counts slots by physical core: on hosts with fewer cores than
  ranks (1 Python + 10 Nek5000 ranks), pass `mpirun --oversubscribe`.
- **Firedrake + PETSc** (CPU image only): `/opt/venvs/firedrake`
  (`source /opt/venvs/firedrake/bin/activate`), with `hydrogym[firedrake]`,
  Stable-Baselines3, TensorBoard and torch (CPU), at the versions of
  `/opt/venvs/hydrogym`. Firedrake's kernel caches (PyOP2, TSFC) are in the
  venv for root; other users get them in `~/.cache/firedrake` (or
  `/tmp/firedrake-cache-uid<uid>` without a writable home directory), unless
  `PYOP2_CACHE_DIR`/`FIREDRAKE_TSFC_KERNEL_CACHE_DIR` are set.
  Not in the CUDA image: Firedrake's build needs the MPI Fortran modules
  (`use mpi`), which the images' HPC-X MPI ships for nvfortran only; Firedrake
  runs on the CPU anyway.
- **Python**: `/opt/venvs/hydrogym` (on PATH, also `/opt/venvs/nekrs`) with
  `hydrogym[maia,nek,nekrs,jax,jaxfluids]` (editable, `/opt/hydrogym`), torch
  (CPU build), JAX (CUDA 12 build in the CUDA image), JAX-Fluids. Versions
  as in `uv.lock`.
- `/opt/hydrogym/IMAGE_VERSION`: the HydroGym version and revision of the
  image (also in the image labels).

CUDA image, toolchain: CUDA 12.9 for everything (m-AIA, NekRS/HYPRE, the
OCCA JIT, JAX), so any NVIDIA driver with CUDA 12 support (>= 525) runs it
(CUDA minor version compatibility; tested with drivers 570 and 595);
NVHPC 26.5 for m-AIA. One MPI, HPC-X Open MPI 5 (CUDA-aware), for m-AIA,
NekRS, Nek5000 and mpi4py; the wrappers in `/opt/mpi-gnu/bin`
(`mpicc`, `mpicxx`, `mpifort` on PATH) put the GNU compilers underneath.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` is set, so a JAX process does not
reserve most of a GPU that a solver shares. The CPU image uses the Ubuntu
Open MPI for everything.

Build and test (Docker; the stages run in parallel with buildx):

```bash
.devcontainer/scripts/build_release_images.sh                    # GPU and CPU image, from HEAD
HYDROGYM_REF=v1.0.2 VARIANTS=cpu .devcontainer/scripts/build_release_images.sh
.devcontainer/scripts/test_release_image.sh clagemann/hydrogym-cpu:1.0.2
GPU=1 .devcontainer/scripts/test_release_image.sh clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
```

The build context comes from `git archive` of `HYDROGYM_REF` plus the solver
submodules it pins (`third_party/nekrs`, `third_party/m-AIA` with
`include/Eigen` and `include/doctest`, the KTH Nek5000/Toolbox), so they must
be initialised. HydroGym is the last layer of each image: a newer
`HYDROGYM_REF` with the same solver pins and lock file only rebuilds that
layer. A build takes several hours (NVHPC base image ~50 GB; m-AIA six
times, NekRS; PETSc and Firedrake for the CPU image);
`BUILD_ARGS="--build-arg JOBS=8 ..."` limits the parallel compile jobs (see
the script header). `VARIANTS=turing_ampere_ada` (cc75-cc89) or
`VARIANTS=hopper_blackwell` (cc90-cc120) build smaller GPU images with the
m-AIA builds of one GPU group only; they are not released.

ARM (aarch64): these Docker images are x86_64 only; for GH200 nodes see the
next section.

## NVIDIA GH200 (aarch64): Apptainer image

`release/apptainer/*.def` build the same stack for GH200 nodes (Grace CPU,
Hopper GPU) as an Apptainer SIF:

- m-AIA for cc90: the `maia` wrapper as above; `maia --list-builds` shows `cc90`.
- NekRS 26.0: CUDA, cudaArch 90, GPU HYPRE, GPU-aware MPI.
- The Nek5000 case binaries.
- `/opt/venvs/hydrogym`: torch (CPU), JAX (CUDA 12), JAX-Fluids, at the
  versions of `uv.lock`. HydroGym itself is installed editable in `/opt/hydrogym`.

Toolchain: CUDA 12.9 and NVHPC 26.5 (`-tp=neoverse-v2` for m-AIA), with
HPC-X Open MPI 5 (CUDA-aware) from the NVIDIA HPC SDK. As in the x86_64 CUDA
image, there is no Firedrake. The image has no public registry location;
build it as below (a prebuilt SIF may be provided later).

Run (`--nv` brings in the host's GPU driver). The image is read-only, so run
in a writable directory:

```bash
apptainer exec --nv hydrogym-gh200.sif python -c "import hydrogym, jax; print(jax.devices())"
apptainer exec --nv hydrogym-gh200.sif maia --list-builds
# One node, the image's own MPI: the whole MPMD launch in one container
cd <run directory>
apptainer exec --nv hydrogym-gh200.sif \
    mpirun --bind-to none -np 1 python <controller.py> : -np 4 maia properties_run.toml
```

With Slurm, `srun` can instead start every rank in its own container. This
also works across several nodes, since the container's UCX uses the host's
InfiniBand devices. The image's Open MPI 5 needs a PMIx plugin of the
cluster's Slurm: check `srun --mpi=list` (e.g. `pmix`, or `pspmix` with
ParaStation PMIx) and use it in place of `pmix` below. Tested on NVIDIA
GH200 (ARM, aarch64) nodes with a PMIx plugin: host and device-buffer
(CUDA-aware) MPI on one and two nodes, and m-AIA MPMD. The
image's environment sets the HPC-X prefixes (`OPAL_PREFIX` etc.) that
`mpirun` would otherwise pass on to the ranks.

```bash
srun --mpi=pmix -n 4 apptainer exec --nv hydrogym-gh200.sif python my_mpi_script.py
```

For an MPMD launch, write one line per rank range into `mpmd.conf`; `%o` is
the offset within the range (here: the GPU). `srun --multi-prog` sets no
MPI application number, so set `MAIA_MPMD_APPNUM` for m-AIA (0 for the
Python ranks, 1 for the m-AIA ranks):

```text
0   env MAIA_MPMD_APPNUM=0 apptainer exec --nv hydrogym-gh200.sif python <controller.py>
1-4 env MAIA_MPMD_APPNUM=1 CUDA_VISIBLE_DEVICES=%o apptainer exec --nv hydrogym-gh200.sif maia properties_run.toml
```

and start it with `srun --mpi=pmix -n 5 --multi-prog mpmd.conf`.

Notes:

- `apptainer exec` passes the host environment in. The image resets the
  compiler and search-path variables of host modules (`CC`, `CPATH`,
  `CMAKE_PREFIX_PATH`, ...), but `PYTHONPATH` or `LD_PRELOAD` still reach the
  container. `--cleanenv` avoids that, but do not use it with `srun`: its
  ranks need the `PMIX_*`/`SLURM_*` variables. With `--cleanenv`, pass what
  the run needs as `APPTAINERENV_<NAME>`, e.g.
  `APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"` for the GPU
  list Slurm sets (`--env CUDA_VISIBLE_DEVICES=0,1,2,3` would split it at
  the commas).
- m-AIA uses the first GPU it sees. Give each m-AIA rank its own GPU
  through `CUDA_VISIBLE_DEVICES`. With `mpirun` (Python is local rank 0),
  launch each m-AIA rank as
  `bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK-1)) exec maia properties_run.toml'`.
  With `srun --multi-prog`, use `%o` as above.
- NekRS writes into the case directory and into its kernel cache
  (`NEKRS_CACHE_DIR`), so run cases from a writable copy. The NekRS channel
  example includes HydroGym's udf header by a relative path. Bind a
  writable copy over it: `--bind <copy>:/opt/hydrogym/examples/nekrs/channel`.
- Hugging Face downloads (case data) go to `~/.cache`. Compute nodes without
  internet need them downloaded first. `--home <dir>` puts that cache in
  `<dir>/.cache`; then set `HF_HUB_OFFLINE=1`.
- The NekRS NACA0012 wing (`examples/nekrs/wing`) needs at least 8 NekRS
  ranks on aarch64 (two nodes of 4 GPUs): with fewer ranks the static
  Nek5000 arrays exceed the aarch64 relocation range.

Build: on an aarch64 machine with Apptainer >= 1.3 and `--fakeroot`, e.g. a
GH200 cluster's login node. The solver compiles can run as batch jobs on a
compute node (see the script header).

```bash
# On the machine with the HydroGym checkout: the build context
STAGE_DIR=ctx STAGE_ONLY=1 HYDROGYM_REF=v1.0.2 .devcontainer/scripts/build_release_images.sh
git rev-parse --short=9 v1.0.2 > ctx/HYDROGYM_REVISION
# On the aarch64 machine, with ctx copied there
CTX=$PWD/ctx WORK=$PWD BUILD_ROOT=/tmp/hg-build ctx/build/scripts/build_apptainer_arm64.sh all
ctx/build/scripts/test_apptainer_image.sh hydrogym-gh200.sif prepare    # case data (internet)
ctx/build/scripts/test_apptainer_image.sh hydrogym-gh200.sif            # on a GH200 node, in an allocation
```

The steps:

1. `env`: the build environment (NVHPC 26.5 SDK image, the HPC-X compiler
   wrappers, the m-AIA libraries).
2. `venv`, `nek5000`, `maia`, `nekrs`: compiled with `apptainer exec` of the
   build environment into `BUILD_ROOT`. `RESUME=1` continues an
   interrupted make.
3. `pack`.
4. `image`: `hydrogym-cuda-base.sif`.
5. `hydrogym`: HydroGym on top. A new `HYDROGYM_REF` with the same solver
   pins and lock file only repeats this step.

## AMD GPUs: self-contained ROCm image

`rocm/Dockerfile.hydrogym-rocm` is a standalone image (not a devcontainer)
for AMD clusters, default target gfx942 (MI300A/MI300X) on ROCm 7.2.4. Unlike
the features above, everything is built in at image build time, so it runs
offline under apptainer or podman:

- NekRS (HIP) + `nekrs-mpmd`, m-AIA (hipstdpar, `maia` on PATH; the wrapper
  sets `HSA_XNACK=1`), the Nek5000 case binaries in
  `/opt/nek5000/cases/<case>/nek5000`, PETSc + Firedrake (CPU).
- `/opt/venvs/hydrogym` (on PATH, also reachable as `/opt/venvs/nekrs`):
  `hydrogym[maia,nek,nekrs,jaxfluids]`, CPU torch, ROCm JAX, JAX-Fluids.
  `/opt/venvs/firedrake`: Firedrake + `hydrogym[firedrake]`.
- One MPI (UCX + Open MPI 5 with ROCm). Without an AMD GPU (e.g. a smoke
  test on another machine) UCX cannot start, so pass
  `-e OMPI_MCA_pml=ob1 -e OMPI_MCA_osc=^ucx -e OMPI_MCA_btl=self,sm`.

```bash
.devcontainer/scripts/build_rocm_image.sh        # -> hydrogym-rocm:rocm7.2.4-gfx942, from HEAD
GPU_ARCH="gfx90a;gfx942" .devcontainer/scripts/build_rocm_image.sh
```

The build context comes from `git archive` plus the solver submodules
(`third_party/nekrs`, `third_party/m-AIA` with `include/Eigen` and
`include/doctest`, the KTH Nek5000/Toolbox), so they must be initialised.

## Gotchas

- **Build parallelism is RAM-aware, not a flat `-j16`**: the base image
  build and MAIA's own build (`maia-gpu`/`maia-cpu` postCreateCommand) get
  their `-j` value from `.devcontainer/base/compute_parallelism.sh`
  (`min(nproc, available_RAM_GB / 5)`) instead of a number hand-picked for
  one machine — some translation units peak at ~4.2GB RSS each, so a flat
  `-j16` either under-uses a big box or OOM-kills the compiler on a
  smaller one. `build_all_containers.sh` still runs configs sequentially,
  not in parallel, since two builds at once would blow past what even this
  per-build calculation accounts for.
- **MAIA host-config auto-detection is hostname-based, not env-var based**:
  `configure.py`'s host detection (`cmake/GetHost.cmake`) pattern-matches
  the container's OS hostname against a hardcoded cluster list — it does
  NOT read `MAIA_HOST`/`HOST` env vars, and this container's hostname
  (`--hostname=LocalGPU` in `runArgs`) doesn't match anything in that list.
  `MAIA_HOST_FILE` (see `maia-gpu`/`maia-cpu` `postCreateCommand.sh`) is the
  real override — it points straight at a host config file and skips
  hostname detection entirely, so none of this matters in practice.
- **No `__version__` attribute**: the Firedrake build has no
  `__version__` attribute, so version
  checks here use `importlib.metadata.version(...)` instead of a bare
  attribute access (see `ensure_hydrogym.sh`).
