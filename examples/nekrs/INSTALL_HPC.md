# Installing NekRS with the HydroGym patches on an HPC cluster

This guide covers a native build (no container) of NekRS 26.0 plus HydroGym's `nekrs-mpmd` driver on

- **a GH200 cluster**: NVIDIA GH200 Grace Hopper, aarch64 CPUs, CUDA (`sm_90`), environment modules (Open MPI or
  an MPICH-based MPI)
- **an MI300A cluster**: HPE Cray EX with AMD MI300A APUs, ROCm (`gfx942`), Cray Programming Environment

and on similar systems. The build runs `.devcontainer/features/nekrs/build_nekrs.sh`, the same script that builds the
devcontainer (CUDA) and the `hydrogym-rocm` image (HIP), so a cluster build has the same patches and configuration
as the tested containers.

> Module names and versions below are typical of such software stacks. They are **not tested on every
> machine**; check them with `module avail` / `module spider` (module-based clusters: `module spider <name>`; Cray EX: `module avail` after `module load PrgEnv-gnu`). The tested reference builds are the CUDA
> devcontainer (HPC-X Open MPI 5, GCC, CUDA 12.9, A6000 `sm_86`) and the ROCm 7.2.4 image (Open MPI 5 + UCX,
> amdclang, gfortran, MI300A `gfx942`).

## 1. What gets built, and what is patched

| Part | Source | HydroGym changes |
|---|---|---|
| NekRS 26.0 (OCCA, hypre, gslib, Nek5000, parRSB, all vendored) | `third_party/nekrs` submodule (upstream tag `v26.0`) | five build patches, `.devcontainer/features/nekrs/patches/` |
| `nekrs-mpmd` | `examples/nekrs/mpmd/driver.cpp` (NekRS's `src/bin/driver.cpp` + `nekrs-26.0_driver-mpmd-split.patch`, already applied in the copy) | `MPI_COMM_WORLD` is split by `NEKRS_MPMD_COLOR`, so the Python controller and the NekRS ranks can run in one MPMD launch |
| coupling to Python | `hydrogym/nekrs/udf/hydrogym_nekrs.hpp`, included by the case `.udf` | no patch, compiled at run time with the case; it accesses NekRS 26.0 internals (`nrs_t::computeCFL`, `fluidSolver_t::o_ADV`) |

The build patches (each file's header explains its patch):

| Patch | Needed for | Symptom without it |
|---|---|---|
| `NekRS-26.0_fix-hypre-gpu-aware-mpi-flags.patch` | `NEKRS_GPU_MPI=ON` (any backend) | hypre configure aborts: `invalid feature name` |
| `NekRS-26.0_hypre-cuda-gpu-arch.patch` | CUDA on GPUs other than `sm_70`/`sm_80`/`sm_86`, e.g. **GH200 with CUDA 12** | hypre kernels fail at run time with `no kernel image is available for execution on the device` |
| `NekRS-26.0_hypre-hip-gpu-arch.patch` | HIP (**MI300A**) | hypre's HIP kernels built for the compiler's default GPU, failing or slow at run time |
| `NekRS-26.0_hypre-hip-warp-sync.patch` | HIP with ROCm ≥ 7 | hypre build error: `static declaration of '__syncwarp' follows non-static declaration` |
| `NekRS-26.0_hypre-vendor-math-path.patch` | CUDA from NVHPC's multi-version layout (`cuda/12.9` + `math_libs/12.9`) | `cublas_api.h: cudaEmulationSpecialValuesSupport is undefined` (wrong cuBLAS headers) |

`build_nekrs.sh` applies all five with `git apply` and skips those already applied. The patches that don't concern
a backend leave that build unchanged, so all of them can be applied everywhere.

## 2. Requirements

### Build tools
- **CMake ≥ 3.21** (NekRS's `cmake_minimum_required`); a recent one (≥ 3.24) is better for HIP language support.
- GNU **make** and **git** (for `git apply` on the patches; the source tree doesn't need to be a git checkout).
  No autotools: hypre's `configure` ships pre-generated.
- `bash`, `sed`, `tar`; about **10 GB** of disk for the source plus build tree. The install is about 1 GB, and the
  build tree can be deleted afterwards.
- Build time: about 30–60 min with 32+ cores. Most of it is GPU hypre; parallel `make` is used, set `NPROC` to limit it.

### Compilers
- **C, C++17 and Fortran** compilers. NekRS's Nek5000 Fortran and its OCCA JIT kernels are written for
  **GCC/gfortran**; GCC ≥ 9.1 is required by NekRS, and GCC 11–13 is a safe choice. NVHPC (`nvc++`) and the Cray
  compilers are untested for NekRS; don't use them as the compilers underneath MPI.
- **HIP (MI300A): the C++ compiler underneath MPI has to compile HIP**. NekRS builds hypre's device code with
  `CMAKE_CXX_COMPILER` (`cmake/hypre.cmake`), so use ROCm's `amdclang++` (`$ROCM_PATH/llvm/bin/amdclang++`) as the
  C++ compiler, with `amdclang` for C and **gfortran** for Fortran. That is the configuration of the tested ROCm image.
- CUDA (GH200): gcc/g++/gfortran; the device code goes through `nvcc`.

### GPU toolkits
- **CUDA ≥ 12.0** (NekRS: `find_package(CUDAToolkit 12.0)`), with **cuBLAS, cuSPARSE, cuRAND, cuSOLVER** (GPU hypre).
  A standard toolkit module has them in `$CUDA_HOME`. With NVHPC they are in `$NVHPC_ROOT/math_libs`, which
  `build_nekrs.sh` handles (it needs `NVHPC_ROOT` set, and `CUDA_VERSION` for the multi-CUDA layout). For GH200
  either CUDA 12.x (with the arch patch) or 13.x works.
- **ROCm ≥ 6.2** with **rocBLAS, rocSPARSE, rocRAND, rocSOLVER** (+ their CMake configs under `$ROCM_PATH/lib/cmake`)
  and `hipcc` (OCCA's default JIT compiler for HIP kernels, override with `OCCA_HIP_COMPILER`). Use the ROCm that
  matches the system's amdgpu driver (a ROCm much older than the driver can hang in the first kernels, see section 9).
- The GPU arch, numeric: `GPUARCH=90` (GH200), `GPUARCH=gfx942` (MI300A). Not `native`/`all`: the value is passed
  to hypre as well.

### MPI
- An MPI with the compiler wrappers **`mpicc`, `mpicxx`, `mpif77`, `mpifort`** in one directory (`MPI_BIN`), whose
  underlying compilers can be overridden by environment variables: Open MPI (`OMPI_CC`/`OMPI_CXX`/`OMPI_FC`) or an
  MPICH (Cray MPICH, ParaStation MPI, MVAPICH: `MPICH_CC`/`MPICH_CXX`/`MPICH_FC`). `build_nekrs.sh` writes wrappers
  with both sets fixed into `$NEKRS_HOME/mpi/bin`. NekRS records its compilers at configure time and calls them
  again **at run time** to build the case's udf and OCCA kernels, so they must work in a job without extra variables.
- GPU-aware MPI: optional (`GPUMPI=true`, default). Without it (`GPUMPI=false`), NekRS copies its halo buffers
  through the host, which is cheap on an MI300A APU and with few ranks per simulation.
- **One MPI for solver and controller**: mpi4py must be built against the same MPI library as NekRS. The MPMD launch
  puts the Python controller and the NekRS ranks into one `MPI_COMM_WORLD`, and two different MPI libraries hang or
  crash in `MPI_Init`.

### Python (the HydroGym controller)
- Python ≥ 3.11 with `numpy`, `scipy`, **`mpi4py` (≥ 4.0, built from source against the MPI above)**, and the
  hydrogym package from this checkout (`pip install -e ".[nekrs]"`). For training add PyTorch; the CPU wheel is
  enough, because the policy runs on the controller rank. Use a venv on a file system that compute nodes can read.

### Other
- **git-lfs** for the wing mesh (`third_party/nek5000/cases/naca0012_200k/naca_wing.re2`); the channel case needs no LFS.
- Compute nodes don't need network access. NekRS vendors all of its dependencies, so only the `pip install` steps use
  the network.

## 3. Get the sources

```bash
git clone https://github.com/dynamicslab/hydrogym.git hydrogym && cd hydrogym
git submodule update --init third_party/nekrs           # v26.0
git -C third_party/nekrs describe --tags                # -> v26.0
git lfs install && git lfs pull                         # wing mesh (channel: not needed)
export REPO=$PWD
```

On machines without outbound git access, clone on another machine and copy the tree. `build_nekrs.sh` also works
on a plain source tree that isn't a git checkout.

## 4. GH200 cluster (CUDA)

### 4.1 Environment

Build in an interactive job (`srun -N1 --pty bash`, or
`salloc`), so the configure checks see a GPU node and `make` gets all cores.

```bash
module load GCC OpenMPI CUDA CMake Python
# or an MPICH-based MPI module instead of OpenMPI
module list
which mpicc nvcc cmake; echo $CUDA_HOME
nvcc --version                     # >= 12.0
mpicc -showme 2>/dev/null || mpicc -show
```

Check that the MPI is CUDA-aware if you keep `GPUMPI=true`:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value   # Open MPI: true
# MPICH-based MPIs: see the centre's documentation for the CUDA-aware variant
```

### 4.2 Build

```bash
export NEKRS_HOME=<install prefix>/nekrs-26.0-hydrogym   # a directory readable from the compute nodes
cd $REPO
NEKRS_SRC=$REPO/third_party/nekrs MPMD_SRC=$REPO/examples/nekrs/mpmd \
BACKEND=CUDA GPUARCH=90 GPUMPI=true BUILDTYPE=RelWithDebInfo \
MPI_BIN=$(dirname $(which mpicc)) MPI_CC=gcc MPI_CXX=g++ MPI_FC=gfortran \
NPROC=32 \
    bash .devcontainer/features/nekrs/build_nekrs.sh
```

Don't set `NVHPC_ROOT` or `CUDA_VERSION` here: with a plain CUDA module the math libraries are taken from
`$CUDA_HOME`. If you build with the NVHPC module instead, set `NVHPC_ROOT` (the module normally does) and
`CUDA_VERSION` to the CUDA subdirectory you want (`ls $NVHPC_ROOT/cuda`).

The script writes `third_party/nekrs/cmake.log` and `make.log`, and prints the first errors on failure. On success it
prints `=== NekRS build successful ===` and installs `nekrs` and `nekrs-mpmd` into `$NEKRS_HOME/bin`, the MPI
wrappers into `$NEKRS_HOME/mpi/bin`, and an `nvcc` wrapper into `$NEKRS_HOME/cuda/bin`.

### 4.3 Run-time environment

```bash
module load GCC OpenMPI CUDA Python      # the same modules as for the build (udf/JIT at run time)
export NEKRS_HOME=<install prefix>/nekrs-26.0-hydrogym
export PATH=$NEKRS_HOME/bin:$PATH
export OCCA_CUDA_COMPILER=$NEKRS_HOME/cuda/bin/nvcc   # JIT kernels with the build's toolkit
export NEKRS_BACKEND=CUDA
ulimit -s unlimited
```

## 5. MI300A cluster (HPE Cray EX, ROCm)

### 5.1 Environment

The MPI of a Cray EX system is Cray MPICH, normally used through the Cray compiler drivers `cc`/`CC`/`ftn`. `build_nekrs.sh` needs
real `mpicc`/`mpicxx`/`mpif77` wrappers whose compilers can be swapped. Cray MPICH ships those as well, in
`$CRAY_MPICH_DIR/bin` (`MPICH_DIR` on some versions), and they honour `MPICH_CC`/`MPICH_CXX`/`MPICH_FC`. That lets
us put `amdclang++` (HIP-capable) under C++ and gfortran under Fortran, as in the tested ROCm image.

```bash
module load PrgEnv-gnu                     # gcc/gfortran + the GNU flavour of cray-mpich
module load rocm                           # must match the amdgpu driver; check: module avail rocm
module load craype-accel-amd-gfx942 cmake cray-python
module load craype-x86-genoa 2>/dev/null   # usually loaded by default (MI300A CPU cores: Zen 4)
module list
echo $ROCM_PATH $CRAY_MPICH_DIR
ls $CRAY_MPICH_DIR/bin                     # mpicc mpicxx mpif77 mpif90 mpifort
$ROCM_PATH/llvm/bin/amdclang++ --version
hipconfig --version
```

The GNU and AMD flavours of cray-mpich are C-ABI compatible. NekRS uses only the C API plus the Fortran
`mpif.h` from gfortran, so the GNU flavour works with amdclang underneath. If `amdclang++` picks an old system GCC's
libstdc++ (C++17 errors in `<filesystem>`), point it at the module's GCC with
`MPI_CXX="$ROCM_PATH/llvm/bin/amdclang++ --gcc-toolchain=$(dirname $(dirname $(which gcc)))"`, and the same for `MPI_CC`.

Build on a compute node (`salloc -N1 -t 2:00:00`, then `srun --pty bash`, or put the build into a batch job): the
login nodes have no MI300A and may have a different CPU.

### 5.2 Build

```bash
export NEKRS_HOME=<install prefix>/nekrs-26.0-hydrogym   # a directory readable from the compute nodes
cd $REPO
NEKRS_SRC=$REPO/third_party/nekrs MPMD_SRC=$REPO/examples/nekrs/mpmd \
BACKEND=HIP GPUARCH=gfx942 GPUMPI=false BUILDTYPE=RelWithDebInfo \
MPI_BIN=$CRAY_MPICH_DIR/bin \
MPI_CC=$ROCM_PATH/llvm/bin/amdclang MPI_CXX=$ROCM_PATH/llvm/bin/amdclang++ MPI_FC=gfortran \
NPROC=48 \
    bash .devcontainer/features/nekrs/build_nekrs.sh
```

`ROCM_PATH` must be set (the `rocm` module does): the script passes it as `CMAKE_PREFIX_PATH`, so CMake finds
rocBLAS etc.

**GPU-aware Cray MPICH** (`GPUMPI=true`) additionally needs the GPU Transport Layer library. The `cc`/`CC` drivers
link it when `craype-accel-amd-gfx942` is loaded, but the plain `mpicc` wrappers don't. Start with `GPUMPI=false`.
For `GPUMPI=true`, build with it and preload GTL at run time:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD=$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_hsa.so      # path: ls $CRAY_MPICH_ROOTDIR/gtl/lib
```

Without GTL, Cray MPICH aborts at start (`MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not
linked`) or, without `MPICH_GPU_SUPPORT_ENABLED`, crashes when a device pointer reaches MPI.

### 5.3 Run-time environment

```bash
module load PrgEnv-gnu rocm craype-accel-amd-gfx942 cray-python    # as for the build
export NEKRS_HOME=<install prefix>/nekrs-26.0-hydrogym
export PATH=$NEKRS_HOME/bin:$PATH
export NEKRS_BACKEND=HIP
ulimit -s unlimited
```

Don't set `HIP_VISIBLE_DEVICES` to an empty string, because that hides all GPUs. Slurm sets `ROCR_VISIBLE_DEVICES`
itself.

## 6. Python environment (both clusters)

The venv's mpi4py must be built with the wrappers from the build (`$NEKRS_HOME/mpi/bin/mpicc`), i.e. against the
same libmpi:

```bash
python3 -m venv $NEKRS_HOME/venv && source $NEKRS_HOME/venv/bin/activate
pip install --upgrade pip
MPICC=$NEKRS_HOME/mpi/bin/mpicc pip install --no-cache-dir --no-binary mpi4py "mpi4py>=4.0"
pip install -e "$REPO[nekrs]"
pip install torch --index-url https://download.pytorch.org/whl/cpu     # training only
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"
```

The last line must name the MPI that NekRS was built with (Open MPI / MPICH / Cray MPICH). On a module-based
system an `mpi4py` module for the same toolchain is an alternative, and on a Cray EX `cray-python` ships an mpi4py built against
cray-mpich. In both cases, check the library line.

## 7. Verify the build

1. `ls $NEKRS_HOME/bin` shows `nekrs`, `nekrs-mpmd`, and the NekRS tool scripts.
2. A NekRS-only run of the channel case on one GPU (the first run JIT-compiles the kernels into `.cache/` in the case
   directory, which takes a few minutes):

   ```bash
   (cd $REPO/examples/nekrs/channel && python ../fetch_case.py)   # mesh + initial field, if tcf.re2 is missing
   cp -r $REPO/examples/nekrs/channel $SCRATCH/channel-test && cd $SCRATCH/channel-test
   srun -N1 -n1 --gpus-per-task=1 nekrs --setup tcf.par --backend $NEKRS_BACKEND
   ```

   The setup log reports the backend and device (`CUDA`/`HIP`, `device id`), and then time steps.
3. HydroGym coupling (controller + NekRS in one MPMD launch): see section 8. Success markers in the log
   (`grep -a`): `probes x` (probe setup), `[CLIENT]` (controller connected), `[HGNEK] TERMN` (clean end).

## 8. Launching HydroGym + NekRS on the clusters

`examples/nekrs/*/run.sh` is written for Open MPI's `mpirun`. It launches

```
mpirun -n 1 python <script> ... : -n NP env NEKRS_MPMD_COLOR=1 bind_numa.sh nekrs-mpmd --setup <case>.par --backend $NEKRS_BACKEND
```

after preparing the run directory, and it sets `NEKRS_UDF_INCLUDES=$REPO/hydrogym/nekrs/udf` (the udf includes
`hydrogym_nekrs.hpp` from there) and `NEKRS_CACHE_DIR`. On a Slurm cluster without `mpirun` (Cray MPICH, or
ParaStation MPI) do the same with `srun --multi-prog`:

```bash
export NEKRS_UDF_INCLUDES=$REPO/hydrogym/nekrs/udf
export PYTHONPATH=$REPO
cat > mpmd.conf <<EOF
0     $NEKRS_HOME/venv/bin/python -u train_td3.py --conf channel_paper.yml --nproc 4 --run-root .
1-4   env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup tcf.par --backend $NEKRS_BACKEND
EOF
srun -n 5 --gpus-per-node=4 --multi-prog mpmd.conf
```

The controller has world rank 0 (color 0, no GPU needed). The NekRS ranks pick their GPU as the node-local rank
among the NekRS ranks. Several simulations use colors 1, 2, … in consecutive rank blocks (see
`hydrogym/nekrs/multisim.py`). `bind_numa.sh` takes the node-local rank from Open MPI
(`OMPI_COMM_WORLD_LOCAL_RANK`) or, under `srun`, from `SLURM_LOCALID`; the wing and channel `run.sh` use
`srun --multi-prog` themselves when the Open MPI installation has no `mpirun`. Otherwise use Slurm's binding
(`--cpu-bind=...`, `--gpu-bind=closest` or the centre's recommended GPU binding).
Unbound ranks on 4-APU MI300A nodes cost up to 6x per step. An Open MPI with `mpirun` can use `run.sh` almost
unchanged: drop `--allow-run-as-root`, and set `PY` to the venv's python.

## 9. Run-time notes and pitfalls

- **Writable case directory**: OCCA kernels and the udf library are compiled into `$NEKRS_CACHE_DIR` (default
  `.cache/` in the case/run directory) at the first run. Use a file system writable from the compute nodes. The
  compilers recorded at build time (`$NEKRS_HOME/mpi/bin/*`, `nvcc`/`hipcc`) are called then, so load the build
  modules in the job.
- The udf is rebuilt only when the `.udf` file changes, not when `hydrogym_nekrs.hpp` changes. `run.sh` deletes
  `$NEKRS_CACHE_DIR/udf` when the header is newer; do the same in your own launch scripts.
- `ulimit -s unlimited` before the launch (Nek5000 Fortran parts use large stack arrays).
- **GPU-aware MPI correctness**: GPU peer-to-peer (UCX `cuda_ipc`) can silently corrupt data on
  systems with the IOMMU enabled: the runs look fine but diverge. Compare a short run with `GPUMPI` on and off (or `UCX_TLS`
  without `cuda_ipc` / `MPICH_GPU_IPC_ENABLED=0`) on multi-GPU runs once per system. Single-rank-per-simulation runs
  are unaffected.
- **ROCm ↔ driver mismatch**: a user-space ROCm much older than the kernel driver can hang in the first kernels
  (e.g. in `findpts`). Use the ROCm module the centre recommends for the installed
  driver.
- Check GPU performance once: NekRS prints an Ax benchmark at setup (GDOF/s and GB/s). Nodes with much lower memory
  bandwidth than nominal make every step correspondingly slower; compare the benchmark across nodes.

## 10. Maintenance (NekRS version bumps)

- All patches and `nekrs-mpmd` target **NekRS 26.0**. After moving the submodule, `build_nekrs.sh` stops with
  `could not apply ... (NekRS pin changed?)` for each patch that no longer applies. Check upstream to see whether
  it is fixed there, and rebase or drop the patch.
- `examples/nekrs/mpmd/driver.cpp` is a copy of NekRS's `src/bin/driver.cpp` with
  `nekrs-26.0_driver-mpmd-split.patch` applied. Recreate it from the new driver.
- `hydrogym/nekrs/udf/hydrogym_nekrs.hpp` reaches private NekRS members through explicit template instantiation
  (`nrs_t::computeCFL`, `fluidSolver_t::o_ADV`). If their names or types change, the udf build fails in that header.

## Quick reference: `build_nekrs.sh` variables

| Variable | GH200 (CUDA) | MI300A (HIP) | Meaning |
|---|---|---|---|
| `NEKRS_SRC` | `$REPO/third_party/nekrs` | same | NekRS source |
| `NEKRS_HOME` | install prefix | install prefix | install prefix |
| `MPMD_SRC` | `$REPO/examples/nekrs/mpmd` | same | driver sources |
| `BACKEND` | `CUDA` | `HIP` | OCCA backend |
| `GPUARCH` | `90` | `gfx942` | `CMAKE_CUDA_ARCHITECTURES` / `CMAKE_HIP_ARCHITECTURES` (also hypre's) |
| `GPUMPI` | `true` (CUDA-aware MPI) | `false` (or `true` + GTL) | `NEKRS_GPU_MPI` |
| `BUILDTYPE` | `RelWithDebInfo` | same | `CMAKE_BUILD_TYPE` |
| `MPI_BIN` | `$(dirname $(which mpicc))` | `$CRAY_MPICH_DIR/bin` | directory of `mpicc mpicxx mpif77 mpifort` |
| `MPI_CC` / `MPI_CXX` / `MPI_FC` | `gcc` / `g++` / `gfortran` | `amdclang` / `amdclang++` / `gfortran` | compilers under the MPI wrappers |
| `CUDA_HOME` | from the CUDA module | – | CUDA toolkit |
| `NVHPC_ROOT`, `CUDA_VERSION` | only with NVHPC | – | NVHPC multi-CUDA layout |
| `ROCM_PATH` | – | from the rocm module | ROCm prefix |
| `NPROC` | optional | optional | parallel make jobs (default `nproc`) |
