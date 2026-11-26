---
sidebar_position: 6
---

# NekRS

[NekRS](https://github.com/Nek5000/nekRS) is the GPU version of Nek5000: the same spectral-element discretisation
(polynomial order N, BDF3/EXT3, P_N-P_N splitting), with OCCA kernels for CUDA and HIP. HydroGym couples a Python
controller to NekRS over MPI (MPMD), as for m-AIA. Probes, actuators, reward, initial conditions and actuator
power are defined in a YAML file and sent to the solver at run time; the NekRS case only marks the boundaries
that may be actuated.

The backend provides two cases:

| Case | Directory | Description |
|---|---|---|
| Channel | `examples/nekrs/channel` | Open channel at Re_τ = 207 (HydroGym `TCFlarge_3D_Re207`): blowing and suction at every wall GLL node. |
| NACA0012 wing | `examples/nekrs/wing` | NACA0012 at Re_c = 2·10⁵ and zero incidence (HydroGym `NACA0012_3D_Re200000_AOA0`): blowing and suction on the suction-side window 0.25 < x/c < 0.86. The NekRS port is validated against the Nek5000 statistics of Wang, Atzori & Vinuesa (2025), see [NekRS NACA0012 validation](../examples/nekrs/wing_validation). |

```
hydrogym/nekrs/
  udf/hydrogym_nekrs.hpp   solver side, included by the case udf
  interface.py             NekRSInterface: the MPI requests
  layout.py                probe and actuator positions, footprints
  env.py                   NekRSFlowEnv (gymnasium.Env), set up from the YAML
  power.py                 actuator power of the wall transpiration (hydrogym.power interface)
  multiagent.py            one agent per actuator (PettingZoo, SB3 VecEnv)
  multisim.py              several NekRS instances for one controller
examples/nekrs/
  mpmd/                    nekrs-mpmd: the NekRS driver that joins the controller's MPI_COMM_WORLD split
  channel/, wing/          cases, run scripts, post-processing
  INSTALL_HPC.md           cluster build (GH200 / CUDA, MI300A / ROCm)
```

## Requirements

| Requirement | Notes |
|---|---|
| GPU | NVIDIA (CUDA 12.x) or AMD (ROCm 6/7). One GPU per NekRS rank by default (`--device-id 0` puts several ranks on one GPU). |
| MPI | GPU-aware Open MPI, MPICH, ParaStation MPI or Cray MPICH. mpi4py has to be built against the same MPI. |
| Compilers | C++17 and Fortran compilers (GCC 11 or newer), CMake ≥ 3.21 |
| Python | ≥ 3.11, `pip install -e ".[nekrs]"` |

## Option 1: release images (recommended)

The HydroGym images (see [Container images](./containers)) contain NekRS 26.0 in `/opt/nekrs` (`NEKRS_HOME`), with
`nekrs` and `nekrs-mpmd` on the `PATH` and `hydrogym[nekrs]` in `/opt/venvs/hydrogym` (also reachable as
`/opt/venvs/nekrs`):

- `clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` (x86_64): OCCA CUDA, GPU HYPRE and GPU-aware MPI (HPC-X Open MPI 5).
- `clagemann/hydrogym-cpu:1.0.2` (x86_64): the serial OCCA backend and CPU HYPRE (`NEKRS_BACKEND=SERIAL`).
- The GH200 Apptainer image (aarch64), built from `.devcontainer/release/apptainer/`: CUDA, cudaArch 90.

The `run.sh` scripts of the cases run as is in these images. The channel case files are included. Notes:

- NekRS gives each node-local rank its own GPU (device = local rank, without wrap-around). For more NekRS ranks per
  node than GPUs, start NekRS through `examples/nekrs/gpu_share.sh`: it makes GPU (rank mod number of GPUs) the only
  visible device of each rank (`CUDA_VISIBLE_DEVICES`, or `ROCR_VISIBLE_DEVICES` for ROCm) and adds `--device-id 0`,
  e.g. `mpirun -n 1 python deploy.py ... : -n 8 gpu_share.sh nekrs-mpmd --setup naca_wing.par --backend CUDA`.
  The wing's `run.sh` does this with `GPU_SHARE=1` (`run_slurm.sbatch`: automatically when `NP` exceeds the GPUs of
  the node).
- GPU-aware MPI needs UCX's CUDA transports: if you restrict `UCX_TLS`, keep `cuda_copy` (and `cuda_ipc` for several
  GPUs), e.g. `UCX_TLS=shm,self,tcp,cuda_copy`.
- The first run JIT-compiles the solver kernels into `NEKRS_CACHE_DIR` (several minutes); keep it between runs.
- Under Apptainer the image is read-only: NekRS writes into the case directory, so bind a writable copy of the
  case over it (the channel's udf includes HydroGym's header by a relative path), e.g.
  `--bind <copy>:/opt/hydrogym/examples/nekrs/channel`.

## Option 2: devcontainer (CUDA)

The devcontainer feature `.devcontainer/features/nekrs` builds NekRS 26.0 from the `third_party/nekrs` submodule
with OCCA CUDA, GPU HYPRE and GPU-aware MPI. It also builds `nekrs-mpmd` and a `hydrogym[nekrs]` venv whose mpi4py
uses the same MPI. The test configuration `.devcontainer/nekrs-test.devcontainer.json` uses the feature alone:

```json
"features": {
  "./features/nekrs": { "cudaArch": "80;86", "cudaVersion": "12.9", "gpuMpi": true }
}
```

- `cudaArch` sets the ahead-of-time compiled parts (HYPRE). OCCA JIT-compiles the solver kernels for the GPU present at run time.
- `cudaVersion` defaults to 12.9: the HYPRE vendored in NekRS 26.0 does not build with CUDA 13's CCCL 3.
- `gpuMpi=false` is for hosts with broken GPU peer-to-peer.

The feature sets `NEKRS_HOME=/workspace/third_party/nekrs/install` and puts `nekrs` and `nekrs-mpmd` on the `PATH`.

## Option 3: ROCm image (AMD)

`.devcontainer/scripts/build_rocm_image.sh` builds a self-contained ROCm image (NekRS with HIP, m-AIA, Nek5000,
Firedrake, JAX-Fluids) from a committed revision:

```bash
.devcontainer/scripts/build_rocm_image.sh                                   # gfx942 (MI300A), ROCm 7.2.4
GPU_ARCH="gfx90a;gfx942" TAG=hydrogym-rocm:test .devcontainer/scripts/build_rocm_image.sh
podman save -o hydrogym-rocm.tar <TAG>      # for an offline cluster, then: podman load -i hydrogym-rocm.tar
apptainer build hydrogym-rocm.sif docker-archive://hydrogym-rocm.tar      # the SIF of the Slurm scripts (BACKEND=HIP)
```

## Option 4: HPC cluster (native build)

`examples/nekrs/INSTALL_HPC.md` gives the module environments and build commands for a GH200 cluster (CUDA)
and an HPE Cray EX MI300A cluster (ROCm). The build itself is one script, shared with the devcontainer:

```bash
export NEKRS_HOME=<install prefix>/nekrs-26.0-hydrogym
NEKRS_SRC=$REPO/third_party/nekrs MPMD_SRC=$REPO/examples/nekrs/mpmd \
BACKEND=CUDA GPUARCH=90 GPUMPI=true BUILDTYPE=RelWithDebInfo \
MPI_BIN=$(dirname $(which mpicc)) MPI_CC=gcc MPI_CXX=g++ MPI_FC=gfortran NPROC=32 \
    bash .devcontainer/features/nekrs/build_nekrs.sh
```

The script applies HydroGym's patches to NekRS (HYPRE GPU architecture and math-library paths, HIP warp-sync shims
on ROCm 7, GPU-aware-MPI flags). It installs `nekrs`, `nekrs-mpmd`, MPI compiler wrappers and an `nvcc` wrapper
into `$NEKRS_HOME`. The Python environment needs mpi4py built with those wrappers:

```bash
python3 -m venv $NEKRS_HOME/venv && source $NEKRS_HOME/venv/bin/activate
MPICC=$NEKRS_HOME/mpi/bin/mpicc pip install --no-cache-dir --no-binary mpi4py "mpi4py>=4.0"
pip install -e "$REPO[nekrs]"     # $REPO: a clone with `git lfs pull` done (meshes and case files are Git LFS files)
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"   # must name NekRS's MPI
```

If NekRS is installed some other way, build only the driver: `cd examples/nekrs/mpmd && ./build.sh` (needs
`NEKRS_HOME`).

## Running

Both cases are started with `run.sh`. It prepares the run directory, sets `NEKRS_UDF_INCLUDES` and
`NEKRS_CACHE_DIR`, and launches the controller (rank 0) and `NP` NekRS ranks in one MPMD job: with `mpirun`, or
with `srun --multi-prog` on Slurm clusters without `mpirun`. The first run JIT-compiles the kernels, which takes a
few minutes.

### Channel

```bash
cd examples/nekrs/channel
python ../fetch_case.py                                          # mesh and initial field from Hugging Face
NP=1 RUN=runs/zero ./run.sh --policy zero --steps 20              # client.py: zero | oppo | random | td3:<zip>
NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --conf channel_td3.yml --episodes 10
NP=1 RUN=runs/power SCRIPT=tests/power_check.py ./run.sh         # actuator-power check on the solver
```

### NACA0012 wing

```bash
cd examples/nekrs/wing
python prepare_case.py                       # 21 GB start field rs8naca_wing0.f00008 (t = 28) into restart/
NP=8 RUN=runs/zero ./run.sh --policy zero --actions 2500
NP=8 RUN=runs/oc   CONF=wing_oc.yml ./run.sh --policy oc --alpha 1
NP=8 RUN=runs/td3  ./run.sh --policy td3:naca0012_wing_re2e5/td3/seed1998     # released policy (policies/)
NP=8 RUN=runs/n1   ./run.sh --policy td3:naca0012_wing_re2e5/td3_N1/seed0
NP=8 RUN=runs/n1f  ./run.sh --policy td3:naca0012_wing_re2e5/td3_N1F/seed3
NP=8 RUN=runs/own  ./run.sh --policy td3:<SB3 zip trained in the channel>
python postprocess.py runs/zero runs/oc runs/td3 --skip 500 --out cf.png
# drag reductions (suction-side strip, C_d,f and C_d of the wing) and actuation power against the uncontrolled run,
# over 1-3 c/U after the start of control at t = 28
python fig4d.py --ref runs/zero --t0 1 --t1 3 --out fig4d TD3=runs/td3:28 OC=runs/oc:28
python power_forms.py --ref runs/zero --t0 1 --t1 3 --out power TD3=runs/td3:28 N1=runs/n1:28 OC=runs/oc:28
python action_hist.py --t0 1 --t1 3 --out action_hist TD3=runs/td3:28 N1=runs/n1:28 OC=runs/oc:28   # actuation statistics
```

The defaults (`naca_wing.par`, `wing.yml`) are the case and deployment of the HydroGym Nek5000 wing
(`third_party/nek5000/cases/naca0012_200k`): Re_c = 2·10⁵, the trip of the reference simulation Ref1 (Wang, Atzori &
Vinuesa, J. Fluid Mech. 2025: one line at x/c = 0.1, 30 spanwise modes renewed every 0.008 c/U∞, amplitude 10),
Ref1's restart field at t = 28, Δt = 4·10⁻⁵ and 31 solver steps (1.24·10⁻³ c/U∞) per action. The uncontrolled flow
with this trip is validated against Ref1 ([NekRS NACA0012 validation](../examples/nekrs/wing_validation)). The
channel policies act in the wall units of their training channel (action 1 = its friction velocity, 0.0624334733 at
Re_τ 206). Their observations [u_t', v_n'] at y+ = 15, along the wall tangent and normal (`frame: wall`; `wing_hg_fortran.yml`: the projection of the original Nek5000
coupling, `frame: hg_fortran`, with which the roll-out of Fig. 4d of the HydroGym paper was deployed: its numbers are reproduced with
`CONF=wing_hg_fortran.yml`; without a `frame` key the interface uses the velocity components as computed, `frame: global`),
are divided by the local friction velocity u_tau(x) of the uncontrolled wing at Re_c = 2·10⁵; zero net mass flux
holds on every spanwise line of nodes. Opposition control (`wing_oc.yml`) acts every 10 solver steps at
Δt = 2·10⁻⁵ with v_w = -v_n' at y+ = 15. The solver time step is `environment.dt` of the YAML; `run.sh` writes it
into the par file (`wing_dt2e-5.yml`: the validation time step Δt = 2·10⁻⁵ at the same 1.24·10⁻³ c/U∞ per action). The released wing policies (`policies/naca0012_wing_re2e5/`, see `examples/nekrs/wing/README.md`)
are the TD3 policy of the HydroGym paper (Stable-Baselines3) and the TD3 policies N1 and N1 + F. A trip of 60 modes
renewed every 0.004 is available for sensitivity studies:

```bash
NP=8 RUN=runs/zero_trip60 TRIP_MODES=60 TRIP_INTERVAL=0.004 CONF=wing_dt8e-5.yml ./run.sh --policy zero
```

With this trip the boundary layer transitions later and C_D is 6.4 % lower, so results with it cannot be compared
with Ref1. At N = 7 the case needs about 20-28 GB per GPU on six GPUs. On aarch64 nodes (e.g. NVIDIA GH200) the case needs at least
8 NekRS ranks: with fewer ranks the static Nek5000 arrays per rank exceed the aarch64 relocation range. With fewer
than 8 GPUs per node, use `GPU_SHARE=1` (`examples/nekrs/gpu_share.sh`), e.g. 2 ranks per GPU on a node with 4 GPUs. `deploy.py` writes a restart chunk every
`--chunk` actions, so a run killed by the batch time limit continues from the same `RUN`.

### In an Apptainer image on a Slurm cluster

`examples/nekrs/wing/run_slurm.sbatch` (and `channel/train_slurm.sbatch`) run `run.sh` inside the image with
`apptainer exec --cleanenv` and pass the environment through `APPTAINERENV_*` variables: `APPTAINERENV_PYTHONPATH`
for the checkout and the GPU list that Slurm sets, `APPTAINERENV_ROCR_VISIBLE_DEVICES` (ROCm image, `--rocm`) or
`APPTAINERENV_CUDA_VISIBLE_DEVICES` (CUDA image, `--nv`). The image is set in `examples/nekrs/slurm_image.sh`:
`BACKEND=CUDA` (default) uses the HydroGym GPU image as a SIF, `hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif` from
`apptainer pull docker://clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` on x86_64 nodes and the GH200 image
`hydrogym-gh200.sif` on aarch64 nodes (`apptainer pull hydrogym-gh200.sif oras://docker.io/clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2-aarch64`, see [Container images](./containers)); `BACKEND=HIP` uses a ROCm image built
with `.devcontainer/scripts/build_rocm_image.sh` (Option 3); `SIF=<path>` selects another one. Do the same in your
own job scripts:

- Without `--cleanenv` the host `PATH` and `LD_LIBRARY_PATH` reach the image. A compiler of a loaded host module
  then builds the udf against the host libraries and the udf build fails, e.g. with `cannot find crt1.o`.
- `--env CUDA_VISIBLE_DEVICES=0,1,2,3` would split the comma-separated list; `APPTAINERENV_` keeps it whole.
- On one node, the image's own `mpirun` starts the controller and the NekRS ranks (the `mpirun` branch of
  `run.sh`); `srun --multi-prog` across nodes needs the PMIx variables of Slurm and therefore no `--cleanenv`.
- `bind_numa.sh` binds each NekRS rank to the NUMA domain of a CPU socket (`HG_BIND=0` switches it off). On GH200 the
  GPU memory appears as further NUMA domains without CPUs of their own; these are skipped, so the binding also
  works with more ranks than sockets.

## Actuator power

`NekRSFlowEnv` reports the power of the wall actuation with the keys of the common HydroGym power interface
(`hydrogym.power`): `actuation_power` (signed), `actuation_power_positive` (Hasegawa & Kasagi 2011),
`actuation_power_abs` (Bewley, Moin & Temam 2001), `actuation_power_per_actuator`, `drag_coefficient`,
`power_sampling` and `objective`, from both `reset()` and `step()`. At every wall node the udf returns the pressure
and the wall velocity it imposes. The power is
π = v_n (p − p̄) + v_n |u_w|²/2, with p̄ the instantaneous spanwise mean of the wall pressure at each station.
Each cost form is taken per node, integrated with the quadrature weights and divided by ½ρU³A. That is the
normalisation of the drag: c × span for the wing, so C_D + C_P is the total power coefficient. For the channel,
U is the bulk velocity and A the actuated wall area.

```yaml
power:
  reference: {velocity: 1.0, length: 1.0, span: 0.1}   # 0.5 rho U^3 c span
  drag: {boundaries: [1, 2], axis: x}                  # C_D from the force monitors
  power_sampling: mean     # mean over every solver step of an action (default) | end: after its last step
  power_cost: abs          # form charged in the reward: signed | positive | abs
  power_weight: 0.0        # reward <- reward - power_weight * C_P[power_cost]; 0 leaves the reward unchanged
```

`power_sampling: mean` advances the solver one step per request, which produces the same flow as one request for
all steps. Use it whenever the power is evaluated: the wall velocity jumps at every decision, and the pressure
response to the jump does work within the first steps after it, which a sample at the end of the action misses.
`power: false` switches the wall requests off.
