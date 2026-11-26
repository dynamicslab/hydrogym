# hydrogym.nekrs: HydroGym control interface for NekRS

Couples a NekRS case to a HydroGym-style Python controller over MPI (MPMD), in the same way as
`hydrogym/maia/mpmd_interface.py` does for m-AIA. Everything the agent sees and does is set in a YAML file and
sent at run time: probe locations, actuator layout (positions, footprint, blowing direction, amplitude,
zero net mass flux), the quantities monitored for the reward, and the initial conditions for each episode. The
NekRS case only marks the boundaries that may be actuated.

```
hydrogym/nekrs/
  udf/hydrogym_nekrs.hpp      solver side (included by the case udf)
  interface.py                NekRSInterface: MPI requests (m-AIA names: getProbeData, setControlProperties, ...)
  layout.py                   position blocks (list, file, grid, boundary_nodes, actuator_offset), footprints
  env.py                      NekRSFlowEnv (gymnasium.Env), YAML-driven
  multiagent.py               one agent per actuator: hydrogym.nek NekParallelEnv / PettingZoo, SB3 VecEnv
  multisim.py                 several simulations of one case feeding a shared policy (MultiSimVecEnv)
  power.py                    actuator power of wall transpiration (hydrogym.power interface)
examples/nekrs/
  podman.sh                   run a command in the image, with this repository mounted
  fetch_case.py               mesh and initial field of TCFlarge_3D_Re207 from Hugging Face
  mpmd/                       patched NekRS driver nekrs-mpmd (joins HydroGym's COMM_WORLD.Split)
  channel/                    channel case (Re_tau 207, pettingzoo_tcflarge_re207.yml)
    tcf.par, tcf.udf            case; [CASEDATA] hydrogymBoundaries = 1
    channel_re207.yml           environment description
    channel_td3.yml             + episodes and a snapshot library (short TD3 run)
    channel_paper.yml           runner settings of the HydroGym channel training (2500 actions per episode),
                                loads the library
    client.py                   zero / opposition / random policies, logging
    train_td3.py                TD3 (Stable-Baselines3) through the multi-agent wrapper
    baselines.py                zero / opposition control from every initial condition of the library
    run.sh                      MPMD launch (host: through podman.sh)
    tests/                      under-resolution, reset and TD3 smoke tests
  legacy_drl/                 NekRS port of the Nek5000 DRL coupling (unchanged hydrogym.nek.NekEnv)
```

## Container

The HydroGym images (`clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2`, x86_64 NVIDIA GPUs; `clagemann/hydrogym-cpu:1.0.2`
with the serial OCCA backend; the GH200 Apptainer image) contain NekRS 26.0 in `/opt/nekrs` (`NEKRS_HOME`, CUDA-aware
Open MPI), `nekrs-mpmd` on the `PATH` and HydroGym in `/opt/venvs/hydrogym` (also `/opt/venvs/nekrs`), see
`docs/docs/installation/containers.md`. The `run.sh` scripts run as is inside them.
`examples/nekrs/podman.sh` runs a command in the GPU image (`ENGINE=docker` for docker) with this repository
mounted over the image copy `/opt/hydrogym` (`MOUNT_REPO=0` uses the copy in the image); on a host without
`NEKRS_HOME`, `run.sh` re-runs itself through it. Outside the images, build the driver with
`examples/nekrs/mpmd/build.sh`; `run.sh` then falls back to `../mpmd/nekrs-mpmd`.

One GPU per NekRS rank by default; `--device-id 0` on the `nekrs-mpmd` command line puts several ranks on one GPU
(`NEKRS_MPMD` pointing at a wrapper that adds it). GPU-aware MPI needs UCX's CUDA transports: if `UCX_TLS` is
restricted, keep `cuda_copy` (and `cuda_ipc` for several GPUs). On hosts with broken GPU peer-to-peer access
(IOMMU), restrict UCX to host-staged transfers, e.g. `UCX_TLS=shm,self,tcp,cuda_copy`.

## Case setup

par file:

```ini
[CASEDATA]
hydrogymBoundaries = 1, 3      # boundary ids that may be actuated (velocity boundaryTypeMap = udfDirichlet)
```

udf:

```cpp
#include "../../../hydrogym/nekrs/udf/hydrogym_nekrs.hpp"   // relative to the case directory
#ifdef __okl__
void udfDirichlet(bcData *bc)
{
  if (isField("fluid velocity")) {
    bc->uxFluid = bc->usrwrk[bc->idxVol + 0 * bc->fieldOffset];
    bc->uyFluid = bc->usrwrk[bc->idxVol + 1 * bc->fieldOffset];
    bc->uzFluid = bc->usrwrk[bc->idxVol + 2 * bc->fieldOffset];
  }
}
#endif
void UDF_Setup0(MPI_Comm comm, setupAide &options) { hgnek::setup0(); }
void UDF_Setup() { hgnek::setup(); }
void UDF_ExecuteStep(double time, int tstep) { hgnek::executeStep(time, tstep); }
```

Set `numSteps` large: the controller ends the run (TERMN).

## Launch

```bash
mpirun -n 1 python controller.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup case.par --backend CUDA
```

For the channel, `examples/nekrs/channel/run.sh` does this in the HydroGym image (one GPU per NekRS rank). It
works from the host (it re-runs itself through `../podman.sh`) and inside the container:

```bash
cd examples/nekrs && python fetch_case.py && cd channel     # once: tcf.re2, rs6tcf0.f00004
GPU=0 NP=1 RUN=runs/zero ./run.sh --policy zero --steps 1500            # client.py
GPU=0 NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --episodes 10     # any controller script
```

```python
from hydrogym.nekrs import NekRSFlowEnv

env = NekRSFlowEnv("channel_re207.yml", run_root="runs/x")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)  # action in [-1, 1] per actuator
env.close()  # always: NekRS waits for TERMN
```

## YAML reference

```yaml
base: other.yml                 # optional, merged underneath (next to this file, else relative to the cwd)
environment:
  nproc: 1                      # NekRS ranks
  dt: 4.0e-5                    # optional: checked against the .par time step
  steps_per_action: 13          # solver steps per action
  episode_actions: 300          # episode length (truncation); 0 = never
  target_cfl: 2.0               # an action whose solver steps reach this CFL ends the episode (terminated)
actuators:
  boundaries: [1]               # subset of hydrogymBoundaries
  positions: {source: boundary_nodes, boundary: 1, nodes: all, sort: [x]}
  footprint: {type: point}      # | {type: gaussian, sigma: [sx, null, sz]} | {type: box, half_width: [...]}
  direction: normal             # | {vector: [vx, vy, vz]} | {angle: deg, towards: [vx, vy, vz]}
  amplitude: 0.063              # wall velocity for action 1
  znmf: uniform                 # none | uniform (remove the net flux over the boundaries) | actions
  resolution_check: {mode: error, min_nodes_per_width: 2.0}   # error | warn | off
  # or an explicit list with per-jet overrides:
  # jets: [{position: [1, 0, 1], footprint: {...}, direction: {...}, amplitude: 0.1}, ...]
probes:
  fields: [u, v]                # u v w p
  positions: {source: actuator_offset, offset: [0, 0.0724637681159420, 0]}
  subtract_plane_mean: y        # optional: subtract the mean over planes normal to this axis (planar_avg)
  # subtract_group_mean: [x, y]  # optional: mean over the probes sharing x and y, weighted by the actuator footprint
  normalization: 0.0630         # one value, one per field, or u_tau (from wall_units)
  frame: global                 # global (default) | wall: u, v -> wall tangent u_t and normal v_n at each actuator |
                                # hg_fortran: u_t = u n_y + v n_x (projection of the Nek5000 wing coupling)
wall_units:                     # optional
  nu: 1.0e-5                    # default: mu / rho of the solver
  u_tau: 0.0624                 # a value | {polynomial: [c0, c1, ...], axis: x}
monitors:                       # optional: further wall quantities recorded at every solver step (solver_steps.csv)
  - {boundary: 2, quantity: force}   # force | gradient
reward:
  type: wall_shear
  boundary: 1
  quantity: gradient            # mean du/dn (as Nek5000 aeroForces) | force (viscous force / (mu A))
  baseline_dudy: 13.10          # r = 1 - <dU/dy> / baseline
  average: legacy               # running mean of drl_reward.f | mean
initial_conditions:             # episode reset, see below
  source: snapshots             # continue (default) | snapshots | files
  snapshots: {count: 6, interval: 1300, warmup: 0, save: runs/td3/snaps/tcf, load: null}   # paths relative to the case directory
  files: ["restart/*.f00001"]   # source files: NekRS field files
  interpolate: false            # files from a different mesh (NekRS +int)
  selection: random             # random (env.np_random, seeded by reset(seed)) | cycle | first
  settle_steps: 0               # zero-action solver steps after a reset
power:                          # actuator power (hydrogym.power info keys); every key optional, `power: false` = off
  boundaries: [1]               # default: actuators.boundaries
  pressure_mean: [x, y]         # p' = p - mean over the nodes sharing x and y (spanwise mean); [] = full pressure
  reference: {velocity: 1.0}    # C_P = P / (0.5 rho U^3 A); A = length * span, area, or (default) the boundary area
  drag: {boundaries: [1, 2], axis: x}   # optional: drag_coefficient from force monitors, same reference
  power_sampling: mean          # mean over every solver step of the action | end (after its last step)
  power_cost: abs               # signed | positive | abs, the form charged in the reward
  power_weight: 0.0             # reward - power_weight * C_P[power_cost]; 0 = reward unchanged
```

Actuator power (`power.py`): WALLS returns the pressure and the wall velocity the udf imposes at every face node of
the power boundaries. From them the env forms pi_p = v_n p' and pi_k = v_n |u_w|^2 / 2 per node. The signed
form is the sum of both. The positive form (Hasegawa & Kasagi 2011) keeps only the positive parts of each, and the
absolute form (Bewley, Moin & Temam 2001) takes their absolute values. Each is integrated with the quadrature
weights and normalised like C_D. `info` from `reset()` and `step()` carries `actuation_power`,
`actuation_power_positive`, `actuation_power_abs`, `actuation_power_per_actuator` (nearest actuator centre),
`actuation_power_full_pressure`, `drag_coefficient`, `power_sampling` and `objective`. With `power_sampling: mean`
the env requests one solver step at a time, which gives the same flow. multisim advances its instances in lockstep.
`env.set_step_hook(every, fn)` calls `fn(env, row)` after every solver step of an action whose index is a multiple of
`every` (the solver is idle during the call, e.g. for `iface.wall_data`); the actions then also advance one solver step
per request. The wing example uses it to sample the walls every 10 solver steps (`deploy.py --wall-steps`).

Position sources: `list` (points), `file` (.npy, .csv with x,y,z, or text), `grid` (x, y, z as value, list or
{start, stop, num}; `order`), `boundary_nodes` (GLL nodes from the solver: `all` keeps copies of shared
edge/corner nodes as in the released HydroGym channel, `facindf`, `unique`), `actuator_offset` (probes only:
`offset` and/or `normal_distance`). `sort: [x, z, ...]` orders them stably, first key primary. With one probe per
actuator, the observation is probe-major, so agent i gets block i.

Resolution check: for gaussian and box footprints the solver counts the GLL nodes per footprint width (FWHM
or full width) along each tangential direction. A footprint with fewer than `min_nodes_per_width`, or with no
node at phi >= 0.5, is under-resolved. `error` rejects the layout and NekRS stops; `warn` gives a Python warning.
Flags and counts go to `actuators.csv`. An actuator that covers no node at all is always an error.

Outputs in `run_root`: `config.yml`, `actuators.csv`, `probes.csv`, `steps.jsonl` (per action: episode,
reward, dUdy, CFL range, net flux, max wall velocity, ...), `solver_steps.csv` (per solver step: tstep, time,
dt, CFL, monitors), `episodes.jsonl` (per reset: library index, state time, solver step).

## Episode reset

`env.reset()` starts each episode from an initial condition in a library:

- **snapshots**: exact copies of the complete time-stepper state on the GPU. This covers all lag levels of
  u and p, the explicit and advection terms, the convecting velocity, the properties and dt[3]. The library
  is built on the first reset: zero actuation, `warmup` steps, then one snapshot every `interval` steps. With
  `save`, each state is also written as one binary file per rank (`<save>_000_r00000`, ...). With `load`, the
  library is read back from those files. The files are checked against the partition, polynomial order,
  buffer sizes and mesh. Restoring a snapshot and replaying the same actions reproduces the trajectory to
  solver tolerance. This has been tested: reward agrees to 4e-10, observations to 4e-8 relative.
- **files**: NekRS field files (u, p) via `restartFromFiles`. A field file has no time history, so the lags
  are rebuilt as a frozen history (all levels equal the loaded state). The first step then behaves like
  BDF1/EXT1 with a shorter effective step, so a few `settle_steps` are advisable. A trajectory from a reset
  differs slightly from a NekRS start from the same file: about 2e-3 in reward over 5 actions, because NekRS
  ramps BDF1→2→3 at the start.
- **continue**: no reset; the flow runs on from the current state. This is the RSETS behaviour of the
  Nek5000 coupling.

The solver clock and step counter keep running across resets; `info["state_time"]` reports the time of the
restored state. The actuation is zeroed at every reset. The projection spaces (initial guesses of the
pressure and velocity solvers) are not reset. This only affects the iteration counts and round-off at the
level of the solver tolerance. Not supported yet: scalars, moving meshes, neknek. Field files can't be used
with advection subcycling (snapshots can).

Low level: `iface.save_snapshot(slot)`, `reset_to_snapshot(slot)`, `reset_from_file(path, interpolate)`,
`write_snapshot(slot, path)`, `read_snapshot(slot, path)`.

## Multi-agent / RL

```python
from hydrogym.nekrs.multiagent import make_parallel_env, make_pettingzoo_env, ParallelToVecEnv

par = make_parallel_env(env)  # hydrogym.nek.parallel_env.NekParallelEnv if hydrogym is importable
vec = ParallelToVecEnv(par)  # SB3 VecEnv, one sub-env per agent (as supersuit.pettingzoo_env_to_vec_env_v1)
```

`NekRSFlowEnv` exposes what the hydrogym.nek wrappers expect from NekEnv: `actuator_info` (NID = rank,
GLLID = element, FACEID, ix, iy, iz), `n_actuators`, `obs_per_actuator`, `action_space` and
`info["reward_per_actuator"]`. Agent names are therefore the same as with Nek5000. `make_pettingzoo_env` needs
the `pettingzoo` package (in `hydrogym[nekrs]` and the HydroGym images); `ParallelToVecEnv` doesn't need it.

`examples/nekrs/channel/train_td3.py` follows `examples/nek/getting_started/3_pettingzoo/train_sb3_pettingzoo.py`
and the runner block of `pettingzoo_tcflarge_re207.yml`. One difference: the action noise is drawn
independently per agent (`VectorizedActionNoise`). A plain `NormalActionNoise` would broadcast one sample to
all agents.

## Wire protocol (version 4)

Each request is a 5-character command (tag 22), an int payload (23) and a double payload (24). Each reply is
ints `[status, ...]` (25), doubles (26) and a message (27). Status 0 means ok; a non-empty message with status 0
is a warning. Status 1 means the request was rejected and nothing changed. Status 2 means a resolution error.
The commands are HELLO, BNDRY, PRBDF, PROBE, ACTDF, ACTUA, MONDF, STEPS, SNAPS, RESET, SNAPF, WALLS, WRFLD and
TERMN (WALLS flags since version 4: 1 geometry, 2 imposed wall velocity, 4 pressure only). The
payloads are documented at the top of `hydrogym_nekrs.hpp`.

## Validation (channel Re_tau 207, A6000)

- Observations are bit-identical to the legacy Nek5000/NekRS coupling (with `client.py --warmup 2`).
- The wall-gradient monitor matches `aeroForces` dU/dy to 1e-10.
- The reported CFL matches the NekRS printout.
- NP1 and NP4 (rcb partition) give the same actuator order, observations, rewards and CFL to about 1e-10.
- Over 1500 actions, mean reward of the last third vs legacy: zero actuation 0.0185 vs 0.0174, opposition
  control 0.2366 vs 0.2351.
- Under-resolved jets are rejected in `error` mode (NekRS stops cleanly) and only warned about in `warn` mode.
- Episode reset: see above (`examples/nekrs/channel/tests/reset_test.py`).

## Gotchas

- NekRS rebuilds the udf only when the `.udf` file changes, not when an included header does. `run.sh` deletes
  `.cache/udf` when `hydrogym_nekrs.hpp` is newer than the library. Do the same in other launchers.
- If the controller fails, NekRS waits forever for the next request. `NekRSFlowEnv.__init__` sends TERMN when
  the setup is rejected. Controller scripts should call `env.close()` in a `finally` block.
- From the host, `run.sh` passes its arguments through `podman.sh` (`bash -lc "... $*"`), so arguments with spaces
  are split. Put overrides in a YAML file (`base:`) instead of on the command line.
- The legacy reward window is one solver step earlier: it includes the state before the action. The generic
  reward uses the `steps_per_action` steps after the action. The two differ by about 2e-5 per action.
- `episode_actions` is a truncation (time limit), not a terminal state. Only a CFL violation terminates.
