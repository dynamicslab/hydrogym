# NACA0012 wing (Re_c = 200k) with the generic NekRS interface: zero-shot deployment

This directory runs controllers on the HydroGym environment `NACA0012_3D_Re200000_AOA0`:
- the channel-trained MARL TD3 policies (zero-shot), among them the released wing policies of `policies/`;
- opposition control (OC);
- uniform blowing (UB);
- the uncontrolled flow.

The case is a NekRS port of the Nek5000 case `third_party/nek5000/cases/naca0012_200k`; the coupling follows the case's
`drl_*.f` user routines.

| file | |
|---|---|
| `naca_wing.par/.udf/.usr` | case at N = 7. The udf ports the Nek5000 case:<br>• frozen far field and outlet pressure;<br>• trip of `tripf.f`;<br>• RT filter, via `regularization = hpfrt`;<br>• bid 2 is the suction-side control window, 0.25 < x/c < 0.86. |
| `naca_wing_smoke.par`, `wing_smoke.yml` | N = 3 smoke test from a uniform free stream on 1 GPU. |
| `wing.yml` | the deployment of the Nek5000 case (`naca_wing.rea`, `drl_*.f`):<br>• dt = 4e-5, 31 solver steps (1.24e-3 c/U) per action;<br>• u_tau(x) is the polynomial `X2Utau` of `drl_init.f` (uncontrolled wing, Re_c = 2·10⁵);<br>• one agent per wall node of the window;<br>• obs [u_t'/u_tau(x), v_n'/u_tau(x)] at y+ = 15 along the wall tangent and normal (`frame: wall`);<br>• action 1 = u_tau,ch = 0.0624334733 (training channel), normal blowing;<br>• zero net flux per spanwise line. |
| `wing_oc.yml` | opposition control: dt = 2e-5, 10 steps (2e-4 c/U) per action, v_w = -v_n' at y+ = 15, not clipped. |
| `wing_dt2e-5.yml` | `wing.yml` at dt = 2e-5 with 62 steps (1.24e-3 c/U) per action: the time step of the validation against Ref1. |
| `wing_dtp.yml` | `wing.yml` with the control interval of the training channel in wall units (dt+ = 0.67); the step count follows from the mean u_tau(x) of the agents. |
| `wing_dt8e-5.yml` | `wing.yml` at dt = 8e-5 with 25 steps (2e-3 c/U) per action (uncontrolled runs, trip sensitivity). |
| `deploy.py` | runs one controller in chunks and records the span-averaged wall traction and pressure, and the actuator power and C_D of every action (env `power` section). |
| `postprocess.py` | Cf(x/c) on the suction side, Cd and Cd,f (time series and mean), and the change against the first run. |
| `fig4d.py` | the quantities of Fig. 4d of the HydroGym paper: drag reductions of controlled runs against a paired uncontrolled run over a window of control time: local skin-friction reduction on the suction side (0.25-0.858), C_d,f and C_d of the complete wing surface (time means over every solver step, from `solver_steps.csv`) and of the stations 0.0197-0.995, c_f(x) with U_inf and with U_e; mean and 95 % interval over start states, paired differences between controllers. |
| `action_hist.py` | distribution of the actuation: recomputes each agent's action from the observations stored with the wall samples (the run's controller, then the env's clip, amplitude and zero net flux per spanwise line); policy output and applied velocity v_w / u_tau(x): share at the bound, blowing / suction, share beyond ±u_tau(x), kurtosis, rms, histograms and a figure. Needs runs with the observations recorded (no `--no-obs`). |
| `power_forms.py` | actuation power of the same runs in the signed, positive and absolute forms: per cent of the uncontrolled friction power of the strip 0.25-0.86, net saving on the strip and of the whole wing (drag reduction against power). |
| `obs_shift.py` | wing against channel training distribution:<br>• actual u_tau(x) against the polynomial, with the probe y+ and dt+ per action that follow;<br>• observation statistics per x/c bin against channel samples from `../channel/client.py --record-every`. |
| `fit_utau.py` | fits u_tau(x) to an uncontrolled run and writes an env YAML (`base: wing.yml`) that uses it; `--dt-plus` also sets the control interval in wall units. |
| `bl_stats.py` | time- and span-averaged statistics from the restart fields (`chk*0.f00000`) into `RUN/field_stats.npz`: mean flow, all Reynolds stresses, 3rd/4th moments, Reynolds-stress budgets (exact element derivatives), wall-normal profiles at every x/c, wake lines, spanwise z-lines (spectra, correlations, pointwise wall shear). |
| `bl_report.py` | boundary-layer report of one or more runs (`report.md`, `summary.json`, CSV, figures): trip, forces, Cp, Cf (mean, rms, spectra), profiles, BL integrals vs x/c, transition, spectra and correlations, budgets, wake, resolution, convergence, snapshots. |
| `bl_stats_slurm.sbatch` | both on the CPUs of one cluster node (`CASES="runs/zero_trip60:500 runs/zero_ref1:2420"`). |
| `../perf_report.py` | runtime and performance numbers from the logs of a run: case size, setup and JIT, NekRS step time, solver iterations, udf share, throughput per GPU, seconds and GPU hours per action, training episodes, and the GPU use/power sampled by the sbatch scripts (`GPU_MONITOR`, `gpu-JOB.log`). |
| `run.sh` | MPMD launch in `RUN/`; continues from the last chunk. The solver dt is `environment.dt` of the YAML (`conf_dt.py`). |
| `run_slurm.sbatch` | the same on one GPU node of a Slurm cluster in a HydroGym image (`../slurm_image.sh`: the GPU image by default, `BACKEND=HIP` for a ROCm image), with automatic resubmission; on aarch64 nodes at least `NP=8`, with `../gpu_share.sh` when the node has fewer GPUs. |
| `prepare_case.py` | gets the 21 GB initial field `rs8naca_wing0.f00008` into `restart/`. |

## Run

```bash
python prepare_case.py                     # or --source <local NACA0012_3D_Re200000_AOA0> (hard link)
GPU=0 NP=1 RUN=runs/smoke PAR=naca_wing_smoke.par CONF=wing_smoke.yml ./run.sh --policy oc --actions 4 --chunk 2

# released policies (policies/MANIFEST.yaml)
GPU=0,1,2,3 NP=4 RUN=runs/td3    ./run.sh --policy td3:naca0012_wing_re2e5/td3/seed1998 --actions 2500 --chunk 250
GPU=0,1,2,3 NP=4 RUN=runs/n1     ./run.sh --policy td3:naca0012_wing_re2e5/td3_N1/seed0
GPU=0,1,2,3 NP=4 RUN=runs/n1f    ./run.sh --policy td3:naca0012_wing_re2e5/td3_N1F/seed3
GPU=0,1,2,3 NP=4 RUN=runs/zero   ./run.sh --policy zero
GPU=0,1,2,3 NP=4 RUN=runs/oc     CONF=wing_oc.yml ./run.sh --policy oc --alpha 1
GPU=0,1,2,3 NP=4 RUN=runs/ub     ./run.sh --policy ub --ub-velocity 0.001
# a policy trained with ../channel
GPU=0,1,2,3 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$PWD/../channel/runs/td3_s0/td3_final.zip

python postprocess.py runs/zero runs/oc runs/ub runs/td3 --skip 500 --out wing.png
# drag reductions and actuation power against the uncontrolled run over t - 28 = 1-3 c/U (start of control: t = 28);
python fig4d.py --ref runs/zero --t0 1 --t1 3 --diff TD3-OC --out fig4d TD3=runs/td3:28 OC=runs/oc:28 UB=runs/ub:28
python power_forms.py --ref runs/zero --t0 1 --t1 3 --out power TD3=runs/td3:28 N1=runs/n1:28 OC=runs/oc:28
python action_hist.py --t0 1 --t1 3 --out action_hist TD3=runs/td3:28 N1=runs/n1:28 OC=runs/oc:28
python bl_stats.py runs/zero --skip 500 --workers 16 && python bl_report.py runs/zero --skip 500 --out bl_report

# channel reference samples (600 actions from snapshot 0, obs every 10), then the comparison
(cd ../channel && GPU=0 NP=1 RUN=runs/obs_zero ./run.sh --conf channel_paper.yml \
    --policy zero --steps 600 --record-every 10 --index 0)
(cd ../channel && GPU=0 NP=1 RUN=runs/obs_td3_s0 ./run.sh --conf channel_paper.yml \
    --policy td3:runs/td3_s0/td3_final.zip --steps 600 --record-every 10 --index 0)
python obs_shift.py runs/td3_s0 --channel ../channel/runs/obs_zero ../channel/runs/obs_td3_s0 --skip 500
```

On a Slurm cluster, submit from this directory (`run_slurm.sbatch` takes the controller and the number of actions
as the environment variables `POLICY` and `ACTIONS`; `run.sh` takes them as `--policy` and `--actions`):

```bash
POLICY=zero RUN=runs/zero sbatch run_slurm.sbatch
POLICY=td3:<policy zip> RUN=runs/td3 BIND=<directory of the zip> sbatch run_slurm.sbatch
```

The defaults (`naca_wing.par`, `wing.yml`) are the case and deployment of the Nek5000 wing
(`third_party/nek5000/cases/naca0012_200k`): Re_c = 2·10⁵, the trip of Ref1 (Wang, Atzori & Vinuesa, JFM 2025:
30 spanwise modes renewed every 0.008 c/U, amplitude 10, at x/c = 0.1), the start field of Ref1 at t = 28
(`rs8naca_wing0.f00008`, written at dt = 2e-5 with this trip), dt = 4e-5 and 31 solver steps (1.24e-3 c/U) per
action, with the observations in the wall-tangent frame. Opposition control runs with `wing_oc.yml` (dt = 2e-5, 10 steps per action). The uncontrolled flow with this
trip is validated against Ref1 (docs page "NekRS NACA0012 validation", at dt = 2e-5 and 8e-5).

```bash
# the validation run: Ref1's trip at dt = 2e-5 (wing_dt2e-5.yml), t = 28-48 at 1.24e-3 c/U per action
POLICY=zero RUN=runs/zero_ref1 CONF=wing_dt2e-5.yml ACTIONS=16130 sbatch run_slurm.sbatch
python bl_stats.py runs/zero_ref1 --skip 2420 --workers 16 && python bl_report.py runs/zero_ref1 --skip 2420
# the same at the time step of the deployment (wing.yml, dt = 4e-5)
POLICY=zero RUN=runs/zero_dt4e-5 ACTIONS=16130 sbatch run_slurm.sbatch
# 60 trip modes every 0.004, the trip-sensitivity run of the validation
POLICY=zero RUN=runs/zero_trip60 TRIP_MODES=60 TRIP_INTERVAL=0.004 CONF=wing_dt8e-5.yml sbatch run_slurm.sbatch
```

## Released policies

`policies/naca0012_wing_re2e5/` (index `policies/MANIFEST.yaml`, loader `policies/load_policy.py`):

| id | policy | file |
|---|---|---|
| `naca0012_wing_re2e5/td3/seed1998` | the TD3 policy of the HydroGym paper (Stable-Baselines3) | `eval_model_lc_omega1_SS_BASE_dr.zip`, and `actor.npz` for the numpy loader |
| `naca0012_wing_re2e5/td3_N1/seed0` | TD3 policy N1 | `actor.npz` |
| `naca0012_wing_re2e5/td3_N1F/seed3` | N1 + F: F on 0.2 <= x/c < 0.48, N1 on 0.48 <= x/c <= 0.9 | `blocks.json`, `actor_F.npz`, `actor_N1.npz` |

All three map the observation [u_t'/u_tau(x), v_n'/u_tau(x)] of one agent to its action in [-1, 1], in units of
u_tau,ch, and run with `wing.yml`. `--policy td3:<id>` runs the SB3 zip with Stable-Baselines3 when the entry has one
and SB3 is installed (as in the HydroGym images), otherwise the numpy actor (equal to SB3 to 3e-7).

The run directory `RUN/` holds:
- the case links;
- `bc_frozen0.f00000`, the far field frozen at the first launch;
- `trip_origin.txt`;
- the chunk fields `chk<actions>0.f00000` and `state.json`;
- `walls_geometry.npz` and `walls_<from>_<to>.npz`: the wall samples (span-averaged traction and pressure per
  station, `b1`, `b2`, and the force on each boundary, `F1`, `F2`) after every solver step whose index is a multiple
  of `--wall-steps` (default 10), also within the actions, with `time`, `step` and `action` (the running action).
  Their mean is the time mean; a policy that changes its action in steps makes the friction vary within the action,
  so samples only at the end of each action are biased (`--wall-steps 0` samples only after the last step of every
  `--walls-every`-th action). The observations of all agents at the end of every `--walls-every`-th
  action (`obs`, float16, with `obs_action` and `obs_time`; about 14 MB per chunk of 250 actions; `--no-obs` to
  skip; files of `--wall-steps 0` runs hold `obs` aligned with the wall samples);
- `agents.npz`, the agent and probe positions and the observation scaling;
- `log<actions>/`, the env logs of each launch; `solver_steps.csv` holds the viscous and pressure forces of the wing
  at every solver step. `fig4d.py` takes C_d,f and C_d from them: a policy that changes its action in steps (TD3)
  makes the friction vary within an action, so the samples at the end of the action are not the time mean
  (`dCDf_end`, `dCD_end` in its JSON, for comparison).

A relaunch with the same `RUN` continues the realization:
- it keeps the frozen far field;
- it replays the trip random sequence from its origin;
- it starts from the last chunk.

The time integration restarts at first order, because the field files carry no lagged states.

## Choices to know

- **Wall units.** The action keeps the wall units of the training channel: action 1 is its friction velocity
  u_tau,ch = 0.0624 (Re_τ 206), as in `drl_action.f`. The observations are divided by the wing's local u_tau(x),
  the polynomial of the uncontrolled wing at Re_c = 2·10⁵, which also places the probes at y+ = 15.
- **Control interval.** `steps_per_action: 31` at dt = 4e-5 is the `ndrl` interval of the Nek5000 case (p089 and
  p012 of `naca_wing.rea`), 1.24e-3 c/U. The channel policies were trained with 13 steps of dt+ ≈ 0.052, about 0.67
  viscous time units. `wing_dtp.yml` (`steps_per_action: {dt_plus: 0.67}`) gives the channel's interval in wall units
  instead.
- **Agent count.** The Nek5000 case (`register_wall_pts`) registers every grid point of the wall faces whose midpoint
  lies in the window, so points shared by two faces are counted twice. Here every distinct wall GLL node of the window
  is an agent: 12931 at N = 3, more at N = 7. The policy is local, so this only changes the actuation resolution. Other
  agent sets can be given as a position file (`positions: {source: file}`).
- **Actuation limit.** The policies act in [-1, 1] × u_tau,ch. OC with `wing_oc.yml` is bounded at 3 u_tau(x), which
  it does not reach (with `wing.yml` it would be clipped where |v'| > u_tau,ch). UB blows `--ub-velocity` U_inf without
  zero net flux.
- **Trip.** `[CASEDATA] tripModes` and `tripInterval` (`TRIP_MODES`, `TRIP_INTERVAL` in `run.sh` and
  `run_slurm.sbatch`) set the spanwise mode count and the time between new random signals (tdt).
  - The default is the trip of Ref1, `naca_wing.wall` and the restart files: 30 modes every 0.008.
  - 60 modes every 0.004 transition later, C_D 6.4 % lower.
  - `DT` sets the solver time step of a YAML without `environment.dt`. With one, the env stops if the two differ: for
    another time step use a YAML that sets it, e.g. `wing_dt2e-5.yml`, `wing_dt8e-5.yml`, `wing_oc.yml`.
  - `trip_origin.txt` records dt, the mode count and tdt. A relaunch with different values stops.
- **Statistics.** Cf and Cd are averaged over the span and over the samples after `--skip` actions. Use the same
  window for every controller.
- **Actuator power.** The `power` section of `wing.yml` gives C_P = P / (0.5 rho U^3 c span) in the signed,
  positive (Hasegawa & Kasagi) and absolute (Bewley et al.) forms with p' about the spanwise mean, and the whole-wing
  C_D. The net saving against an uncontrolled run is S = 1 - (C_D + C_P) / C_D0. `--power-sampling mean` (default)
  averages over every solver step, `end` samples after the last step of each action; the step average includes the
  work of the pressure response to each jump of the wall velocity, which the end sample misses.
- **Probe frame.** `frame: wall` (`wing.yml`, default) rotates (u, v) into the downstream wall tangent and the wall
  normal.

## Tested

- **Smoke test** at N = 3, 1 GPU, with OC:
  - the udf compiles;
  - frozen far field and trip are active (138864 forced nodes);
  - 12931 agents and probes;
  - the WALLS force on bid 2 equals the aeroForces monitor;
  - the resume from a chunk works, including the frozen-field read and the trip replay.
- **Full resolution:** 2 OC actions at N = 7 from `rs8naca_wing0.f00008` (t = 28.0) on 6 RTX A6000:
  - about 20 GB per GPU at setup, about 28 GB per GPU while stepping;
  - about 0.95 s per step, so about 38 s per action and about 26 h for 2500 actions;
  - 69595 agents in 449 spanwise lines;
  - dUdy 470 against the Nek5000 baseline of 480.5;
  - Cd 0.0139, Cd,f 0.0110, Cl 0.005 (instantaneous).
- **Field sizes:** each chunk and the frozen field are 6.3 GB.
- **Defaults and released policies** (8 NVIDIA GH200 GPUs on 2 nodes, NP = 8): 3 actions each of
  `td3:naca0012_wing_re2e5/td3/seed1998`, `td3_N1/seed0`, `td3_N1F/seed3` (`wing.yml`) and OC (`wing_oc.yml`) from
  `rs8naca_wing0.f00008` with the trip of `naca_wing.wall`. Rewards, dU/dy, CFL, wall velocities and the force monitors
  of every solver step (93 steps, OC 30) are bit-identical to runs of the same configuration made with the case files
  and policy files before their release.
- **Actuator power** (8 NVIDIA GH200 GPUs): OC for 6 actions from t = 28 with the trip of `naca_wing.wall`
  (1119008 forced nodes, 30 modes every 0.008), dt 2e-5, 30 steps per action (a test configuration; `wing_oc.yml`
  uses 10), 69595 agents, power on 90112 face nodes of bid 2:
  - `--power-sampling mean` and `end` in two launches give bit-identical rewards and dUdy (stepping one by one
    does not change the flow), about 4.4 against 4.2 s per action;
  - C_D from the aeroForces monitors (end of action) equals the WALLS quadrature of deploy.py to 5 digits
    (0.013957 / 0.0139577, 0.014021 / 0.0140213, 0.014074 / 0.0140749 at actions 2, 4, 6);
  - C_P of OC, step mean against end of action: signed 1.4e-3 / 6.0e-5 at the first action (jump from zero),
    1.7e-5 / 2.8e-5 at action 6; abs 3.2e-3 / 1.5e-4 and 9.0e-4 / 1.1e-4. The step mean contains the work of the
    pressure response to each jump of the wall velocity, which the end sample misses.
