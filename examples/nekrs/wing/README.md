# NACA0012 wing (Re_c = 200k) with the generic NekRS interface: zero-shot deployment

This directory runs controllers on the HydroGym environment `NACA0012_3D_Re200000_AOA0` (paper Fig. 4):
- the channel-trained MARL TD3 policies (zero-shot);
- opposition control (OC);
- uniform blowing (UB);
- the uncontrolled flow.

The case is a NekRS port of the Nek5000 case `third_party/nek5000/cases/naca0012_200k`; the spec is in
`../notes/naca0012_wing_spec.md`.

| file | |
|---|---|
| `naca_wing.par/.udf/.usr` | case at N = 7. The udf ports the Nek5000 case:<br>• frozen far field and outlet pressure;<br>• trip of `tripf.f`;<br>• RT filter, via `regularization = hpfrt`;<br>• bid 2 is the suction-side control window, 0.25 < x/c < 0.86. |
| `naca_wing_smoke.par`, `wing_smoke.yml` | N = 3 smoke test from a uniform free stream on 1 GPU. |
| `wing.yml` | agents, probes and reward in wall units:<br>• u_tau(x) is the polynomial of `drl_init.f`;<br>• one agent per wall node of the window;<br>• obs [u_t'/u_tau, v_n'/u_tau] at y+ = 15;<br>• action 1 = u_tau, normal blowing;<br>• zero net flux per spanwise line. |
| `deploy.py` | runs one controller in chunks and records the span-averaged wall traction and pressure. |
| `postprocess.py` | Cf(x/c) on the suction side, Cd and Cd,f (time series and mean), and the change against the first run. |
| `run.sh` | MPMD launch in `RUN/`; continues from the last chunk. |
| `run_aac.sbatch` | the same on the AAC MI300A nodes, with automatic resubmission. |
| `prepare_case.py` | gets the 21 GB initial field `rs8naca_wing0.f00008` into `restart/`. |

## Run

```bash
python prepare_case.py                     # or --source <local NACA0012_3D_Re200000_AOA0> (hard link)
GPU=1 NP=1 RUN=runs/smoke PAR=naca_wing_smoke.par CONF=wing_smoke.yml ./run.sh --policy oc --actions 4 --chunk 2

POL=$PWD/../channel/runs/paper_s0/td3_final.zip
GPU=1,2,4,5 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$POL --actions 2500 --chunk 250
GPU=1,2,4,5 NP=4 RUN=runs/zero   ./run.sh --policy zero
GPU=1,2,4,5 NP=4 RUN=runs/oc     ./run.sh --policy oc --alpha 1
GPU=1,2,4,5 NP=4 RUN=runs/ub     ./run.sh --policy ub --ub-velocity 0.001

python postprocess.py runs/zero runs/oc runs/ub runs/td3_s0 --skip 500 --out fig4.png
```

On AAC, submit from this directory:

```bash
POLICY=zero RUN=runs/zero sbatch run_aac.sbatch
POLICY=td3:$HOME/policies/paper_s0/td3_final.zip RUN=runs/td3_s0 sbatch run_aac.sbatch
```

The run directory `RUN/` holds:
- the case links;
- `bc_frozen0.f00000`, the far field frozen at the first launch;
- `trip_origin.txt`;
- the chunk fields `chk<actions>0.f00000` and `state.json`;
- `walls_geometry.npz` and `walls_<from>_<to>.npz`, the samples every `--walls-every` actions;
- `log<actions>/`, the env logs of each launch.

A relaunch with the same `RUN` continues the realization:
- it keeps the frozen far field;
- it replays the trip random sequence from its origin;
- it starts from the last chunk.

The time integration restarts at first order, because the field files carry no lagged states.

## Choices to know

- **Control interval.** `steps_per_action: 40` is `ndrl` of the Nek5000 case, i.e. 6e-4 c/U, about 0.28 viscous
  time units at the window's mean u_tau. The channel policies were trained with 13 steps of dt+ ≈ 0.046, about 0.6
  viscous time units. `{dt_plus: 0.6}` in `wing.yml` gives the channel's interval in wall units instead.
- **Agent count.** The Nek5000 case uses one agent per element face midpoint (`register_wall_pts`). Here every
  distinct wall GLL node of the window is an agent: 12931 at N = 3, more at N = 7. The policy is local, so this only
  changes the actuation resolution. For face-midpoint agents, use a position file (`positions: {source: file}`).
- **Actuation limit.** OC and the policy act in [-1, 1] × u_tau(x): OC is clipped where |v'| > u_tau. UB blows
  `--ub-velocity` U_inf without zero net flux.
- **Statistics.** Cf and Cd are averaged over the span and over the samples after `--skip` actions. Use the same
  window for every controller.

## Status

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
- **Not yet run:** a resume at N = 7, and anything on AAC.
