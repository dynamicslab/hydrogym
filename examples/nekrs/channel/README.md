# Open channel at Re_τ = 207 with the generic NekRS interface

The HydroGym environment `TCFlarge_3D_Re207` in NekRS: blowing and suction at every wall GLL node, observations
[u'/u_tau, v'/u_tau] at y+ = 15, reward from the wall shear. Probes, actuators, reward and initial conditions are
set in the YAML files; the case files only mark the boundaries that may be actuated. The installation and the
general usage are described on the docs page "NekRS" (`docs/docs/installation/nekrs.md`).

| file | |
|---|---|
| `tcf.par`, `tcf.udf`, `tcf.usr` | the case; `tcf_rcb.par` uses RCB partitioning. The mesh `tcf.re2` and the initial field `rs6tcf0.f00004` come from `../fetch_case.py`. |
| `channel_re207.yml` | the environment: probes, actuators, reward (`tests/power.yml` adds the actuator power). |
| `channel_td3.yml` | short TD3 episodes (300 actions) with a snapshot library of 6 initial conditions. |
| `channel_paper.yml` | the settings of the HydroGym channel training: episodes of 2500 actions from the same 6 initial conditions. |
| `client.py` | test controller: `zero`, `oppo` (opposition control), `random`, `td3:<zip>`; `--record-every n` stores observations and actions for `../wing/obs_shift.py`. |
| `train_td3.py` | TD3 with one agent per actuator and a shared policy; `--sims M` runs M simulations for one policy. |
| `baselines.py` | zero actuation and opposition control from every initial condition of the library. |
| `run.sh`, `train_slurm.sbatch` | MPMD launch (controller on rank 0, NekRS on the other ranks), locally or in a HydroGym image on a Slurm node. |
| `tests/` | YAML files and scripts of the interface tests (reset, power, several simulations, an under-resolved case). |

## Run

```bash
python ../fetch_case.py                                                   # tcf.re2 and rs6tcf0.f00004
GPU=0 NP=1 RUN=runs/zero ./run.sh --policy zero --steps 20
GPU=0 NP=1 RUN=runs/oc   ./run.sh --policy oppo --steps 20
GPU=0 NP=1 RUN=runs/td3  SCRIPT=train_td3.py ./run.sh --episodes 10
GPU=0,1 NSIM=2 RUN=runs/td3_x2 SCRIPT=train_td3.py ./run.sh --episodes 10    # 2 simulations, one policy
RUN=runs/td3_x4 NSIM=4 EPISODES=100 sbatch train_slurm.sbatch               # Slurm node, HydroGym image
```
