---
sidebar_label: multisim
title: hydrogym.nekrs.multisim
---

Several simulations of the same NekRS case feeding one shared policy: one launch with M instances of N ranks,

mpirun -n 1 python train.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup case.par \
: -n N env NEKRS_MPMD_COLOR=2 nekrs-mpmd --setup case.par : ...

envs = make_envs("channel.yml", n_sims=M, nproc=N, run_root="runs/x")
vec = MultiSimVecEnv(envs)         # Stable-Baselines3 VecEnv, one sub-env per actuator of every simulation

The instances advance concurrently (step_async on all, then step_wait on all), so M simulations take about the
wall time of one per action and give M times the transitions. Each simulation keeps its own episode: it is reset
(its own initial condition draw) when it ends, independently of the others.

#### make\_envs

```python
def make_envs(config: Union[str, Dict],
              n_sims: int,
              nproc: int = 1,
              run_root: str = "runs/nekrs") -> List[NekRSFlowEnv]
```

One NekRSFlowEnv per instance, logging to run_root/sim&lt;k>. A snapshot library that is built (snapshots.save
without load) is built once by the first simulation and read from disk by the others.

#### MultiSimVecEnv

```python
def MultiSimVecEnv(envs: List[NekRSFlowEnv])
```

SB3 VecEnv over the actuators of all simulations (agent i of simulation k = sub-env offset[k] + i).

