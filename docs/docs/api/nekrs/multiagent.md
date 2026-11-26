---
sidebar_label: multiagent
title: hydrogym.nekrs.multiagent
---

Multi-agent views of NekRSFlowEnv, one agent per actuator, as in hydrogym.nek:

par = make_parallel_env(env)       # dict API (hydrogym.nek.parallel_env.NekParallelEnv)
pz = make_pettingzoo_env(env)      # PettingZoo ParallelEnv (hydrogym.nek.pettingzoo_env)

NekRSFlowEnv provides what these wrappers use from NekEnv: actuator_info (NID, GLLID, FACEID, ix, iy, iz ->
agent names jet_np..._gid..._iface..._ix..._iy..._iz...), n_actuators, obs_per_actuator (observations are
probe-major, so with one probe per actuator agent i gets block i), action_space and
info["reward_per_actuator"]. The hydrogym classes are used when hydrogym is installed; otherwise an equivalent
local implementation is used, so agent names and observation blocks are the same either way.

## NekRSParallelEnv Objects

```python
class NekRSParallelEnv()
```

Dict-based multi-agent wrapper, same behaviour as hydrogym.nek.parallel_env.NekParallelEnv.

## NekRSPettingZooEnv Objects

```python
class NekRSPettingZooEnv(ParallelEnv)
```

PettingZoo ParallelEnv on top of the dict wrapper (as hydrogym.nek.pettingzoo_env.NekPettingZooEnv).

## ParallelToVecEnv Objects

```python
class ParallelToVecEnv()
```

Stable-Baselines3 VecEnv with one sub-environment per agent (shared policy), the equivalent of
supersuit.pettingzoo_env_to_vec_env_v1 used by examples/nek/getting_started/3_pettingzoo.
All agents end their episode together; the reset is done here (SB3 auto-reset convention).

#### make\_parallel\_env

```python
def make_parallel_env(env, use_hydrogym: bool = True)
```

Dict multi-agent env; hydrogym's NekParallelEnv if available (and requested), else the local equivalent.

