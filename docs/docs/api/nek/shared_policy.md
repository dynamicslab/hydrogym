---
sidebar_label: shared_policy
title: hydrogym.nek.shared_policy
---

Shared-policy multi-agent training on Nek wall actuators, without per-agent dictionaries.

The wall actuators of a Nek environment are the wall GLL nodes of the elements that touch the actuated wall. Nodes on
element edges belong to two (at corners four) elements, and on a periodic wall the nodes at both ends of a periodic
direction are the same physical point. `unique_wall_points` maps every actuator to one representative per
physical point, so that each physical actuator enters training once: for the released `TCFlarge` channel
(10 x 9 wall elements of 8 x 8 nodes) this gives 4410 agents from 5760 actuators.

`make_shared_policy_vec_env` turns a `NekEnv` into a Stable-Baselines3 `VecEnv` with
one sub-environment per agent. All agents share one policy (the multi-agent setting of the HydroGym channel), every
agent's transition enters the replay buffer, and all array operations are O(number of agents). The dictionary-based
PettingZoo route (`hydrogym.nek.pettingzoo_env` + SuperSuit) does the same thing but scales quadratically with
the number of agents and is about 50 times slower for the channel.

#### unique\_wall\_points

```python
def unique_wall_points(x: Sequence[float],
                       y: Sequence[float],
                       z: Sequence[float],
                       periodic: Iterable[str] = ("x", "z"),
                       rel_tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]
```

Representatives of the physically distinct actuator points.

**Arguments**:

  x, y, z: Actuator coordinates (one entry per actuator, duplicates included).
- `periodic` - Coordinate directions in which the actuated wall is periodic. The period is the extent of the
  coordinates in that direction (both ends are present on a periodic GLL mesh); points at the upper end
  are mapped to the lower end.
- `rel_tol` - Tolerance relative to the largest extent for two points to coincide.
  

**Returns**:

  `(rep, inverse)`: `rep[k]` is the actuator index representing agent `k` (the first actuator at that
  point); `inverse[i]` is the agent of actuator `i`, so `rep[inverse]` gives each actuator's
  representative and `action_agents[inverse]` broadcasts agent actions to all actuators.

#### make\_shared\_policy\_vec\_env

```python
def make_shared_policy_vec_env(nek_env,
                               deduplicate: bool = True,
                               periodic: Iterable[str] = ("x", "z"))
```

Stable-Baselines3 `VecEnv` with one sub-environment per wall actuator (shared policy).

**Arguments**:

- `nek_env` - A `NekEnv` (or an object with the same `n_actuators`,
  `obs_per_actuator`, `actuator_info`, `action_space`, `reset` and `step`; `step` must return
  `info["reward_per_actuator"]`).
- `deduplicate` - Train on the physically distinct wall points only (see `unique_wall_points`).
- `periodic` - Periodic directions of the actuated wall, used for the deduplication.
  

**Returns**:

  A `VecEnv`; `num_envs` is the number of agents. Observations have shape `(num_envs, obs_per_actuator)`,
  actions `(num_envs, 1)` in the per-actuator action space of `nek_env`.

