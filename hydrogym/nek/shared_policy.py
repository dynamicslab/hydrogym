"""Shared-policy multi-agent training on Nek wall actuators, without per-agent dictionaries.

The wall actuators of a Nek environment are the wall GLL nodes of the elements that touch the actuated wall. Nodes on
element edges belong to two (at corners four) elements, and on a periodic wall the nodes at both ends of a periodic
direction are the same physical point. ``unique_wall_points`` maps every actuator to one representative per
physical point, so that each physical actuator enters training once: for the released ``TCFlarge`` channel
(10 x 9 wall elements of 8 x 8 nodes) this gives 4410 agents from 5760 actuators.

``make_shared_policy_vec_env`` turns a ``NekEnv`` into a Stable-Baselines3 ``VecEnv`` with
one sub-environment per agent. All agents share one policy (the multi-agent setting of the HydroGym channel), every
agent's transition enters the replay buffer, and all array operations are O(number of agents). The dictionary-based
PettingZoo route (``hydrogym.nek.pettingzoo_env`` + SuperSuit) does the same thing but scales quadratically with
the number of agents and is about 50 times slower for the channel.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def unique_wall_points(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    periodic: Iterable[str] = ("x", "z"),
    rel_tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Representatives of the physically distinct actuator points.

    Args:
        x, y, z: Actuator coordinates (one entry per actuator, duplicates included).
        periodic: Coordinate directions in which the actuated wall is periodic. The period is the extent of the
            coordinates in that direction (both ends are present on a periodic GLL mesh); points at the upper end
            are mapped to the lower end.
        rel_tol: Tolerance relative to the largest extent for two points to coincide.

    Returns:
        ``(rep, inverse)``: ``rep[k]`` is the actuator index representing agent ``k`` (the first actuator at that
        point); ``inverse[i]`` is the agent of actuator ``i``, so ``rep[inverse]`` gives each actuator's
        representative and ``action_agents[inverse]`` broadcasts agent actions to all actuators.
    """
    coords = np.stack([np.asarray(c, dtype=float).ravel() for c in (x, y, z)], axis=1)
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    extent = coords.max(axis=0) - coords.min(axis=0)
    tol = rel_tol * max(float(extent.max()), 1.0)
    for name in periodic:
        d = "xyz".index(name)
        if extent[d] <= tol:
            continue
        lo, hi = coords[:, d].min(), coords[:, d].max()
        at_hi = coords[:, d] >= hi - tol
        coords[at_hi, d] = lo
    keys = np.round((coords - coords.min(axis=0)) / tol).astype(np.int64)
    _, rep, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # np.unique sorts the keys; reorder agents by their first actuator so that agent order follows actuator order
    order = np.argsort(rep, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rep[order], rank[np.asarray(inverse).ravel()]


def make_shared_policy_vec_env(nek_env, deduplicate: bool = True, periodic: Iterable[str] = ("x", "z")):
    """Stable-Baselines3 ``VecEnv`` with one sub-environment per wall actuator (shared policy).

    Args:
        nek_env: A ``NekEnv`` (or an object with the same ``n_actuators``,
            ``obs_per_actuator``, ``actuator_info``, ``action_space``, ``reset`` and ``step``; ``step`` must return
            ``info["reward_per_actuator"]``).
        deduplicate: Train on the physically distinct wall points only (see ``unique_wall_points``).
        periodic: Periodic directions of the actuated wall, used for the deduplication.

    Returns:
        A ``VecEnv``; ``num_envs`` is the number of agents. Observations have shape ``(num_envs, obs_per_actuator)``,
        actions ``(num_envs, 1)`` in the per-actuator action space of ``nek_env``.
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    n_act = int(nek_env.n_actuators)
    n_obs = int(nek_env.obs_per_actuator)
    info = nek_env.actuator_info
    if deduplicate:
        rep, inverse = unique_wall_points(info["x"], info["y"], info["z"], periodic=periodic)
    else:
        rep, inverse = np.arange(n_act), np.arange(n_act)
    n_agents = int(rep.size)

    low = float(np.min(nek_env.action_space.low))
    high = float(np.max(nek_env.action_space.high))
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
    act_space = gym.spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

    class NekSharedPolicyVecEnv(VecEnv):
        """One sub-environment per agent; all agents step together in one Nek environment."""

        def __init__(self):
            super().__init__(n_agents, obs_space, act_space)
            self.nek_env = nek_env
            self.agent_actuator = rep
            self.actuator_agent = inverse
            self._actions: Optional[np.ndarray] = None
            self._seed: Optional[int] = None

        def _agent_obs(self, flat_obs: np.ndarray) -> np.ndarray:
            return np.asarray(flat_obs, dtype=np.float32).reshape(n_act, n_obs)[rep]

        def reset(self):
            obs, _ = nek_env.reset(seed=self._seed)
            self._seed = None
            return self._agent_obs(obs)

        def step_async(self, actions):
            self._actions = np.asarray(actions, dtype=np.float32).reshape(n_agents)

        def step_wait(self):
            flat_actions = self._actions[inverse]
            obs, reward, terminated, truncated, step_info = nek_env.step(flat_actions)
            r = np.asarray(step_info["reward_per_actuator"], dtype=np.float32).reshape(n_act)[rep]
            done = bool(terminated or truncated)
            agent_obs = self._agent_obs(obs)
            dones = np.full(n_agents, done, dtype=bool)
            infos = [{} for _ in range(n_agents)]
            if done:
                for k in range(n_agents):
                    infos[k]["terminal_observation"] = agent_obs[k]
                    infos[k]["TimeLimit.truncated"] = bool(truncated and not terminated)
                infos[0]["episode_reward_mean"] = float(np.mean(r))
                agent_obs = self.reset()
            return agent_obs, r, dones, infos

        def close(self):
            nek_env.close()

        def seed(self, seed=None):
            self._seed = seed
            return [seed] * n_agents

        def get_attr(self, attr_name, indices=None):
            return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

        def set_attr(self, attr_name, value, indices=None):
            setattr(self, attr_name, value)

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            out = getattr(nek_env, method_name)(*method_args, **method_kwargs)
            return [out] * len(self._get_indices(indices))

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._get_indices(indices))

        def _get_indices(self, indices):
            if indices is None:
                return range(n_agents)
            if isinstance(indices, int):
                return [indices]
            return indices

    return NekSharedPolicyVecEnv()
