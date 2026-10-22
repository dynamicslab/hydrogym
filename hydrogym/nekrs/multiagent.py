"""
Multi-agent views of NekRSFlowEnv, one agent per actuator, as in hydrogym.nek:

    par = make_parallel_env(env)       # dict API (hydrogym.nek.parallel_env.NekParallelEnv)
    pz = make_pettingzoo_env(env)      # PettingZoo ParallelEnv (hydrogym.nek.pettingzoo_env)

NekRSFlowEnv provides what these wrappers use from NekEnv: actuator_info (NID, GLLID, FACEID, ix, iy, iz ->
agent names jet_np..._gid..._iface..._ix..._iy..._iz...), n_actuators, obs_per_actuator (observations are
probe-major, so with one probe per actuator agent i gets block i), action_space and
info["reward_per_actuator"]. The hydrogym classes are used when hydrogym is installed; otherwise an equivalent
local implementation is used, so agent names and observation blocks are the same either way.
"""

import functools
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

try:
    from hydrogym.nek.parallel_env import NekParallelEnv as _HGParallel
except Exception:  # hydrogym (or its Nek5000 dependencies) not installed
    _HGParallel = None

try:
    from pettingzoo import ParallelEnv
except ImportError:
    ParallelEnv = object


class NekRSParallelEnv:
    """Dict-based multi-agent wrapper, same behaviour as hydrogym.nek.parallel_env.NekParallelEnv."""

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        info = env.actuator_info
        self.possible_agents = [
            self._name_agent(info["NID"][i], info["GLLID"][i], info["FACEID"][i], info["ix"][i], info["iy"][i], info["iz"][i])
            for i in range(env.n_actuators)
        ]
        if len(set(self.possible_agents)) != len(self.possible_agents):
            # free jets have no grid identifiers; make the names unique by index
            self.possible_agents = [f"jet_{i:04d}" for i in range(env.n_actuators)]
        self.agents = self.possible_agents[:]
        self.obs_per_agent = env.obs_per_actuator
        if self.obs_per_agent * env.n_actuators != env.observation_space.shape[0]:
            raise ValueError(
                f"{env.observation_space.shape[0]} observations cannot be split over {env.n_actuators} agents"
            )

    @staticmethod
    def _name_agent(nid, gllid, iface, ix, iy, iz):
        return f"jet_np{int(nid):08d}_gid{int(gllid):08d}_iface{int(iface)}_ix{int(ix):08d}_iy{int(iy):08d}_iz{int(iz):08d}"

    def observation_space(self, agent):
        return gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_per_agent,), dtype=np.float32)

    def action_space(self, agent):
        i = self.possible_agents.index(agent)
        low, high = self.env.action_space.low[i], self.env.action_space.high[i]
        return gym.spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

    def _split(self, obs):
        k = self.obs_per_agent
        return {a: obs[i * k : (i + 1) * k].astype(np.float32) for i, a in enumerate(self.agents)}

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._split(obs), info

    def step(self, actions: Dict[str, np.ndarray]):
        a = np.array([float(np.asarray(actions[g]).reshape(-1)[0]) for g in self.agents], dtype=np.float32)
        obs, reward, term, trunc, info = self.env.step(a)
        rpa = info.get("reward_per_actuator")
        rewards = {g: float(rpa[i]) if rpa is not None else reward for i, g in enumerate(self.agents)}
        infos = {g: {"time": info.get("time", 0.0)} for g in self.agents}
        return (
            self._split(obs),
            rewards,
            {g: term for g in self.agents},
            {g: trunc for g in self.agents},
            infos,
        )

    def close(self):
        self.env.close()


class NekRSPettingZooEnv(ParallelEnv):
    """PettingZoo ParallelEnv on top of the dict wrapper (as hydrogym.nek.pettingzoo_env.NekPettingZooEnv)."""

    metadata = {"render_modes": [], "name": "nekrs_v1"}

    def __init__(self, parallel_env, render_mode=None):
        if ParallelEnv is object:
            raise ImportError("PettingZoo is not installed (pip install pettingzoo)")
        self.env = parallel_env
        self.render_mode = render_mode
        self.possible_agents = list(parallel_env.possible_agents)
        self.agents = list(self.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {a: dict(info) for a in self.agents}

    def step(self, actions):
        obs, rew, term, trunc, infos = self.env.step(actions)
        if any(term.values()) or any(trunc.values()):
            self.agents = []
        return obs, rew, term, trunc, infos

    def close(self):
        self.env.close()


class ParallelToVecEnv:
    """
    Stable-Baselines3 VecEnv with one sub-environment per agent (shared policy), the equivalent of
    supersuit.pettingzoo_env_to_vec_env_v1 used by hydrogym/examples/nek/getting_started/3_pettingzoo.
    All agents end their episode together; the reset is done here (SB3 auto-reset convention).
    """

    def __new__(cls, *args, **kwargs):
        from stable_baselines3.common.vec_env import VecEnv

        class _Vec(VecEnv):
            def __init__(self, par):
                self.par = par
                a0 = par.possible_agents[0]
                super().__init__(len(par.possible_agents), par.observation_space(a0), par.action_space(a0))
                self._actions = None
                self.last_info = self.episode_info = {}

            def _stack(self, d):
                return np.stack([d[a] for a in self.par.possible_agents])

            def reset(self):
                seed = self._seeds[0] if self._seeds else None
                obs, info = self.par.reset(seed=seed)
                self._reset_seeds()
                self.last_info = info
                return self._stack(obs)

            def step_async(self, actions):
                self._actions = actions

            def step_wait(self):
                agents = self.par.possible_agents
                obs, rew, term, trunc, infos = self.par.step({a: self._actions[i] for i, a in enumerate(agents)})
                o = self._stack(obs)
                r = np.array([rew[a] for a in agents], dtype=np.float32)
                te = np.array([term[a] for a in agents])
                tr = np.array([trunc[a] for a in agents])
                done = te | tr
                info = [dict(infos[a]) for a in agents]
                if done.any():
                    self.episode_info = self.last_info  # reset info of the episode that just ended
                    for i in range(len(agents)):
                        info[i]["terminal_observation"] = o[i]
                        info[i]["TimeLimit.truncated"] = bool(tr[i] and not te[i])
                    o = self.reset()
                return o, r, done, info

            def close(self):
                self.par.close()

            def get_attr(self, name, indices=None):
                return [getattr(self.par, name, None)] * len(self._idx(indices))

            def set_attr(self, name, value, indices=None):
                setattr(self.par, name, value)

            def env_method(self, name, *args, indices=None, **kwargs):
                return [getattr(self.par, name)(*args, **kwargs)] * len(self._idx(indices))

            def env_is_wrapped(self, wrapper_class, indices=None):
                return [False] * len(self._idx(indices))

            def _idx(self, indices):
                return range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else indices)

        return _Vec(*args, **kwargs)


def make_parallel_env(env, use_hydrogym: bool = True):
    """Dict multi-agent env; hydrogym's NekParallelEnv if available (and requested), else the local equivalent."""
    if use_hydrogym and _HGParallel is not None:
        return _HGParallel(env)
    return NekRSParallelEnv(env)


def make_pettingzoo_env(env, use_hydrogym: bool = True):
    return NekRSPettingZooEnv(make_parallel_env(env, use_hydrogym))
