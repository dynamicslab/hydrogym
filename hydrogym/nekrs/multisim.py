"""
Several simulations of the same NekRS case feeding one shared policy: one launch with M instances of N ranks,

    mpirun -n 1 python train.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup case.par \\
                                : -n N env NEKRS_MPMD_COLOR=2 nekrs-mpmd --setup case.par : ...

    envs = make_envs("channel.yml", n_sims=M, nproc=N, run_root="runs/x")
    vec = MultiSimVecEnv(envs)         # Stable-Baselines3 VecEnv, one sub-env per actuator of every simulation

The instances advance concurrently (step_async on all, then step_wait on all), so M simulations take about the
wall time of one per action and give M times the transitions. Each simulation keeps its own episode: it is reset
(its own initial condition draw) when it ends, independently of the others.
"""

import copy
import os
from typing import Dict, List, Union

import numpy as np
from mpi4py import MPI

from .env import NekRSFlowEnv, load_config
from .interface import TAG_CMD, NekRSInterface, connect


def make_envs(
    config: Union[str, Dict], n_sims: int, nproc: int = 1, run_root: str = "runs/nekrs"
) -> List[NekRSFlowEnv]:
    """
    One NekRSFlowEnv per instance, logging to run_root/sim<k>. A snapshot library that is built (snapshots.save
    without load) is built once by the first simulation and read from disk by the others.
    """
    cfg = load_config(config)
    comms = connect(n_sims, nproc)
    ifaces, envs = [], []
    try:
        for c in comms:
            ifaces.append(NekRSInterface(comm=c))
        for k, iface in enumerate(ifaces):
            c = copy.deepcopy(cfg)
            ic = c.get("initial_conditions") or {}
            sn = ic.get("snapshots") or {}
            if k and ic.get("source") == "snapshots" and sn.get("save") and not sn.get("load"):
                sn["load"] = sn.pop("save")
            envs.append(NekRSFlowEnv(c, os.path.join(run_root, f"sim{k}"), nproc, iface=iface))
            if k == 0 and not envs[0]._library_ready:
                envs[0]._build_library()
    except BaseException:
        # every instance waits for requests until it gets TERMN
        for e in envs:
            e.close()
        for iface in ifaces[len(envs) :]:
            iface.finishRun()
        for c in comms[len(ifaces) :]:
            c.Send([b"TERMN", MPI.CHAR], dest=0, tag=TAG_CMD)
        raise
    return envs


def MultiSimVecEnv(envs: List[NekRSFlowEnv]):
    """SB3 VecEnv over the actuators of all simulations (agent i of simulation k = sub-env offset[k] + i)."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv

    class _Vec(VecEnv):
        def __init__(self, envs):
            self.envs = envs
            k = envs[0].obs_per_actuator
            for e in envs:
                if e.obs_per_actuator != k or e.observation_space.shape[0] != e.n_actuators * k:
                    raise ValueError("all simulations need the same observations per actuator")
            self.obs_per_agent = k
            self.sizes = [e.n_actuators for e in envs]
            self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
            obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(k,), dtype=np.float32)
            act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            super().__init__(int(self.offsets[-1]), obs_space, act_space)
            self._actions = None
            self._obs = [None] * len(envs)
            self._ep = [None] * len(envs)
            self.finished: List[Dict] = []  # episodes ended since the caller last emptied this list

        def _split(self, k, obs):
            return obs.reshape(self.sizes[k], self.obs_per_agent)

        def _reset_sim(self, k, seed=None):
            obs, info = self.envs[k].reset(seed=seed)
            self._obs[k] = self._split(k, obs)
            self._ep[k] = {"info": info, "r": [], "dUdy": []}

        def reset(self):
            for k in range(len(self.envs)):
                self._reset_sim(k, self._seeds[self.offsets[k]] if self._seeds else None)
            self._reset_seeds()
            return np.concatenate(self._obs)

        def step_async(self, actions):
            self._actions = np.asarray(actions).reshape(self.num_envs, -1)[:, 0]

        def step_wait(self):
            for k, e in enumerate(self.envs):
                e.step_async(self._actions[self.offsets[k] : self.offsets[k + 1]])
            # power sampled at every solver step: advance all instances one step at a time, in lockstep
            pending = list(self.envs)
            while pending:
                pending = [e for e in pending if not e.power_step_progress()]
            rews, dones, infos = [], [], []
            for k, e in enumerate(self.envs):
                obs, reward, term, trunc, info = e.step_wait()
                n = self.sizes[k]
                o = self._split(k, obs)
                ep = self._ep[k]
                ep["r"].append(reward)
                ep["dUdy"].append(info.get("dUdy", np.nan))
                rews.append(info["reward_per_actuator"].astype(np.float32))
                dones.append(np.full(n, term or trunc))
                agent_info = [{"time": info["time"], "sim": k} for _ in range(n)]
                if term or trunc:
                    for i in range(n):
                        agent_info[i]["terminal_observation"] = o[i]
                        agent_info[i]["TimeLimit.truncated"] = bool(trunc and not term)
                    self.finished.append(
                        {
                            "sim": k,
                            "episode": e.episode,
                            "terminated": bool(term),
                            "index": ep["info"].get("index"),
                            "rewards": ep["r"],
                            "dUdy": ep["dUdy"],
                        }
                    )
                    self._reset_sim(k)
                else:
                    self._obs[k] = o
                infos += agent_info
            return np.concatenate(self._obs), np.concatenate(rews), np.concatenate(dones), infos

        def close(self):
            for e in self.envs:
                e.close()

        def get_attr(self, name, indices=None):
            return [getattr(self.envs[self._sim(i)], name, None) for i in self._idx(indices)]

        def set_attr(self, name, value, indices=None):
            for e in self.envs:
                setattr(e, name, value)

        def env_method(self, name, *args, indices=None, **kwargs):
            return [getattr(self.envs[self._sim(i)], name)(*args, **kwargs) for i in self._idx(indices)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._idx(indices))

        def _sim(self, i):
            return int(np.searchsorted(self.offsets, i, side="right") - 1)

        def _idx(self, indices):
            return range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else indices)

    return _Vec(envs)
