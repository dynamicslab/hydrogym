"""Shared-policy Nek VecEnv on a fake environment with the TCFlarge wall layout (no Nek5000 needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the module by path: importing the hydrogym.nek package needs MPI and the Nek stack.
_spec = importlib.util.spec_from_file_location(
    "nek_shared_policy", Path(__file__).resolve().parents[1] / "hydrogym" / "nek" / "shared_policy.py"
)
shared_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared_policy)
unique_wall_points = shared_policy.unique_wall_points
make_shared_policy_vec_env = shared_policy.make_shared_policy_vec_env


def gll_nodes(n):
    """Gauss-Lobatto-Legendre-like nodes on [0, 1] (Chebyshev-Lobatto; only the end points matter here)."""
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / (n - 1)))


def tcf_wall(nex=10, nez=9, lx1=8, lx=2 * np.pi, lz=np.pi):
    """Wall GLL nodes of nex x nez elements, each element listing its own lx1 x lx1 face nodes."""
    g = gll_nodes(lx1)
    xs, zs = [], []
    for ex in range(nex):
        for ez in range(nez):
            X, Z = np.meshgrid((ex + g) * lx / nex, (ez + g) * lz / nez, indexing="ij")
            xs.append(X.ravel())
            zs.append(Z.ravel())
    x, z = np.concatenate(xs), np.concatenate(zs)
    return x, np.zeros_like(x), z


def test_tcflarge_dedup_count():
    x, y, z = tcf_wall()
    assert x.size == 5760
    rep, inv = unique_wall_points(x, y, z)
    assert rep.size == 70 * 63 == 4410
    # every actuator maps to a representative at the same (periodically wrapped) point
    np.testing.assert_allclose(np.mod(x[rep[inv]] - x + 1e-9, 2 * np.pi), 1e-9, atol=1e-6)
    # representatives are the first actuator of each point, in actuator order
    assert np.all(np.diff(rep) > 0)
    assert np.all(rep[inv] <= np.arange(x.size))


def test_no_periodic_wrap():
    x, y, z = tcf_wall()
    rep, _ = unique_wall_points(x, y, z, periodic=())
    assert rep.size == 71 * 64


class FakeNekEnv:
    """Minimal stand-in for NekEnv: obs = (x, z) per actuator, reward = action actually applied there."""

    def __init__(self, episode_length=3):
        import gymnasium as gym

        x, y, z = tcf_wall(nex=2, nez=2, lx1=4)
        self.actuator_info = {"x": x, "y": y, "z": z}
        self.n_actuators = x.size
        self.obs_per_actuator = 2
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n_actuators,), dtype=np.float32)
        self.episode_length = episode_length
        self.t = 0
        self.resets = 0
        self.last_action = None

    def _obs(self):
        return np.stack([self.actuator_info["x"], self.actuator_info["z"] + self.t], 1).ravel()

    def reset(self, seed=None):
        self.t = 0
        self.resets += 1
        return self._obs(), {}

    def step(self, action):
        assert action.shape == (self.n_actuators,)
        self.last_action = action.copy()
        self.t += 1
        trunc = self.t >= self.episode_length
        return self._obs(), float(action.mean()), False, trunc, {"reward_per_actuator": action.astype(float)}

    def close(self):
        pass


def test_vec_env_broadcast_and_reset():
    pytest.importorskip("stable_baselines3")
    nek = FakeNekEnv()
    venv = make_shared_policy_vec_env(nek)
    n = venv.num_envs
    assert n == 6 * 6  # 2x2 elements of 4x4 nodes, periodic: (2*3) x (2*3)
    obs = venv.reset()
    assert obs.shape == (n, 2)
    acts = np.linspace(-1, 1, n).reshape(n, 1)
    obs, rew, dones, infos = venv.step(acts)
    # every actuator got its agent's action; the reward of agent k is the reward at its representative
    np.testing.assert_allclose(nek.last_action, acts.ravel()[venv.actuator_agent])
    np.testing.assert_allclose(rew, acts.ravel())
    assert not dones.any()
    venv.step(acts)
    obs, rew, dones, infos = venv.step(acts)
    assert dones.all() and "terminal_observation" in infos[0] and infos[0]["TimeLimit.truncated"]
    assert nek.resets == 2 and nek.t == 0


def test_td3_learns_a_few_steps():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import TD3

    venv = make_shared_policy_vec_env(FakeNekEnv(episode_length=5))
    model = TD3(
        "MlpPolicy",
        venv,
        policy_kwargs={"net_arch": {"pi": [8], "qf": [16, 16, 64]}},
        buffer_size=10_000,
        batch_size=64,
        learning_starts=2,
        train_freq=(2, "step"),
        gradient_steps=2,
        seed=1,
    )
    model.learn(total_timesteps=12 * venv.num_envs)
    assert model.replay_buffer.size() > 0
