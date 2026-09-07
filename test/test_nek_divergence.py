"""Unit tests for audit Task 6.3: catchable divergence on Nek.

The CFL-blowup branch in NekEnv._evolve used to call a bare ``exit()``,
killing the controller process (and with it any gym wrapper / RL loop
error handling) mid-step. It now raises ``NekDivergenceError`` AFTER
shutting the solver side down cleanly (TERMN + comm free + MPI finalize),
so the failure is observable and catchable.

No Nek solver or live MPI workers are needed: _evolve is exercised
against a fake intercomm whose Recv returns a scripted CFL value.

importorskip guards: hydrogym.nek.env pulls mpi4py/pandas/gymnasium.
"""

import inspect
import re

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from hydrogym.nek import NekDivergenceError  # noqa: E402
from hydrogym.nek.env import NekEnv, tag_dict  # noqa: E402


def test_exception_is_catchable_runtime_error():
    assert issubclass(NekDivergenceError, RuntimeError)
    # Also exported at the backend namespace level.
    import hydrogym.nek as nek_pkg

    assert nek_pkg.NekDivergenceError is NekDivergenceError


def test_no_bare_exit_calls_remain():
    """Pin the fix: no bare `exit()` statements anywhere in nek/env.py."""
    import hydrogym.nek.env as env_mod

    src = inspect.getsource(env_mod)
    assert re.search(r"^\s*exit\(\)", src, re.M) is None


class FakeComm:
    """Fake intercomm: Send accepts everything; Recv scripts values by tag.

    mpi4py buffer args arrive as [ndarray, dtype] pairs, so the data lives
    at ``buf[0]``.
    """

    def __init__(self, cfl_by_call, rewards=None):
        self.cfl_by_call = list(cfl_by_call)
        self.rewards = rewards if rewards is not None else np.zeros(1)
        self.sent = []

    def Send(self, buf, dest, tag):
        self.sent.append((buf[0], dest, tag))

    def Recv(self, buf, source, tag):
        arr = buf[0]
        if tag == tag_dict["current_cfl"]["tag"]:
            arr[0] = self.cfl_by_call.pop(0)
        else:  # reward buffers
            arr[:] = self.rewards
        return arr


def _make_env(monkeypatch):
    from unittest.mock import patch

    with (
        patch.object(NekEnv, "_init_from_hf", lambda self, *a, **k: None),
        patch.object(NekEnv, "_init_from_legacy", lambda self, *a, **k: None),
    ):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1})

    env.ndrl = 3
    env.target_cfl = 2.0
    env.nNID = 1
    env.uniqID = [7]
    env.TOTCTRL = 1
    env.actuator_info = {"NID": np.array([7])}
    env.n_actuators = 1
    env.baseline_dudy = 1.0
    env.reward_log = []
    env.restart_index = 0
    env.act_index = 0

    class _FakeLogger:
        def log_rewards(self, **kwargs):
            pass

    env.reward_logger = _FakeLogger()

    # _end_simulation does real MPI teardown (Finalize) -- record instead.
    env._end_simulation_calls = []
    monkeypatch.setattr(env, "_end_simulation", lambda farewell=False: env._end_simulation_calls.append(farewell))
    return env


class TestCFLBlowupRaises:
    def test_divergence_raises_catchable_error(self, monkeypatch):
        env = _make_env(monkeypatch)
        env.sub_comm = FakeComm(cfl_by_call=[5.0])  # first CFL already >= 2.0

        with pytest.raises(NekDivergenceError, match="CFL"):
            env._evolve()

        # Solver side was shut down cleanly BEFORE the raise.
        assert env._end_simulation_calls == [True]

    def test_below_threshold_completes_evolve(self, monkeypatch):
        """Sanity: a healthy CFL trace still runs to completion and returns
        rewards (ndrl=3 CFL reads, reward recv at the last one)."""
        env = _make_env(monkeypatch)
        env.sub_comm = FakeComm(cfl_by_call=[0.5, 0.6, 0.7], rewards=np.array([0.25]))

        rewards = env._evolve()

        assert rewards.shape == (1,)
        assert np.isfinite(rewards).all()
        assert env._end_simulation_calls == []  # no shutdown on the healthy path
