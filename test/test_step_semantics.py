"""Regression tests for audit Task 5.1: terminated/truncated semantics.

Target contract (matching core.py's FlowEnv, which already implements it):

  - ``terminated`` = physics invalid (CFL blowup / divergence). For MAIA
    and Nek this NEVER arrives as a returned flag: Nek raises
    ``NekDivergenceError`` from _evolve (Task 6.3), and MAIA's solver
    exposes no in-band divergence signal at all.
  - ``truncated`` = the episode budget was reached (max_episode_steps /
    nb_interactions / tmax).

Both backends previously folded the budget signal into ``terminated``
(and MAIA duplicated it into both flags). These tests pin the split.

MAIA is exercised behaviorally through a __new__-constructed
MaiaFlowEnv with a stubbed solver interface; Nek through a stubbed-init
NekEnv with stubbed wire methods -- no live solver or MPMD world needed.

importorskip guards: maia pulls mpi4py/einops, nek pulls mpi4py/pandas.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")

from hydrogym.maia.env_core import MaiaFlowEnv  # noqa: E402


class FakeMaiaInterface:
    """Records the calls step() makes; returns a fixed probe field."""

    def __init__(self):
        self.run_calls = []

    def runTimeSteps(self, n):
        self.run_calls.append(n)

    def setControlProperties(self, props):
        self.props = props

    def getProbeData(self, locations):
        return np.array([1.0, 2.0])  # one probe, two vars (u, v)

    def continueRun(self):
        pass


def _make_maia_env(max_episode_steps=3):
    env = MaiaFlowEnv.__new__(MaiaFlowEnv)  # skip the solver-wiring __init__
    env.MAX_CONTROL = 1.0
    env.num_substeps_per_iteration = 1
    env.maiaInterface = FakeMaiaInterface()
    env.probe_locations = None
    env.noProbes = 1
    env.observation_type = ["u", "v"]
    env.nDim = 2
    env.bcId = []
    env.convert_action = lambda action=None: action
    env.get_reward = lambda: (0.5, {})
    env.obs_loc = np.zeros(2)
    env.obs_scale = np.ones(2)
    env.iter = 0
    env.max_episode_steps = max_episode_steps
    return env


class TestMaiaStepSemantics:
    def test_mid_episode_neither_flag(self):
        env = _make_maia_env()
        _, _, terminated, truncated, _ = env.step(np.array([0.1]))
        assert terminated is False
        assert truncated is False

    def test_budget_reached_is_truncation_not_termination(self):
        # NOTE: check_complete() is strict (iter > max_episode_steps), so the
        # flag flips on the step AFTER the budget-th step. Pinned as-is; the
        # Task 5.1 change is the flag split, not the episode length.
        env = _make_maia_env(max_episode_steps=3)
        env.iter = 3  # this step pushes iter to 4 > 3
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert env.iter == 4
        assert terminated is False
        assert truncated is True

    def test_budget_stays_truncated_after_budget(self):
        env = _make_maia_env(max_episode_steps=3)
        env.iter = 10
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert terminated is False
        assert truncated is True

    def test_check_complete_is_budget_only(self):
        env = _make_maia_env()
        env.iter = env.max_episode_steps + 1
        assert env.check_complete() is True
        env.iter = env.max_episode_steps
        assert env.check_complete() is False


# ---------------------------------------------------------------------------
# Nek
# ---------------------------------------------------------------------------

from hydrogym.nek.env import NekEnv  # noqa: E402


def _make_nek_env(monkeypatch, flow_time=5.0, nb_interactions=3, tmax=10.0):
    from unittest.mock import patch

    with (
        patch.object(NekEnv, "_init_from_hf", lambda self, *a, **k: None),
        patch.object(NekEnv, "_init_from_legacy", lambda self, *a, **k: None),
    ):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1})

    env.n_actuators = 1
    env.rescale_actions = False
    env.reward_agg = "mean"
    env.tmax = tmax
    env.act_index = 0
    env.nb_interactions = nb_interactions
    monkeypatch.setattr(env, "_send_action", lambda action: None)
    monkeypatch.setattr(env, "_evolve", lambda: np.array([0.5]))
    monkeypatch.setattr(env, "_get_state", lambda: (flow_time, np.zeros(1)))
    return env


class TestNekStepSemantics:
    def test_mid_episode_neither_flag(self, monkeypatch):
        env = _make_nek_env(monkeypatch)
        _, _, terminated, truncated, _ = env.step(np.array([0.1]))
        assert terminated is False
        assert truncated is False

    def test_tmax_reached_is_truncation_not_termination(self, monkeypatch):
        env = _make_nek_env(monkeypatch, flow_time=11.0, tmax=10.0)
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert terminated is False
        assert truncated is True

    def test_step_budget_is_truncation_not_termination(self, monkeypatch):
        env = _make_nek_env(monkeypatch, nb_interactions=3)
        env.act_index = 2  # this step pushes act_index to 3
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert terminated is False
        assert truncated is True

    def test_both_budgets_hit_still_truncated_only(self, monkeypatch):
        env = _make_nek_env(monkeypatch, flow_time=99.0, nb_interactions=1)
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert terminated is False
        assert truncated is True
