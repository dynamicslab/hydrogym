"""Tests for the minimal external-process solver skeleton (audit Task 4.4).

Plain-CI tier: no real MPI split, no solver binary, no GPU. The split
delegation is validated against a stubbed `mpi_split`, exactly the idiom
the production backends' unit tests use.
"""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from solver_env import StubSolverEnv

import hydrogym.core_external as core_external
from hydrogym.core_external import ExternalProcessEnvMixin


def test_imports_without_mpi4py_usage():
    """Importing the skeleton (and core_external) must not require mpi4py;
    only launch()/the split functions do, and they raise a clear error."""
    assert issubclass(StubSolverEnv, ExternalProcessEnvMixin)
    assert issubclass(StubSolverEnv, gym.Env)


def test_default_protocol_knobs():
    env = StubSolverEnv({"nproc": 3})
    assert env.CONTROLLER_RANK == 0
    assert env.INTERCOMM_TAG == 99
    assert env.MPI_SPLIT_LOG_PREFIX == "[MPI_SPLIT] "


def test_config_keys_consumed_and_unknown_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = StubSolverEnv({"nproc": 3, "nprocc": 3})  # typo'd key
        assert env.nproc == 3
        assert any("unsupported env_config key" in str(w.message) for w in caught)


def test_split_delegation_uses_mixin_knobs(monkeypatch):
    calls = {}

    def fake_split(comm, nproc=None, controller_rank=0, intercomm_tag=99, log_prefix=""):
        calls.update(nproc=nproc, controller_rank=controller_rank, intercomm_tag=intercomm_tag)
        return "SENTINEL"

    monkeypatch.setattr(core_external, "mpi_split", fake_split)
    env = StubSolverEnv({"nproc": 3})
    assert env.launch(comm_world="FAKE_COMM") == "SENTINEL"
    assert calls == {"nproc": 3, "controller_rank": 0, "intercomm_tag": 99}


def test_step_semantics_and_budget(monkeypatch):
    """step() with stubbed wire methods: terminated=False (no physics
    failure concept here), truncated at the step budget (Task 5.1 target
    semantics)."""
    env = StubSolverEnv({"max_steps": 2})
    env.reset(seed=0)
    monkeypatch.setattr(env, "_recv_observations", lambda: np.zeros(4, dtype=float))
    monkeypatch.setattr(env, "_send_command", lambda name, data: None)

    _, _, terminated, truncated, _ = env.step([0.0, 0.0])
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, _ = env.step([0.0, 0.0])
    assert terminated is False
    assert truncated is True


def test_close_without_comm_is_safe():
    env = StubSolverEnv()
    env.close()  # must not raise when launch() never happened
