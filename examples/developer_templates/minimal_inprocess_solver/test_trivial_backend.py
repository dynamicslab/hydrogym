"""Tests for the minimal in-process solver skeleton (audit Task 4.4).

Deliberately plain-CI: no Firedrake, no MPI, no GPU. These run with
`pytest examples/developer_templates/minimal_inprocess_solver/`.
"""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from trivial_backend import TrivialFlow, TrivialSolver, make


@pytest.fixture
def env():
    return make()


def test_flow_and_solver_construct_without_backend_deps():
    flow = TrivialFlow(n_dofs=4)
    solver = TrivialSolver(flow, dt=0.01)
    assert flow.num_inputs == 1
    assert flow.num_outputs == 4
    assert solver.dt == 0.01


def test_env_reset_and_step(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert np.all(obs == 0.0)

    obs, reward, terminated, truncated, info = env.step([0.5])
    assert obs.shape == (8,)
    # q grew by one Euler step; actuator pushed the first dof positive
    assert obs[0] > 0.0
    assert not terminated
    assert isinstance(truncated, (bool, np.bool_))


def test_unknown_flow_config_key_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrivialFlow(n_dofs=4, n_dofz=4)  # typo'd key
        assert any("unsupported flow_config key" in str(w.message) for w in caught)


def test_factory_is_gymnasium_registerable():
    """The `make` entry point works through gym.make (unregistered use via
    direct call shown here; registering is the backend author's choice of
    id/namespace)."""
    env = make({"max_steps": 5})
    assert isinstance(env, gym.Env)
    env.reset()
    for _ in range(3):
        env.step([0.0])
    assert not env.check_complete()
