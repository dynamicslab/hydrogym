"""Exercises the real, unmocked gym.make() path for Firedrake.

test_registration.py covers dispatch logic with every backend mocked away.
Firedrake needs no MPI/solver binary/GPU, so this runs with the rest of
test/. The MAIA and Nek gym.make() paths run via MPMD in the dev-container
smoke-test harness (.devcontainer/scripts/test_cpu_solvers.sh /
test_gpu_solvers.sh) -- see gym_make_demo.py in each backend's
getting_started directory."""

import gymnasium as gym
import pytest

pytest.importorskip("firedrake")

import hydrogym.registration  # noqa: E402, F401  (import performs the registration)

FIREDRAKE_IDS = [
    "hydrogym/Cylinder-v0",
    "hydrogym/RotaryCylinder-v0",
    "hydrogym/Cavity-v0",
    "hydrogym/Pinball-v0",
    "hydrogym/Step-v0",
]


@pytest.mark.parametrize("env_id", FIREDRAKE_IDS)
def test_gym_make_zero_args_reset_and_step(env_id):
    """Every registered Firedrake id must construct, reset, and step with
    zero extra arguments, as a unified entry point should."""
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        assert obs is not None
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_gym_make_caller_can_still_override_defaults():
    """The registration factory's defaults must not prevent a caller from
    passing their own env_config -- gym.make() accepts the same settings as the classes."""
    env = gym.make(
        "hydrogym/Cylinder-v0",
        env_config={"solver_config": {"dt": 5e-3}},
    )
    try:
        env.reset()
    finally:
        env.close()
