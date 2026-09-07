"""Live (non-mocked) test that gym.make() is a genuine unified entry point.

test_registration.py covers dispatch *logic* with every backend mocked
away -- valuable, but it would not have caught any of the 4 real bugs
found in the "gym.make() as a Unified Entry Point" investigation
(HYDROGYM_ENGINEERING_AUDIT_v2.md): SemiImplicitBDF's required `dt`,
MAIA's lazy-loader bypass, MAIA's missing probe_locations handling, and
JAX-Fluids' unexported env classes -- all four were only visible by
calling the real thing. This file exercises the real, unmocked path for
the one backend that needs no MPI/solver binary/GPU (Firedrake), so it
runs in the same plain Firedrake CI tier as the rest of test/. The MAIA
and Nek gym.make() paths are exercised for real too, but via MPMD in the
dev-container smoke-test harness (.devcontainer/scripts/test_cpu_solvers.sh
/ test_gpu_solvers.sh), not here -- see gym_make_smoke.py in each
backend's getting_started directory.
"""

import gymnasium as gym
import pytest

import hydrogym.registration  # noqa: F401  (import performs the registration)

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
    zero extra arguments -- the whole point of a unified entry point."""
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
    passing their own env_config -- gym.make() is additive, not a cage."""
    env = gym.make(
        "hydrogym/Cylinder-v0",
        env_config={"solver_config": {"dt": 5e-3}},
    )
    try:
        env.reset()
    finally:
        env.close()
