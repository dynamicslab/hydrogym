"""`reset(seed=...)` of the m-AIA environments seeds the gymnasium RNG and the action space."""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")
pytest.importorskip("pandas")

from hydrogym.maia.env_core import MaiaFlowEnv  # noqa: E402


class _Stub(gym.Env):
    """Carries only what `_seed_reset` touches; the real environments need a running m-AIA."""

    _seed_reset = MaiaFlowEnv._seed_reset

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)


def test_same_seed_gives_same_samples():
    a, b = _Stub(), _Stub()
    a._seed_reset(7)
    b._seed_reset(7)
    assert np.array_equal(a.action_space.sample(), b.action_space.sample())
    assert a.np_random.integers(1 << 30) == b.np_random.integers(1 << 30)


def test_different_seed_gives_different_samples():
    a, b = _Stub(), _Stub()
    a._seed_reset(1)
    b._seed_reset(2)
    assert not np.array_equal(a.action_space.sample(), b.action_space.sample())


def test_no_seed_keeps_the_stream():
    env = _Stub()
    env._seed_reset(3)
    first = env.action_space.sample()
    env._seed_reset(None)  # must not restart the stream
    again = env.action_space.sample()
    assert not np.array_equal(first, again)
