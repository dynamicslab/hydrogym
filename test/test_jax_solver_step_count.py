"""JAX solver step count: whole steps of dt per save interval / window, robust to float round-off."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("tree_math")

from hydrogym.jax.solvers.base import _n_steps  # noqa: E402


@pytest.mark.parametrize(
    "duration,dt,n",
    [(1, 1e-3, 1000), (4.0, 1e-3, 4000), (1, 1e-2, 100), (4.0, 0.01, 400), (0.3, 0.1, 3), (4.5, 1.0, 4), (1, 0.3, 3)],
)
def test_n_steps(duration, dt, n):
    assert _n_steps(duration, dt) == n


def test_kolmogorov_preset_window_is_4000_steps():
    """The extreme-event preset integrates 4 x 1000 steps of 1e-3 per action (4000 steps per action)."""
    pytest.importorskip("gymnax")
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    p = KolmogorovFlow.extreme_events_re40().default_params
    save = _n_steps(float(p.save_time), float(p.dt))
    total = _n_steps(float(p.action_time), float(p.dt))
    assert (save, total, total // save) == (1000, 4000, 4)
