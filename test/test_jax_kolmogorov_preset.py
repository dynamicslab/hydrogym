"""Kolmogorov env: parameter overrides, dissipation reward / L2 penalty, and the extreme-event preset."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("gymnax")
jnp = jax.numpy

from hydrogym.jax.envs.kolmogorov import (  # noqa: E402
    KOLMOGOROV_EXTREME_EVENTS_RE40,
    KolmogorovFlow,
    KolmogorovFlowParams,
)


def test_default_params_unchanged():
    """Without overrides the released defaults (TKE objective, L1 penalty of weight 1) are unchanged."""
    p = KolmogorovFlow().default_params
    assert (p.min_action, p.max_action, p.action_time, p.max_episode_steps) == (-0.5, 0.5, 10.0, 1000)
    assert (p.reward_objective, p.action_penalty_weight, p.action_penalty_power) == (0, 1.0, 1)


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="unknown"):
        KolmogorovFlow(env_config={"params": {"actiontime": 4.0}})


def test_preset_params_and_flow():
    env = KolmogorovFlow.extreme_events_re40()
    p = env.default_params
    want = KOLMOGOROV_EXTREME_EVENTS_RE40["env_config"]["params"]
    for k, v in want.items():
        assert getattr(p, k) == pytest.approx(v), k
    assert env.flow.Re == 40 and env.flow.k == 4 and tuple(env.flow.grid_size) == (64, 64)
    assert env.flow.legacy_grid_endpoint is True and env.flow.legacy_dealiasing is True
    # merged override
    env2 = KolmogorovFlow.extreme_events_re40(env_config={"params": {"max_episode_steps": 7}})
    assert env2.default_params.max_episode_steps == 7 and env2.default_params.action_time == 4.0


def test_dealiasing_two_thirds_rule():
    """The advection term keeps the modes |k_x|, k_y <= n // 3 and nothing else; ``legacy_dealiasing`` keeps all."""
    from hydrogym.jax.utils.utils import dealiasing

    n = 64
    a = jnp.ones((n, n // 2 + 1), dtype=jnp.complex64)
    kept = np.asarray(dealiasing(a)).real
    kx = np.abs(np.fft.fftfreq(n, 1.0 / n))[:, None]
    ky = np.arange(n // 2 + 1)[None, :]
    np.testing.assert_array_equal(kept, ((kx <= n // 3) & (ky <= n // 3)).astype(float))

    w = jnp.fft.rfftn(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    new = KolmogorovFlow().equation
    old = KolmogorovFlow(flow_config={"legacy_dealiasing": True}).equation
    hi = np.asarray(kx > n // 3) | np.asarray(ky > n // 3)
    adv_new = np.asarray(new.nonlinear_terms(w) - new.forcing_term())
    adv_old = np.asarray(old.nonlinear_terms(w) - old.forcing_term())
    assert np.abs(adv_new[np.broadcast_to(hi, adv_new.shape)]).max() == 0.0
    assert np.abs(adv_old[np.broadcast_to(hi, adv_old.shape)]).max() > 0.0


def test_reward_forms():
    env = KolmogorovFlow()
    traj = jnp.zeros((2, env.n, env.m // 2 + 1), dtype=jnp.complex64)
    a = jnp.array([0.1, -0.2, 0.05, 0.0])
    base = KolmogorovFlowParams(action_dim=4, obs_dim=64)
    # default: -(alpha * TKE + sum |a|); TKE of a zero field is 0
    assert float(env._reward(a, traj, base)) == pytest.approx(-0.35)
    l2 = base.replace(action_penalty_power=2, action_penalty_weight=2.0)
    assert float(env._reward(a, traj, l2)) == pytest.approx(-2.0 * 0.0525)


def test_preset_step_reports_dissipation_and_reward():
    """One step of the preset: 4 snapshots, reward = -(5 diss + 2 sum a^2), clipped to [-0.4, 0.15]."""
    env = KolmogorovFlow.extreme_events_re40()
    p = env.default_params
    obs, state = env.reset_env(jax.random.PRNGKey(0), p)
    assert obs.shape == (64,) and state.trajectory.shape[0] == 4
    a = jnp.array([0.3, -0.5, 0.1, 0.0])  # clipped to (0.15, -0.4, 0.1, 0.0)
    obs, state, reward, done, info = env.step_env(jax.random.PRNGKey(1), state, a, p)
    ac = np.clip(np.asarray(a), -0.4, 0.15)
    assert float(info["control_cost"]) == pytest.approx(float(np.sum(ac**2)), rel=1e-6)
    expected = -(5.0 * float(info["mean_dissipation"]) + 2.0 * float(np.sum(ac**2)))
    assert float(reward) == pytest.approx(expected, rel=1e-5)
    assert np.isfinite(float(info["mean_dissipation"])) and float(info["mean_dissipation"]) > 0


def test_rollout_objective_gradient_is_finite():
    env = KolmogorovFlow.extreme_events_re40()
    p = env.default_params
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    g = jax.jacfwd(lambda a: env.rollout_objective(state, a, p))(jnp.zeros(4))
    assert g.shape == (4,) and bool(jnp.all(jnp.isfinite(g))) and float(jnp.linalg.norm(g)) > 0
