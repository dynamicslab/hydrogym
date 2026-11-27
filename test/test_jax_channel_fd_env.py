"""Unit tests for hydrogym.jax.envs.channel_fd (finite-difference channel environment and the solver factory).

Hermetic: small grids, laminar start (generated at reset) or synthetic field files; no Hugging Face Hub access.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hydrogym.jax.envs.channel_fd import ChannelFlowEnv, ChannelFlowFDEnv  # noqa: E402

CFG = dict(
    Nx=16,
    Ny=16,
    Nz=16,
    ctanh=0.3,
    initial_condition="laminar",
    dtype="float64",
    nsteps=4,
    nx_jets=3,
    ny_jets=2,
    jet_radius=0.6,
    max_steps_in_episode=8,
    perturbation=0.1,
)


@pytest.fixture(scope="module")
def env():
    return ChannelFlowFDEnv(dict(CFG))


def test_spaces_and_defaults(env):
    p = env.default_params
    assert p.action_dim == 6 and p.nsteps == 4
    assert env.action_space(p).shape == (6,)
    assert env.observation_space(p).shape == (6,)
    full = ChannelFlowFDEnv({"initial_condition": "laminar"})
    assert full.Nx == full.Ny == full.Nz == 72
    assert full.n_jets == 24 and full.nsteps == 100
    assert full.solver.gx == pytest.approx(0.116964, abs=1e-6)  # 8 nu^2 Re_tau^2 / H^3
    # jet amplitude: 0.0624 friction velocities (nominal u_tau = sqrt(g_x H / 2))
    assert full.jet_amplitude == 0.0624 and full.jet_gain == pytest.approx(0.0624 * np.sqrt(0.116964), rel=1e-6)
    assert ChannelFlowFDEnv({"initial_condition": "laminar", "jet_gain": 0.3}).jet_gain == 0.3
    assert (
        full.cfg.cfl == 0.9
        and full.cfg.poisson_solver == "direct"
        and full.cfg.dtype == "float32"
        and full.cfg.lx == pytest.approx(2 * np.pi)
    )


def test_jets_have_zero_net_flux_and_vanish_at_their_edges(env):
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = jnp.asarray(rng.uniform(-1, 1, env.n_jets))
        w = np.asarray(env.jet_wall_velocity(a))
        assert abs(w.sum()) < 1e-12 * np.abs(w).sum()
        assert np.abs(w).max() <= env.jet_gain * 2.0 + 1e-12
    union = np.asarray(env._jet_labels).sum(axis=0) > 0
    assert np.all(np.asarray(env.jet_wall_velocity(jnp.arange(env.n_jets) * 0.3))[~union] == 0.0)
    assert np.all(np.asarray(env.jet_wall_velocity(jnp.zeros(env.n_jets))) == 0.0)
    assert np.all(np.asarray(env.jet_wall_velocity(jnp.ones(env.n_jets))) == 0.0)  # the common mode carries no flux


def test_jet_profile_is_parabolic(env):
    prof = np.asarray(env._profiles[0])
    assert 0.9 < prof.max() <= 1.0
    s = env.solver
    xc = (np.linspace(3.0, int(env.Nx * 2 / 3) - 4, env.nx_jets)[0] + 0.5) * s.dx_cell
    yc = (np.linspace(4.0, env.Ny - 5, env.ny_jets)[0] + 0.5) * s.dy_cell
    dx = np.abs(s.x[:, None] - xc)
    dy = np.abs(s.y[None, :] - yc)
    dx, dy = np.minimum(dx, s.cfg.lx - dx), np.minimum(dy, s.cfg.ly - dy)
    np.testing.assert_allclose(prof, np.clip(1.0 - (dx**2 + dy**2) / env.jet_radius**2, 0.0, None), atol=1e-12)


def test_observation_plane_is_near_y_plus_15(env):
    full = ChannelFlowFDEnv({"initial_condition": "laminar"})
    z = full.z_obs
    u_tau = np.sqrt(full.solver.gx * 2.0 / 2.0)
    yplus = z * u_tau / full.cfg.nu
    dz_wall_units = full.solver.dz_cell[full.k_obs] * u_tau / full.cfg.nu
    assert abs(yplus - 15.0) <= 0.5 * dz_wall_units + 1e-9


def test_reset_and_step_under_jit(env):
    p = env.default_params
    key = jax.random.PRNGKey(0)
    obs, st = env.reset(key, p)
    assert obs.shape == (6,) and bool(jnp.all(jnp.isfinite(obs)))
    a = jnp.linspace(-1.0, 1.0, env.n_jets)
    obs1, st1, r, done, info = env.step(key, st, a, p)
    assert obs1.shape == (6,) and bool(jnp.isfinite(r)) and not bool(done)
    assert int(st1.time) == env.nsteps and float(st1.t) > 0.0
    for k in (
        "discount",
        "actuation_power",
        "actuation_power_positive",
        "actuation_power_abs",
        "actuation_power_per_actuator",
        "power_sampling",
        "drag_coefficient",
        "objective",
        "wall_shear_stress",
        "obs_yplus",
    ):
        assert k in info, k
    assert info["actuation_power_per_actuator"].shape == (6,)
    assert "wss" in info["objective"]
    assert float(info["wall_shear_stress"]) == pytest.approx(float(info["objective"]["wss"]))
    # the episode ends after max_steps_in_episode solver steps
    _, st2, _, done2, _ = env.step(key, st1, a, p)
    assert bool(done2) and int(st2.time) == 0  # auto-reset


def test_zero_action_is_the_uncontrolled_flow():
    """With the exact (direct) pressure solve the environment step equals the plain solver run without jets."""
    e = ChannelFlowFDEnv({**CFG, "poisson_solver": "direct"})
    p = e.default_params
    key = jax.random.PRNGKey(1)
    _, st = e.reset(key, p)
    _, st1, _, _, info = e.step_env(key, st, jnp.zeros(e.n_jets), p)
    from hydrogym.jax.solvers.channel_fd import ChannelFDState

    ref, _ = e.solver.run(ChannelFDState(st.U, st.V, st.W, st.P, st.dt, st.t), e.nsteps)
    for a, b in ((st1.U, ref.u), (st1.V, ref.v), (st1.W, ref.w)):
        assert float(jnp.abs(a - b).max()) < 1e-10
    assert float(jnp.abs(info["actuation_power"])) == 0.0


def test_reward_forms():
    base = dict(CFG)
    key = jax.random.PRNGKey(2)
    e1 = ChannelFlowFDEnv({**base, "tau_ref": 0.2})
    e2 = ChannelFlowFDEnv({**base, "reward": "neg_tau"})
    p = e1.default_params
    _, st = e1.reset(key, p)
    a = jnp.linspace(-0.5, 0.5, e1.n_jets)
    _, _, r1, _, i1 = e1.step_env(key, st, a, p)
    _, _, r2, _, i2 = e2.step_env(key, st, a, p)
    assert float(r1) == pytest.approx(0.2 / float(i1["wall_shear_stress"]), rel=1e-12)
    assert float(r2) == pytest.approx(-float(i2["wall_shear_stress"]), rel=1e-12)
    with pytest.raises(ValueError):
        ChannelFlowFDEnv({**base, "reward": "bogus"})


def test_power_weight_charges_the_reward():
    key = jax.random.PRNGKey(3)
    e0 = ChannelFlowFDEnv(dict(CFG))
    e1 = ChannelFlowFDEnv({**CFG, "power_weight": 5.0, "power_cost": "abs"})
    p = e0.default_params
    _, st = e0.reset(key, p)
    a = jnp.linspace(-1.0, 1.0, e0.n_jets)
    _, _, r0, _, i0 = e0.step_env(key, st, a, p)
    _, _, r1, _, i1 = e1.step_env(key, st, a, p)
    assert float(r1) == pytest.approx(float(r0) - 5.0 * float(i1["actuation_power_abs"]), rel=1e-10)
    assert float(i1["actuation_power_abs"]) > 0.0


def test_vmap_over_environments(env):
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(keys, p)
    assert obs.shape == (2, 6)
    acts = jnp.stack([jnp.zeros(6), jnp.linspace(-1, 1, 6)])
    obs1, st1, r, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, st, acts, p)
    assert obs1.shape == (2, 6) and r.shape == (2,) and bool(jnp.all(jnp.isfinite(st1.U)))
    # the two environments saw different perturbations and actions
    assert float(jnp.abs(st1.U[0] - st1.U[1]).max()) > 0.0


def test_gradient_of_the_reward_with_respect_to_the_action():
    """Reverse-mode gradient through the step: finite differences (exact solve) and BiCGStab against direct solve."""
    key = jax.random.PRNGKey(5)
    grads = {}
    for solver, extra in (("direct", {}), ("bicgstab", {"pressure_tol": 1e-10, "pressure_maxit": 5000})):
        e = ChannelFlowFDEnv({**CFG, "poisson_solver": solver, "reward": "neg_tau", "nsteps": 2, **extra})
        p = e.default_params
        _, st = e.reset(key, p)
        f = lambda a: e.step_env(key, st, a, p)[2]  # noqa: E731
        a0 = jnp.linspace(-0.4, 0.6, e.n_jets)
        grads[solver] = np.asarray(jax.grad(f)(a0))
        assert np.all(np.isfinite(grads[solver])) and np.abs(grads[solver]).max() > 0.0
        if solver == "direct":
            d = jnp.asarray(np.random.default_rng(0).standard_normal(e.n_jets))
            eps = 1e-5
            fd = float((f(a0 + eps * d) - f(a0 - eps * d)) / (2 * eps))
            assert float(grads[solver] @ np.asarray(d)) == pytest.approx(fd, rel=1e-5)
    assert np.linalg.norm(grads["bicgstab"] - grads["direct"]) <= 2e-3 * np.linalg.norm(grads["direct"])


def test_float32_environment_runs():
    e = ChannelFlowFDEnv({**CFG, "dtype": "float32"})
    key = jax.random.PRNGKey(6)
    obs, st = e.reset(key, e.default_params)
    assert st.U.dtype == jnp.float32
    obs, st, r, done, info = e.step(key, st, jnp.zeros(e.n_jets, jnp.float32), e.default_params)
    assert obs.dtype == jnp.float32 and bool(jnp.isfinite(r))


def test_direct_solver_option_matches_bicgstab():
    key = jax.random.PRNGKey(7)
    ea = ChannelFlowFDEnv({**CFG, "poisson_solver": "bicgstab", "pressure_tol": 1e-11, "pressure_maxit": 5000})
    eb = ChannelFlowFDEnv({**CFG, "poisson_solver": "direct"})
    p = ea.default_params
    _, st = ea.reset(key, p)
    a = jnp.linspace(-1, 1, ea.n_jets)
    _, sa, ra, _, _ = ea.step_env(key, st, a, p)
    _, sb, rb, _, _ = eb.step_env(key, st, a, p)
    assert float(jnp.abs(sa.U - sb.U).max()) < 1e-7
    assert float(ra) == pytest.approx(float(rb), rel=1e-7)


def test_turbulent_start_from_field_files(tmp_path):
    d = tmp_path / "initial_field"
    d.mkdir()
    rng = np.random.default_rng(0)
    for n in ("U", "V", "W"):
        np.save(d / f"{n}.npy", 0.01 * rng.standard_normal((16, 16, 16)))
    e = ChannelFlowFDEnv({**CFG, "initial_condition": "turbulent", "initial_field_dir": str(d)})
    obs, st = e.reset(jax.random.PRNGKey(0), e.default_params)
    np.testing.assert_allclose(np.asarray(st.U), np.load(d / "U.npy"))
    bad = tmp_path / "bad"
    bad.mkdir()
    for n in ("U", "V", "W"):
        np.save(bad / f"{n}.npy", np.zeros((8, 8, 8)))
    with pytest.raises(ValueError):
        ChannelFlowFDEnv({**CFG, "initial_condition": "turbulent", "initial_field_dir": str(bad)})


def test_factory_dispatches_on_solver(tmp_path):
    assert isinstance(ChannelFlowEnv({**CFG}), ChannelFlowFDEnv)
    assert isinstance(ChannelFlowEnv({**CFG, "solver": "fd"}), ChannelFlowFDEnv)
    d = tmp_path / "initial_field"
    d.mkdir()
    for n in ("U", "V", "W"):
        np.save(d / f"{n}.npy", np.zeros((8, 8, 8), dtype=np.float32))
    from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

    legacy = ChannelFlowEnv({"solver": "spectral", "initial_field_dir": str(d)})
    assert isinstance(legacy, ChannelFlowSpectralEnv)
    with pytest.raises(ValueError):
        ChannelFlowEnv({"solver": "lbm"})


def test_importing_the_fd_modules_does_not_change_the_precision_setting():
    import subprocess
    import sys

    code = "import jax; from hydrogym.jax.envs.channel_fd import ChannelFlowFDEnv; print(jax.config.jax_enable_x64)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr[-500:]


def test_bicgstab_is_rejected_in_float32():
    with pytest.raises(ValueError, match="float64"):
        ChannelFlowFDEnv({**CFG, "dtype": "float32", "poisson_solver": "bicgstab"})
