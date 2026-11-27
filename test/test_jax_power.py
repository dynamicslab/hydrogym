"""Actuation power of the JAX environments (Kolmogorov forcing, channel jets), hydrogym.power.

Physics check (Kolmogorov): the kinetic-energy budget of the pseudo-spectral solver,
``dE/dt = P_forcing + P_control - eps + T_dealias``, closes with the control power computed by the environment.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("gymnax")
jax.config.update("jax_enable_x64", True)  # the channel module does this on import as well

import jax.numpy as jnp  # noqa: E402

from hydrogym.jax.utils.power import StaticStr, power_info_jax  # noqa: E402
from hydrogym.power import POWER_COSTS  # noqa: E402

INFO_KEYS = (
    "actuation_power",
    "actuation_power_positive",
    "actuation_power_abs",
    "actuation_power_per_actuator",
    "power_sampling",
    "objective",
)
FORM_KEY = {"signed": "actuation_power", "positive": "actuation_power_positive", "abs": "actuation_power_abs"}
KEY = jax.random.PRNGKey(0)


def test_static_str_survives_jit_and_vmap():
    @jax.jit
    def f(x):
        return power_info_jax(jnp.stack([x, -x]), sampling="mean")

    info = f(1.5)
    assert isinstance(info["power_sampling"], StaticStr) and info["power_sampling"] == "mean"
    assert float(info["actuation_power"]) == 0.0
    assert float(info["actuation_power_positive"]) == 1.5 and float(info["actuation_power_abs"]) == 3.0
    vinfo = jax.vmap(f)(jnp.array([1.0, -2.0]))
    np.testing.assert_allclose(vinfo["actuation_power_abs"], [2.0, 4.0])
    assert vinfo["power_sampling"] == "mean"


# ----------------------------------------------------------------------------------------------------------------
# Kolmogorov
# ----------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kolmogorov():
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    envs = {
        "mean": KolmogorovFlow({"dt": 1e-2}),
        "end": KolmogorovFlow({"dt": 1e-2, "power_sampling": "end"}),
        "zero": KolmogorovFlow({"dt": 1e-2, "power_weight": 0.0, "power_cost": "signed"}),
        **{c: KolmogorovFlow({"dt": 1e-2, "power_weight": 4.0, "power_cost": c}) for c in POWER_COSTS},
    }
    params = envs["mean"].default_params.replace(reward_alpha=1.0)  # make the reward depend on the flow
    _, state = envs["mean"].reset(KEY, params)
    return envs, params, state


def test_kolmogorov_mean_product_is_parseval(kolmogorov):
    envs, _, _ = kolmogorov
    env = envs["mean"]
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2, env.n, env.m))
    got = env._mean_product(jnp.fft.rfftn(a), jnp.fft.rfftn(b))
    assert float(got) == pytest.approx(np.mean(a * b), rel=1e-12)


@pytest.mark.parametrize("legacy", [False, True])
def test_kolmogorov_power_is_work_of_the_applied_force(legacy):
    """``<psi s_i>`` equals ``<g_i . u>`` for the body force ``g_i`` whose curl the solver adds.

    Default: ``g_i = sin(k_i y)``, the documented basis function. With ``legacy_control_scaling`` the solver's
    control term is ``k_x f_y - k_y f_x`` (no ``2 pi i``, unlike ``forcing_term``), so the applied force of mode
    ``a sin(k y)`` is ``g_hat = f_hat / (2 pi i)``, not ``f`` itself.
    """
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow
    from hydrogym.jax.utils.utils import compute_velocity_fft

    fc = {"legacy_control_scaling": True, "legacy_grid_endpoint": True} if legacy else None
    env = KolmogorovFlow({"dt": 1e-2}, flow_config=fc)
    params = env.default_params
    _, state = env.reset(KEY, params)
    w = state.omega_hat
    uh, vh = compute_velocity_fft(w, env.kx, env.ky)
    u = jnp.fft.irfftn(uh)
    psi_hat = env._stream_function_hat(w)
    sources = env._unit_mode_sources(params)
    for i, k in enumerate((params.k1, params.k2, params.k3, params.k4)):
        # drop the Nyquist column (ky = m/2): irfftn cannot represent the imaginary Nyquist part of g_hat
        fx_hat = jnp.fft.rfftn(jnp.sin(k * env.y)).at[:, -1].set(0.0)
        gx = jnp.fft.irfftn(fx_hat / (2j * jnp.pi)) if legacy else jnp.fft.irfftn(fx_hat)
        source = sources[i].at[:, -1].set(0.0)
        assert float(env._mean_product(psi_hat, source)) == pytest.approx(float(jnp.mean(gx * u)), rel=1e-9)


def test_kolmogorov_control_force_is_the_basis_function():
    """Action ``a`` on mode ``k`` adds the vorticity source ``curl(a sin(k y) e_x) = -a k cos(k y)``; the grid is
    periodic with the FFT spacing; legacy switches reproduce ``-a cos(k y) / (2 pi)`` and the end-point grid."""
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    new = KolmogorovFlow({"dt": 1e-2})
    old = KolmogorovFlow({"dt": 1e-2}, flow_config={"legacy_control_scaling": True, "legacy_grid_endpoint": True})
    n, m = new.n, new.m
    y = 2 * np.pi * np.arange(m) / m
    assert np.allclose(np.asarray(new.y[0]), y, rtol=0, atol=1e-14)
    assert float(old.y[0, -1]) == pytest.approx(2 * np.pi, rel=1e-15)  # legacy: end point included
    omega = jnp.zeros((n, m // 2 + 1), dtype=jnp.complex128)
    Y = np.broadcast_to(y, (n, m))
    for k in (4, 5, 6, 7):
        cf = (jnp.asarray(np.sin(k * Y)), jnp.zeros((n, m)))
        s_new = np.asarray(jnp.fft.irfftn(new.equation.control_term(omega, control_field=cf), s=(n, m)))
        s_old = np.asarray(jnp.fft.irfftn(old.equation.control_term(omega, control_field=cf), s=(n, m)))
        assert np.abs(s_new - (-k * np.cos(k * Y))).max() < 1e-10
        assert np.abs(s_old - (-(k / (2 * np.pi)) * np.sin(k * Y))).max() < 1e-10
    # the background forcing sin(4 y) is a single Fourier mode on the periodic grid, but leaks on the legacy grid
    a_new = np.abs(np.asarray(new.equation.forcing_term()))
    a_old = np.abs(np.asarray(old.equation.forcing_term()))
    assert int((a_new > 1e-10 * a_new.max()).sum()) == 1
    assert int((a_old > 1e-10 * a_old.max()).sum()) > 10


def test_kolmogorov_legacy_reproduces_v1(monkeypatch):
    """The three legacy switches reproduce the v1 results (HydroGym <= 1.0 reference values, CPU, x64).

    HydroGym <= 1.0 counted solver steps by float floor division (``1 // 1e-2 = 99``), whereas
    ``hydrogym.jax.solvers.base._n_steps`` gives 100; the v1 values are reproduced with the floor-division count
    patched in.
    """
    import hydrogym.jax.solvers.base as solver_base
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    monkeypatch.setattr(solver_base, "_n_steps", lambda duration, dt: int(duration // dt))

    legacy = {"legacy_control_scaling": True, "legacy_grid_endpoint": True, "legacy_dealiasing": True}
    env = KolmogorovFlow({"dt": 1e-2}, flow_config=legacy)
    params = env.default_params.replace(reward_alpha=1.0)
    _, state = env.reset(KEY, params)
    _, _, r, _, info = env.step(KEY, state, jnp.array([0.3, -0.2, 0.1, 0.4]), params)
    assert float(r) == pytest.approx(-2.3862644334740404, rel=1e-12)
    assert float(info["mean_tke"]) == pytest.approx(1.3862644334740406, rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(info["actuation_power_per_actuator"]),
        [0.0009918320346001537, 7.450960203453792e-05, -2.553205802708377e-05, 0.00035042133381570225],
        rtol=1e-9,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_kolmogorov_energy_budget(legacy):
    """dE/dt = P_forcing + P_control - nu <omega^2> + T (T: energy transfer of the dealiased advection term)."""
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    dt = 1e-3
    fc = {"legacy_control_scaling": True, "legacy_grid_endpoint": True} if legacy else None
    env = KolmogorovFlow({"dt": dt}, flow_config=fc)
    params = env.default_params
    _, state = env.reset(KEY, params)
    action = jnp.array([0.5, -0.4, 0.3, 0.6])
    cf = env._control_field(action, params)
    g = env._power_fn(action, params)
    eq = env.equation
    step = jax.jit(env.integrator.RK4_CN(control_field=cf))
    nu = env.flow.nu

    @jax.jit
    def diagnostics(w):
        psi = env._stream_function_hat(w)
        d = g(w)
        adv = eq.nonlinear_terms(w, cf) - eq.forcing_term() - eq.control_term(w, control_field=cf)
        return jnp.array(
            [
                0.5 * env._mean_product(psi, w),  # E
                jnp.sum(d["per_actuator"]),  # P_control
                d["forcing_power"],  # P_forcing
                nu * env._mean_product(w, w),  # eps
                env._mean_product(psi, adv),  # T
            ]
        )

    w = state.omega_hat
    assert w.dtype == jnp.complex128
    rows = [np.asarray(diagnostics(w))]
    for _ in range(200):
        w = step(w)
        rows.append(np.asarray(diagnostics(w)))
    E, pc, pf, eps, T = np.array(rows).T
    dEdt = (E[2:] - E[:-2]) / (2 * dt)
    residual = dEdt - (pf + pc - eps + T)[1:-1]
    print(
        f"P_control {pc.mean():.4e}  P_forcing {pf.mean():.4e}  eps {eps.mean():.4e}  T {np.abs(T).max():.2e}  "
        f"max|residual| {np.abs(residual).max():.2e}"
    )
    assert np.abs(pc).min() > 1e-3
    assert np.abs(residual).max() < 1e-3 * np.abs(pc).max()


def test_kolmogorov_reward_unchanged_and_charged(kolmogorov):
    envs, params, state = kolmogorov
    action = jnp.array([0.3, -0.2, 0.1, 0.4])
    out = {k: env.step(KEY, state, action, params) for k, env in envs.items()}
    obs_m, st_m, r_m, _, info_m = out["mean"]
    obs_e, st_e, r_e, _, info_e = out["end"]
    # the power diagnostics do not touch the rollout: same state, obs and reward for every sampling / weight 0
    assert bool(jnp.array_equal(st_m.omega_hat, st_e.omega_hat)) and bool(jnp.array_equal(obs_m, obs_e))
    assert float(r_m) == float(r_e) == float(out["zero"][2])
    assert float(r_m) == float(-(params.reward_alpha * info_m["mean_tke"] + jnp.sum(jnp.abs(action))))
    for info, sampling in ((info_m, "mean"), (info_e, "end")):
        for key in INFO_KEYS:
            assert key in info
        assert info["power_sampling"] == sampling
        assert info["actuation_power_per_actuator"].shape == (4,)
        assert (
            float(info["actuation_power"])
            <= float(info["actuation_power_positive"])
            <= float(info["actuation_power_abs"])
        )
    for cost in POWER_COSTS:
        r_w = float(out[cost][2])
        assert r_w == pytest.approx(float(r_m) - 4.0 * float(info_m[FORM_KEY[cost]]), rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------
# Channel
# ----------------------------------------------------------------------------------------------------------------


def test_clenshaw_curtis_weights():
    from hydrogym.jax.envs.channel import cheb_D_matrices, clenshaw_curtis_weights

    for n in (17, 18, 72):
        z = np.asarray(cheb_D_matrices(n, 2.0)[0])
        w = clenshaw_curtis_weights(n, 2.0)
        assert w.sum() == pytest.approx(2.0, rel=1e-12)
        assert w @ z**2 == pytest.approx(2.0 / 3.0, rel=1e-12)
        assert w @ z**4 == pytest.approx(2.0 / 5.0, rel=1e-12)


def _channel_field_dir(tmp_path_factory, n=16, nz=17):
    from hydrogym.jax.envs.channel import cheb_D_matrices

    z = np.asarray(cheb_D_matrices(nz, 2.0)[0])
    rng = np.random.default_rng(0)
    damp = 1 - z**2
    fields = {
        "U": 12.0 * damp + 0.5 * rng.standard_normal((n, n, nz)) * damp,
        "V": 0.5 * rng.standard_normal((n, n, nz)) * damp,
        "W": 0.5 * rng.standard_normal((n, n, nz)) * damp,
    }
    field_dir = tmp_path_factory.mktemp("channel_init")
    for name, arr in fields.items():
        np.save(field_dir / f"{name}.npy", arr.astype(np.float32))
    return {"initial_field_dir": str(field_dir), "Nx": n, "Ny": n, "Nz": nz}


@pytest.fixture(scope="module")
def channel(tmp_path_factory):
    from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

    cfg = _channel_field_dir(tmp_path_factory)
    envs = {
        "mean": ChannelFlowSpectralEnv(cfg),
        "end": ChannelFlowSpectralEnv({**cfg, "power_sampling": "end"}),
        "legacy": ChannelFlowSpectralEnv({**cfg, "legacy_jets": True}),
        **{c: ChannelFlowSpectralEnv({**cfg, "power_weight": 2.0, "power_cost": c}) for c in POWER_COSTS},
    }
    params = envs["mean"].default_params
    _, state = envs["mean"].reset(KEY, params)
    return envs, params, state


def test_channel_power_rollout_is_the_reward_rollout(channel):
    envs, params, state = channel
    env = envs["mean"]
    action = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, 24))

    def wss_plain(a):
        return env._wss_with_aux(state, a, params)

    def wss_power(a):
        wss, aux = env._wss_with_aux_power(state, a, params)
        return wss, aux[:3]

    (w0, aux0), _ = jax.jit(jax.value_and_grad(wss_plain, has_aux=True))(action)
    (w1, aux1), _ = jax.jit(jax.value_and_grad(wss_power, has_aux=True))(action)
    assert float(w0) == float(w1)
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(aux0, aux1))


def test_channel_info_reward_and_signs(channel):
    envs, params, state = channel
    action = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, 24))
    out = {k: env.step(KEY, state, action, params) for k, env in envs.items()}
    _, st_m, r_m, _, info_m = out["mean"]
    _, st_e, r_e, _, info_e = out["end"]
    assert float(r_m) == float(r_e) == -float(info_m["objective"]["wss"])
    assert bool(jnp.array_equal(st_m.U, st_e.U))
    for info, sampling in ((info_m, "mean"), (info_e, "end")):
        for key in (*INFO_KEYS, "drag_coefficient"):
            assert key in info
        assert info["power_sampling"] == sampling
        assert info["actuation_power_per_actuator"].shape == (24,)
        assert float(jnp.sum(info["actuation_power_per_actuator"])) == pytest.approx(
            float(info["actuation_power"]), rel=1e-4, abs=1e-7
        )
        assert (
            float(info["actuation_power"])
            <= float(info["actuation_power_positive"])
            <= float(info["actuation_power_abs"])
        )
    for cost in POWER_COSTS:
        assert float(out[cost][2]) == pytest.approx(float(r_m) - 2.0 * float(info_m[FORM_KEY[cost]]), rel=1e-6)

    # zero action: the wall jets are off, no power
    _, _, _, _, info0 = envs["mean"].step(KEY, state, jnp.zeros(24), params)
    assert float(info0["actuation_power_abs"]) == 0.0
    # legacy jets: at zero action the layer constraint holds w = 0 and can only remove energy
    _, _, _, _, info0 = envs["legacy"].step(KEY, state, jnp.zeros(24), params)
    assert float(info0["actuation_power_positive"]) == 0.0
    assert float(info0["actuation_power"]) < 0.0
    assert float(info0["actuation_power_abs"]) == pytest.approx(-float(info0["actuation_power"]), rel=1e-6)


def test_channel_zero_action_is_the_uncontrolled_channel(channel):
    """Zero action gives the rollout without actuation (action=None); the legacy jets do not.

    The boundary conditions and the projection are bit-identical; the full jitted rollouts agree to round-off
    (XLA fuses the two programs differently).
    """
    from hydrogym.jax.solvers.base import VelocityState

    envs, params, state = channel
    a0 = jnp.zeros(24)
    for name, same in (("mean", True), ("legacy", False)):
        env = envs[name]
        eq = env.equation
        t = env._substep_time(state, 7) + state.dt
        f = jnp.asarray(np.random.default_rng(3).standard_normal((eq.Nx, eq.Ny, eq.Nz)), dtype=eq.dtype)
        bc0, bcn = eq.enforce_noslip(f, f, f, action=a0, t=t), eq.enforce_noslip(f, f, f, action=None, t=t)
        sh = eq.to_spectral(VelocityState(f, f, f))
        p0, pn = eq.project(sh, state.dt, action=a0, t=t), eq.project(sh, state.dt, action=None, t=t)
        bitwise = all(bool(jnp.array_equal(a, b)) for a, b in zip((*bc0, *p0), (*bcn, *pn)))
        assert bitwise is same, name

        _, st0, _, _, _ = env.step(KEY, state, a0, params)

        def free(s0):
            def body(sh, n):
                t = env._substep_time(state, n)
                return env.integrator.rk4_step(sh, dt=state.dt, action=None, t=t, fx=2.0, target_bulk_u=8.0), None

            sh, _ = jax.lax.scan(body, eq.to_spectral(VelocityState(*s0)), jnp.arange(params.nsteps))
            return eq.to_physical(sh)

        ref = jax.jit(free)((state.U, state.V, state.W))
        diff = max(float(jnp.abs(a - b).max()) for a, b in zip((st0.U, st0.V, st0.W), ref))
        if same:
            assert diff < 1e-4, diff
        else:
            assert diff > 1e-2, diff


def test_channel_gain_ramp_is_per_action(channel):
    """Wall jets: gain 0 -> 0.3 over 10 DNS steps, 0.3, back to 0 over the last 10, for every action.

    Legacy: the gain used the episode step count and is negative from the second action on (-28.5 at DNS step 1000).
    """
    envs, params, state = channel
    env = envs["mean"]
    eq, dt = env.equation, float(params.dt)
    t = np.arange(params.nsteps + 1) * dt
    g = np.asarray(jax.vmap(eq.jet_gain)(jnp.asarray(t)))
    assert g[0] == 0.0 and g[-1] == pytest.approx(0.0, abs=1e-12)
    assert g[5] == pytest.approx(0.15) and g[10] == pytest.approx(0.3) and np.all(g[10:41] == pytest.approx(0.3))
    assert g[45] == pytest.approx(0.15) and np.all(g >= 0.0) and np.all(g <= 0.3)
    # the substep times of any action are the same (they do not depend on the episode time)
    later = state.replace(time=10 * params.nsteps)
    assert bool(jnp.array_equal(env._substep_time(state, jnp.arange(50)), env._substep_time(later, jnp.arange(50))))
    leg = envs["legacy"].equation
    assert float(leg.jet_gain(1000.0)) == pytest.approx(-28.5)
    assert float(leg.jet_gain(25.0)) == pytest.approx(0.3)


def test_channel_wall_jets_are_wall_transpiration(channel):
    """Wall jets set w on the bottom wall only (zero net flux); the interior layers are free."""
    envs, params, state = channel
    eq = envs["mean"].equation
    action = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, 24))
    t = 25 * float(params.dt)
    w_in = jnp.asarray(np.random.default_rng(2).standard_normal((eq.Nx, eq.Ny, eq.Nz)))
    u, v, w = eq.enforce_noslip(w_in, w_in, w_in, action=action, t=t)
    w_jet = eq.jet_wall_velocity(action, t)
    assert bool(jnp.array_equal(w[:, :, 0], w_jet.astype(w.dtype)))
    assert float(jnp.abs(w_jet).max()) > 0.05 and abs(float(jnp.mean(w_jet))) < 1e-6
    assert bool(jnp.array_equal(w[:, :, 1:-1], w_in[:, :, 1:-1])) and float(jnp.abs(w[:, :, -1]).max()) == 0.0
    assert float(jnp.abs(u[:, :, 0]).max()) == float(jnp.abs(v[:, :, 0]).max()) == 0.0
    # the projection keeps the transpiration velocity at the wall
    sh = eq.to_spectral(_vs(w_in))
    projected = eq.to_physical(eq.project(sh, float(params.dt), action=action, t=t))
    np.testing.assert_allclose(np.asarray(projected.w[:, :, 0]), np.asarray(w_jet), atol=1e-6)


def _vs(f):
    from hydrogym.jax.solvers.base import VelocityState

    return VelocityState(f, f, f)


def _channel_wall_jet_budget_residual(tmp_path, n):
    """Max relative residual of dE/dt = P_jets - eps (fluid initially at rest, no driving force) after the ramp."""
    from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv, clenshaw_curtis_weights

    nz = n + 1
    field_dir = tmp_path / f"rest{n}"
    field_dir.mkdir()
    for k in "UVW":
        np.save(field_dir / f"{k}.npy", np.zeros((n, n, nz)))
    env = ChannelFlowSpectralEnv({"initial_field_dir": str(field_dir), "Nx": n, "Ny": n, "Nz": nz, "dtype": "float64"})
    eq, dt = env.equation, 2e-4
    eq.action_time, eq.ramp_time = 1e9, 20 * dt  # ramp up, then hold the jets
    action = 0.3 * jnp.asarray(np.random.default_rng(1).uniform(-1, 1, 24))
    wz = jnp.asarray(clenshaw_curtis_weights(nz, env.Lz))
    dA = env._wall_dA

    def energy(s):
        return 0.5 * jnp.sum((s.u**2 + s.v**2 + s.w**2) * wz) * dA

    def eps(s):
        tot = 0.0
        for f in (s.u, s.v, s.w):
            fh = eq.fft_xy(f)
            tot += jnp.sum((eq.ifft_xy(eq.dx_hat(fh)) ** 2 + eq.ifft_xy(eq.dy_hat(fh)) ** 2 + eq.dz_phys(f) ** 2) * wz)
        return env.nu * tot * dA

    @jax.jit
    def run(sh):
        def body(sh, i):
            t = i * dt
            new, p_wall = env.integrator.rk4_step(
                sh, dt=dt, action=action, t=t, fx=0.0, enforce_const_massflux=False, return_pre_bc=True
            )
            wj = eq.jet_wall_velocity(action, t + dt)
            power = jnp.sum(wj * (p_wall - jnp.mean(p_wall) + 0.5 * wj**2)) * dA
            phys = eq.to_physical(new)
            return new, jnp.stack([energy(phys), power, eps(phys)])

        return jax.lax.scan(body, sh, jnp.arange(200))[1]

    zero = jnp.zeros((n, n, nz))
    rows = np.asarray(run(eq.to_spectral(_vs(zero))))
    E = np.concatenate([[0.0], rows[:, 0]])
    P, ep = rows[:, 1], rows[:, 2]
    dEdt = np.diff(E) / dt
    epm = 0.5 * (ep + np.concatenate([[0.0], ep[:-1]]))
    res = (dEdt - (P - epm))[40:] / P[40:]
    print(f"n={n}: P {P[-1]:.3e} eps {epm[-1]:.3e} max rel. residual {np.abs(res).max():.3e}")
    assert np.all(P[40:] > 0)
    return np.abs(res).max()


def test_channel_wall_jet_power_closes_the_energy_budget(tmp_path):
    """dE/dt = P_jets - eps with P_jets = sum w_jet (p - <p> + w_jet^2/2) dA, up to a residual that converges.

    The residual is the discretisation error of the projection at the wall row (about 11 % of P at 16^3 and 4 %
    at 32^3 in the transient after the ramp, 1 % at 32^3 later).
    """
    r16 = _channel_wall_jet_budget_residual(tmp_path, 16)
    r32 = _channel_wall_jet_budget_residual(tmp_path, 32)
    assert r16 < 0.15 and r32 < 0.05 and r32 < 0.5 * r16


def test_channel_legacy_jets_reproduce_v1(channel):
    """``legacy_jets`` reproduces the v1 results (HydroGym <= 1.0 reference values, CPU)."""
    envs, params, state = channel
    env = envs["legacy"]
    action = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, 24))
    _, st, r0, _, info0 = env.step(KEY, state, action, params)
    _, _, r1, _, info1 = env.step(KEY, st, action, params)
    # float32 reward: allow cross-platform round-off (CPU architectures / XLA versions differ at ~1e-6)
    assert float(r0) == pytest.approx(-0.05654633790254593, rel=5e-6)
    assert float(r1) == pytest.approx(-0.05662553012371063, rel=5e-6)
    assert float(info0["actuation_power"]) == pytest.approx(0.0027359589003026485, rel=1e-5)
    assert float(info1["actuation_power"]) == pytest.approx(0.1763174831867218, rel=1e-5)


def test_channel_legacy_and_wall_jets_differ_in_drag(channel):
    """Wall transpiration jets and the legacy jets (HydroGym <= 1.0) give different wall shear.

    The legacy model overwrites w in five wall layers, also at zero action, so its reward at zero action is not
    the uncontrolled flow's. The corrected jets at zero action reproduce the uncontrolled flow.
    """
    envs, params, state = channel
    out = {}
    for label, action in (("zero", jnp.zeros(24)), ("const", 0.5 * jnp.ones(24))):
        for name in ("mean", "legacy"):
            _, _, r, _, _ = envs[name].step(KEY, state, action, params)
            out[label, name] = float(r)
    assert all(np.isfinite(v) for v in out.values())
    for label in ("zero", "const"):
        assert out[label, "mean"] != out[label, "legacy"], f"{label}: {out}"
