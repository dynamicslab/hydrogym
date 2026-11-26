"""Actuation power of the Firedrake environments (hydrogym.power).

Unit checks (info keys, sign conventions, reward unchanged at weight 0, reward - w C_P at weight w) and physics
checks:

* rotating cylinder, steady Navier-Stokes (Newton): shaft power ``Omega M`` from the wall traction against an
  independent control-volume (weak-form) momentum balance over an annulus around the cylinder;
* jets: power of a prescribed jet state against the closed-form integral of the jet profile.
"""

import numpy as np
import pytest

fd = pytest.importorskip("firedrake")

import hydrogym.firedrake as hgym  # noqa: E402
from hydrogym.core import FlowEnv  # noqa: E402
from hydrogym.power import POWER_COSTS  # noqa: E402

INFO_KEYS = (
    "actuation_power",
    "actuation_power_positive",
    "actuation_power_abs",
    "actuation_power_per_actuator",
    "power_sampling",
    "objective",
)


def _env(flow, power=None, substeps=1, dt=1e-2, **flow_kw):
    cfg = {
        "flow": flow,
        "flow_config": {"use_HF_data_manager": False, **flow_kw},
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": dt},
        "actuation_config": {"num_substeps": substeps},
    }
    cfg.update(power or {})
    return FlowEnv(cfg)


def _control_volume_power(flow, omega, center, radius_cv):
    """``int_CV [(u . grad u) . w + sigma : eps(w)] dx`` with ``w = phi(r) Omega e_z x (x - x_c)``.

    ``phi = 1`` on the cylinder and decays linearly to 0 at ``r = radius_cv``. For a steady state this equals the
    power the wall puts into the fluid, ``int u_w . sigma n ds`` (momentum balance tested with ``w``).
    """
    from ufl import dot, inner, nabla_grad

    u, p = fd.split(flow.q)
    x0, y0 = center
    rx, ry = flow.x - x0, flow.y - y0
    r = fd.sqrt(rx**2 + ry**2)
    phi = fd.max_value(fd.min_value((radius_cv - r) / (radius_cv - 0.5), 1.0), 0.0)
    w_expr = fd.as_vector((-omega * ry * phi, omega * rx * phi))
    w = fd.Function(flow.velocity_space).interpolate(w_expr)
    form = inner(dot(u, nabla_grad(u)), w) * fd.dx + inner(flow.sigma(u, p), flow.epsilon(w)) * fd.dx
    return float(fd.assemble(form))


@pytest.mark.parametrize("omega", [0.5, -0.5])
def test_rotary_shaft_power_vs_control_volume(omega):
    flow = hgym.RotaryCylinder(Re=40, mesh="medium", use_HF_data_manager=False)
    flow.set_control(omega)
    hgym.NewtonSolver(flow).solve()

    (cp,) = flow.actuation_power()
    p_surface = cp * flow.power_scale
    assert p_surface == pytest.approx(omega * flow.shaft_moment(flow.CYLINDER, (0.0, 0.0)))
    assert p_surface > 0  # steady rotation always puts energy into the fluid
    for radius_cv in (1.0, 2.0):
        p_cv = _control_volume_power(flow, omega, (0.0, 0.0), radius_cv)
        print(f"Omega={omega:+.2f} R_cv={radius_cv}: P_surface={p_surface:.6f} P_cv={p_cv:.6f}")
        assert p_surface == pytest.approx(p_cv, rel=1e-2)


def _set_jet_state(flow, c, pressure):
    flow.set_control(c)
    flow.q.assign(0.0)
    flow.bcu_actuation[0].apply(flow.q)
    flow.q.subfunctions[1].interpolate(pressure)


@pytest.mark.parametrize("legacy", [False, True])
def test_jet_power_closed_form(legacy):
    """Jets with velocity = BC and uniform pressure: ``pi_p = 0`` and ``P_j = int v_n^3 / 2 dA`` per jet.

    Compared with the closed-form integral of the cosine jet profile (the boundary values are the P2
    interpolant of the profile, so a few per cent discretisation error is expected on the medium mesh).
    Default: zero-net-mass-flux pair (upper jet blows, lower jet sucks); ``legacy_jets``: both blow.
    """
    flow = hgym.Cylinder(mesh="medium", use_HF_data_manager=False, legacy_jets=legacy)
    c = 0.1
    _set_jet_state(flow, c, fd.Constant(0.3))
    omega_jet, radius = np.pi / 18, 0.5
    ke_flux = c**3 * np.pi**2 / (12 * omega_jet**2 * radius**2)  # int v_n^3 / 2 r dtheta, one jet
    cp, forms = flow.actuation_power_forms()
    sign_lo = 1.0 if legacy else -1.0
    print(f"jet KE flux C_P: {cp} closed form {2 * ke_flux:.6f}")
    np.testing.assert_allclose(cp, [2 * ke_flux, sign_lo * 2 * ke_flux], rtol=5e-2)
    q_up = flow.transpiration_flux(flow.CYLINDER, region=flow.y > 0)
    q_lo = flow.transpiration_flux(flow.CYLINDER, region=flow.y < 0)
    print(f"jet flow rates: upper {q_up:.6f} lower {q_lo:.6f} (closed form {c})")
    assert q_up == pytest.approx(c, rel=5e-2) and q_lo == pytest.approx(sign_lo * c, rel=5e-2)
    if legacy:
        # both blow and pi_p = 0 (up to the quadrature of p_ref): the three forms coincide
        assert forms["signed"] == pytest.approx(forms["positive"], rel=1e-4) == pytest.approx(forms["abs"], rel=1e-4)
    else:
        # zero net mass flux; the blowing jet's KE flux is the positive part, the sucking jet's cancels it
        assert abs(q_up + q_lo) < 1e-8 * abs(q_up)
        assert abs(forms["signed"]) < 1e-3 * forms["abs"]
        assert forms["positive"] == pytest.approx(0.5 * forms["abs"], rel=1e-3)


@pytest.mark.parametrize("name", ["Cylinder", "Cavity"])
def test_jet_power_gauge_independent(name):
    """Adding a constant to the pressure leaves all three forms (and the per-jet values) unchanged."""
    kw = {"mesh": "medium"}
    flow = getattr(hgym, name)(use_HF_data_manager=False, **kw)
    p_field = 0.4 * flow.x - 0.7 * flow.y + 0.2 * flow.x * flow.y
    _set_jet_state(flow, -0.05, p_field)
    cp0, f0 = flow.actuation_power_forms()
    _set_jet_state(flow, -0.05, p_field + 3.7)
    cp1, f1 = flow.actuation_power_forms()
    print(name, cp0, f0)
    np.testing.assert_allclose(cp1, cp0, rtol=1e-9, atol=1e-12)
    for k in f0:
        assert f1[k] == pytest.approx(f0[k], rel=1e-9, abs=1e-12)
    assert f0["signed"] <= f0["positive"] <= f0["abs"]
    assert f0["positive"] < f0["abs"]  # suction: kinetic term negative, pressure term of both signs


@pytest.mark.parametrize(
    "name,kw",
    [
        ("RotaryCylinder", {}),
        ("Cylinder", {}),
        ("Pinball", {}),
        ("Cavity", {"mesh": "medium"}),
        ("Step", {"mesh": "coarse"}),
    ],
)
def test_info_keys(name, kw):
    flow_cls = getattr(hgym, name)
    env = _env(flow_cls, substeps=2, dt=1e-3, **kw)
    _, info = env.reset()
    for key in INFO_KEYS:
        assert key in info, key
    assert info["actuation_power"] == 0.0  # zero control at reset
    _, _, _, _, info = env.step(0.5 * env.action_space.high)
    for key in INFO_KEYS:
        assert key in info, key
    assert info["power_sampling"] == "mean"
    assert len(info["actuation_power_per_actuator"]) == {"Cylinder": 2, "Pinball": 3}.get(name, 1)
    assert ("drag_coefficient" in info) == (name in ("RotaryCylinder", "Cylinder", "Pinball"))
    assert info["actuation_power"] <= info["actuation_power_positive"] <= info["actuation_power_abs"]


@pytest.mark.parametrize("substeps,sampling", [(1, None), (3, None), (3, "end")])
def test_reward_unchanged_at_zero_weight_and_charged(substeps, sampling):
    actions = [np.array([0.3]), np.array([-0.6]), np.array([1.0])]
    base = {} if sampling is None else {"power_sampling": sampling}
    envs = {
        "plain": _env(hgym.RotaryCylinder, power=None, substeps=substeps),
        "zero": _env(hgym.RotaryCylinder, power={**base, "power_weight": 0.0}, substeps=substeps),
        **{
            cost: _env(hgym.RotaryCylinder, power={**base, "power_weight": 2.5, "power_cost": cost}, substeps=substeps)
            for cost in POWER_COSTS
        },
    }
    for env in envs.values():
        env.reset()
    for a in actions:
        out = {k: env.step(a) for k, env in envs.items()}
        r_plain, info = out["plain"][1], out["plain"][4]
        assert info["power_sampling"] == "mean"
        assert out["zero"][4]["power_sampling"] == (sampling or "mean")
        assert out["zero"][1] == r_plain  # bit-identical
        if substeps == 1:
            assert r_plain == -envs["plain"].solver.dt * envs["plain"].flow.evaluate_objective()
        key = {"signed": "actuation_power", "positive": "actuation_power_positive", "abs": "actuation_power_abs"}
        for cost in POWER_COSTS:
            r_w, info_w = out[cost][1], out[cost][4]
            assert info_w["drag_coefficient"] == out["zero"][4]["drag_coefficient"]  # same trajectory and sampling
            assert r_w == pytest.approx(r_plain - 2.5 * info_w[key[cost]], rel=1e-12, abs=1e-14)


def test_mean_sampling_averages_substeps():
    """With power_sampling='mean' the reported C_P is the mean of the per-substep values."""
    env_mean = _env(hgym.Pinball, substeps=4, dt=1e-2)
    env_end = _env(hgym.Pinball, power={"power_sampling": "end"}, substeps=4, dt=1e-2)
    env_one = _env(hgym.Pinball, substeps=1, dt=1e-2)
    for env in (env_mean, env_end, env_one):
        env.reset()
    a = np.array([1.0, -2.0, 2.0])
    _, _, _, _, info_mean = env_mean.step(a)
    _, _, _, _, info_end = env_end.step(a)
    per_step = [env_one.step(a)[4]["actuation_power_per_actuator"] for _ in range(4)]
    np.testing.assert_allclose(info_mean["actuation_power_per_actuator"], np.mean(per_step, axis=0), rtol=1e-6)
    np.testing.assert_allclose(info_end["actuation_power_per_actuator"], per_step[-1], rtol=1e-6)
