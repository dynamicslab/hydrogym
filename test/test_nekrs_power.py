"""Actuator power of the NekRS backend mapped onto hydrogym.power (no solver, no MPI needed).

WallPower is tested on hand-made wall data; NekRSFlowEnv runs against a fake interface that answers the requests of
the udf (HELLO info, ACTDF, PRBDF/PROBE, MONDF, ACTUA, STEPS, WALLS with the wall velocity) with a simple flow.
"""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("yaml")

_STUBBED = False
try:  # hydrogym.nekrs.interface imports mpi4py at module level; the fake interface below never uses MPI
    import mpi4py  # noqa: F401
except ImportError:
    _mpi = types.ModuleType("mpi4py.MPI")
    _mpi.Comm = object
    _mpi.Intercomm = object
    _mpi.COMM_WORLD = None
    _pkg = types.ModuleType("mpi4py")
    _pkg.MPI = _mpi
    sys.modules["mpi4py"], sys.modules["mpi4py.MPI"] = _pkg, _mpi
    _STUBBED = True
try:
    from hydrogym.nekrs.env import NekRSFlowEnv  # noqa: E402
    from hydrogym.nekrs.power import PowerAverage, WallPower, reference_power  # noqa: E402
finally:  # the stub must not make other test modules believe that mpi4py is installed
    if _STUBBED:
        del sys.modules["mpi4py"], sys.modules["mpi4py.MPI"]
from hydrogym.power import forms_from_info  # noqa: E402

# ---------------------------------------------------------------------------------------------------- WallPower


def _wall(nx=4, nz=5, span=0.1):
    """Flat wall y = 0 (normal +y into the fluid), nx stations x nz spanwise nodes, equal weights."""
    x, z = np.meshgrid(np.linspace(0.3, 0.6, nx), np.linspace(0.0, span, nz), indexing="ij")
    xyz = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=1)
    w = np.full(len(xyz), 0.3 * span / len(xyz))
    normal = np.tile([0.0, 1.0, 0.0], (len(xyz), 1))
    return xyz, w, normal


def test_single_blowing_node_all_forms_agree():
    xyz, w, nrm = _wall(1, 1)
    wp = WallPower(xyz, w, nrm, xyz, reference_power=1.0, pressure_mean=None)
    v, p = 0.05, 0.2
    r = wp.evaluate([p], [[0.0, v, 0.0]])
    expect = w[0] * (p * v + 0.5 * v**3)
    assert r["signed"] == pytest.approx(expect)
    assert r["positive"] == pytest.approx(expect)
    assert r["abs"] == pytest.approx(expect)
    assert r["per_actuator"] == pytest.approx([expect])


def test_forms_ordering_per_actuator_sum_and_normalisation():
    rng = np.random.default_rng(0)
    xyz, w, nrm = _wall()
    p = rng.normal(size=len(xyz))
    u = np.zeros((len(xyz), 3))
    u[:, 1] = rng.normal(scale=0.05, size=len(xyz))
    ref = reference_power(1.0, {"velocity": 1.0, "length": 1.0, "span": 0.1}, boundary_area=123.0)
    assert ref == pytest.approx(0.05)
    r = WallPower(xyz, w, nrm, xyz, ref).evaluate(p, u)
    assert r["signed"] <= r["positive"] <= r["abs"]
    assert np.sum(r["per_actuator"]) == pytest.approx(r["signed"])
    r1 = WallPower(xyz, w, nrm, xyz, 1.0).evaluate(p, u)
    assert r1["abs"] == pytest.approx(ref * r["abs"])


def test_span_mean_pressure_drops_out_of_signed_power_with_zero_net_flux_per_line():
    rng = np.random.default_rng(1)
    xyz, w, nrm = _wall(nx=3, nz=6)
    u = np.zeros((len(xyz), 3))
    v = rng.normal(size=(3, 6))
    v -= v.mean(axis=1, keepdims=True)  # zero net flux on every spanwise line (equal weights)
    u[:, 1] = 0.01 * v.ravel()
    p = rng.normal(size=len(xyz)) + 5.0  # large mean pressure
    wp = WallPower(xyz, w, nrm, xyz, 1.0)
    r = wp.evaluate(p, u)
    assert r["signed"] == pytest.approx(r["signed_full_pressure"], rel=1e-9, abs=1e-15)
    # with a net flux the mean pressure does not drop out, and p' is what the absolute form charges
    u[:, 1] += 0.01
    r = wp.evaluate(p, u)
    assert abs(r["signed_full_pressure"] - r["signed"]) > 1e-4
    pip, pik, _ = wp.node_terms(p, u)
    assert r["abs"] == pytest.approx(np.sum(w * (np.abs(pip) + np.abs(pik))))


def test_kinetic_term_is_half_vn_cubed_for_normal_blowing_and_positive_counts_blowing_only():
    xyz, w, nrm = _wall(1, 2)
    wp = WallPower(xyz, w, nrm, xyz, 1.0, pressure_mean=None)
    u = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])  # blowing and suction, zero pressure
    r = wp.evaluate([0.0, 0.0], u)
    assert r["signed"] == pytest.approx(0.0, abs=1e-15)
    assert r["positive"] == pytest.approx(w[0] * 0.5 * 0.1**3)
    assert r["abs"] == pytest.approx(2 * w[0] * 0.5 * 0.1**3)


def test_nodes_attributed_to_nearest_actuator():
    xyz, w, nrm = _wall(nx=2, nz=4)
    act = np.array([[0.3, 0.0, 0.05], [0.6, 0.0, 0.05]])  # one actuator per station
    wp = WallPower(xyz, w, nrm, act, 1.0, pressure_mean=None)
    u = np.zeros((len(xyz), 3))
    u[:4, 1] = 0.1  # station 0 blows
    r = wp.evaluate(np.ones(len(xyz)), u)
    assert r["per_actuator"][1] == 0.0
    assert r["per_actuator"][0] == pytest.approx(r["signed"])


def test_power_average_weights():
    a = PowerAverage()
    a.add({"signed": 1.0, "per_actuator": np.array([1.0, 0.0])}, weight=1.0)
    a.add({"signed": 4.0, "per_actuator": np.array([0.0, 3.0])}, weight=3.0)
    m = a.mean()
    assert m["signed"] == pytest.approx(13.0 / 4.0)
    assert m["per_actuator"] == pytest.approx([0.25, 2.25])


# ---------------------------------------------------------------------------------------------------- env


class FakeIface:
    """Answers the env's requests; actuator i drives wall node i (point actuators on a flat wall y = 0)."""

    def __init__(self, nx=3, nz=4, dt=2e-5):
        self.xyz, self.w, self.normal = _wall(nx, nz)
        self.info = {"mu": 1e-5, "rho": 1.0, "dt": dt, "tstep": 10, "time": 1.0}
        self.time, self.tstep, self.dt = 1.0, 10, dt
        self.vel = np.zeros(len(self.xyz))
        self.profile = None
        self.start_calls = []
        self._pending = 0
        self.n_mon = 0

    # setup
    def define_actuators(self, boundaries, centres, **kw):
        n = len(centres)
        return {
            "nodes": np.ones(n, int),
            "under_resolved": np.zeros(n, bool),
            "flux_per_unit": self.w[:n].copy(),
            "nodes_per_width": -np.ones(n),
            "area": float(self.w.sum()),
        }

    def define_probes(self, xyz, fields, mean_axis=None):
        self.n_probes, self.nf = len(xyz), len(fields)
        return {"n_border": 0, "max_distance": 0.0}

    def define_monitors(self, monitors):
        self.n_mon = len(monitors)
        return np.full(len(monitors), float(self.w.sum()))

    # stepping: the pressure changes along the span and in time, so every solver step gives a different power
    def _p(self):
        return np.sin(self.xyz[:, 0] * 7.0 + 1e4 * self.time + 40.0 * self.xyz[:, 2])

    def setControlProperties(self, vel):
        self.vel = np.asarray(vel, float).copy()
        return {
            "net_flux": 0.0,
            "max_wall_velocity": float(np.abs(vel).max()),
            "znmf_correction": 0.0,
            "max_command": 0.0,
        }

    def start_time_steps(self, n):
        self.start_calls.append(int(n))
        self._pending = int(n)

    def wait_time_steps(self):
        rows = []
        for _ in range(self._pending):
            self.tstep += 1
            self.time += self.dt
            mon = []
            for m in range(self.n_mon):  # force monitors: Fv = (2 + m, 0, 0), Fp = (1 + sum(vel), 0, 0)
                mon += [2.0 + m, 0.0, 0.0, 1.0 + self.vel.sum(), 0.0, 0.0]
            rows.append([self.tstep, self.time, self.dt, 0.3] + mon)
        self._pending = 0
        return np.asarray(rows, float)

    def get_probes(self):
        return self.time, 0.3, np.zeros((self.n_probes, self.nf))

    def wall_data(self, bid, geometry=True, velocity=False, traction=True):
        out = {"p": self._p()}
        if geometry:
            out.update(xyz=self.xyz.copy(), w=self.w.copy(), normal=self.normal.copy())
        if traction:
            out["traction"] = np.zeros((len(self.xyz), 3))
        if velocity:
            u = np.zeros((len(self.xyz), 3))
            u[:, 1] = self.vel
            out["u_wall"] = u
        return out

    def finishRun(self):
        pass


def _cfg(**power):
    fake = FakeIface()
    cfg = {
        "environment": {"steps_per_action": 5},
        "actuators": {
            "boundaries": [2],
            "positions": {"source": "list", "points": fake.xyz.tolist()},
            "amplitude": 0.05,
            "znmf": "none",
        },
        "probes": {"fields": ["u", "v"], "positions": {"source": "actuator_offset", "offset": [0, 0.01, 0]}},
        "reward": {"type": "wall_shear", "boundary": 2, "quantity": "force", "baseline_dudy": 1.0},
        "monitors": [{"boundary": 1, "quantity": "force"}],
        "power": {
            "reference": {"velocity": 1.0, "length": 1.0, "span": 0.1},
            "drag": {"boundaries": [1, 2], "axis": "x"},
            **power,
        },
    }
    return cfg, fake


def _env(tmp_path, name, power=None, **kw):
    cfg, fake = _cfg(**(power or {}))
    return NekRSFlowEnv(cfg, run_root=str(tmp_path / name), iface=fake, **kw), fake


def _actions(n, k):
    return np.random.default_rng(k).uniform(-1, 1, n)


def test_env_info_keys_and_reset(tmp_path):
    env, _ = _env(tmp_path, "a")
    obs, info = env.reset()
    assert info["actuation_power"] == 0.0 and info["power_sampling"] == "end"
    obs, r, term, trunc, info = env.step(_actions(env.n_actuators, 0))
    for key in (
        "actuation_power",
        "actuation_power_positive",
        "actuation_power_abs",
        "actuation_power_per_actuator",
        "drag_coefficient",
        "power_sampling",
        "objective",
    ):
        assert key in info
    assert info["power_sampling"] == "mean"
    assert len(info["actuation_power_per_actuator"]) == env.n_actuators
    assert info["actuation_power"] <= info["actuation_power_positive"] <= info["actuation_power_abs"]
    assert np.sum(info["actuation_power_per_actuator"]) == pytest.approx(info["actuation_power"])
    assert "dUdy" in info["objective"]
    env.close()


def test_env_reward_unchanged_at_zero_weight_and_charged_at_weight(tmp_path):
    rewards = {}
    infos = {}
    for name, power in (("off", {"enabled": False}), ("w0", {}), ("w", {"power_weight": 0.5, "power_cost": "abs"})):
        env, _ = _env(tmp_path, name, power)
        env.reset()
        rewards[name], infos[name] = [], []
        for k in range(3):
            _, r, _, _, info = env.step(_actions(env.n_actuators, k))
            rewards[name].append(r)
            infos[name].append(info)
        env.close()
    assert "actuation_power" not in infos["off"][0]
    assert rewards["w0"] == rewards["off"]  # bit-identical
    for r0, rw, info in zip(rewards["off"], rewards["w"], infos["w"]):
        assert rw == pytest.approx(r0 - 0.5 * forms_from_info(info)["abs"])
        assert info["actuation_power_abs"] > 0


def test_env_mean_sampling_steps_one_by_one_and_averages(tmp_path):
    env_m, fm = _env(tmp_path, "m")
    env_e, fe = _env(tmp_path, "e", power_sampling="end")
    env_m.reset()
    env_e.reset()
    a = _actions(env_m.n_actuators, 3)
    _, rm, _, _, im = env_m.step(a)
    _, re, _, _, ie = env_e.step(a)
    assert fm.start_calls == [1] * 5 and fe.start_calls == [5]
    assert fm.time == fe.time and fm.tstep == fe.tstep
    assert rm == re  # same flow, same reward without a power weight
    assert ie["power_sampling"] == "end"
    # end = the instantaneous power after the last step; mean = average over the 5 steps
    wp = env_e._power
    u = np.zeros((len(fe.xyz), 3))
    u[:, 1] = fe.vel
    assert ie["actuation_power"] == pytest.approx(wp.evaluate(fe._p(), u)["signed"])
    assert im["actuation_power"] != pytest.approx(ie["actuation_power"])
    # drag: monitors b2 (reward, Fv 2) and b1 (Fv 3), pressure 1 + sum(vel) each, over 0.5 U^2 c span = 0.05
    cd = (2.0 + 3.0 + 2 * (1.0 + fe.vel.sum())) / 0.05
    assert ie["drag_coefficient"] == pytest.approx(cd)
    assert im["drag_coefficient"] == pytest.approx(cd)


def test_env_power_options_validated(tmp_path):
    with pytest.raises(ValueError):
        _env(tmp_path, "bad", {"power_cost": "squared"})
    with pytest.raises(ValueError):
        _env(tmp_path, "bad2", {"enabled": False, "power_weight": 1.0})
    with pytest.raises(ValueError):
        _env(tmp_path, "bad3", {"drag": {"boundaries": [7]}})


def test_power_step_progress_lockstep(tmp_path):
    envs = [_env(tmp_path, f"s{k}")[0] for k in range(2)]
    for e in envs:
        e.reset()
        e.step_async(_actions(e.n_actuators, 5))
    pending, rounds = list(envs), 0
    while pending:  # as multisim: every instance one solver step per round
        pending = [e for e in pending if not e.power_step_progress()]
        rounds += 1
    assert rounds == 5
    infos = [e.step_wait()[4] for e in envs]
    assert infos[0]["actuation_power"] == pytest.approx(infos[1]["actuation_power"])
    for e in envs:
        assert e.iface.start_calls == [1] * 5


# ------------------------------------------------------------------------------------------------------ step hook


def _hooked(tmp_path, name, every, power=None, **kw):
    env, fake = _env(tmp_path, name, power, **kw)
    calls = []
    env.set_step_hook(every, lambda e, row: calls.append((int(row[0]), float(row[1]), e.iface.tstep)))
    return env, fake, calls


@pytest.mark.parametrize("power", [{"enabled": False}, {"power_sampling": "end"}, {}])
def test_step_hook_at_global_step_multiples_same_flow(tmp_path, power):
    """Hook after every 3rd solver step (global index), one step per request; flow, reward and power unchanged."""
    env, fake, calls = _hooked(tmp_path, "h", 3, power)
    ref, fref = _env(tmp_path, "r", power)
    env.reset()
    ref.reset()
    out, out_ref = [], []
    for k in range(3):  # tstep 10 -> 25 in 3 actions of 5 steps
        a = _actions(env.n_actuators, k)
        out.append(env.step(a))
        out_ref.append(ref.step(a))
    assert [c[0] for c in calls] == [12, 15, 18, 21, 24]
    assert all(c[0] == c[2] for c in calls)  # the solver is at that step during the call
    assert [c[1] for c in calls] == pytest.approx([1.0 + (s - 10) * fake.dt for s in (12, 15, 18, 21, 24)])
    assert fake.start_calls == [1] * 15
    assert fake.tstep == fref.tstep and fake.time == fref.time
    for (o, r, te, tr, i), (o0, r0, te0, tr0, i0) in zip(out, out_ref):
        assert r == r0 and np.array_equal(o, o0)
        for key in ("actuation_power", "drag_coefficient"):
            if key in i0:
                assert i[key] == pytest.approx(i0[key])
        assert ("actuation_power" in i) == ("actuation_power" in i0)


def test_step_hook_removed_and_validated(tmp_path):
    env, fake, calls = _hooked(tmp_path, "h0", 2, {"enabled": False})
    env.reset()
    env.step(_actions(env.n_actuators, 0))
    n = len(calls)
    assert n == 2 and fake.start_calls == [1] * 5  # steps 11-15: 12, 14
    env.set_step_hook(0)
    env.step(_actions(env.n_actuators, 1))
    assert len(calls) == n and fake.start_calls[-1] == 5  # back to n steps per request
    with pytest.raises(ValueError):
        env.set_step_hook(-1, lambda e, row: None)


def test_step_hook_lockstep(tmp_path):
    """multisim drives power_step_progress: the hook runs there too, with or without power sampling."""
    envs = []
    for k in range(2):
        e, _, c = _hooked(tmp_path, f"l{k}", 5, {"enabled": False})
        e.reset()
        e.step_async(_actions(e.n_actuators, 5))
        envs.append((e, c))
    pending = [e for e, _ in envs]
    while pending:
        pending = [e for e in pending if not e.power_step_progress()]
    for e, c in envs:
        e.step_wait()
        assert [x[0] for x in c] == [15]
