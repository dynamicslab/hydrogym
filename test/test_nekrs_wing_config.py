"""NACA0012 wing (examples/nekrs/wing) against a fake interface: the YAML deployment and the Nek5000 case agree.

wing.yml must carry the time step, control interval, action amplitude, observation scaling and trip of the Nek5000
case third_party/nek5000/cases/naca0012_200k (naca_wing.rea, naca_wing.wall, drl_*.f). No solver, no MPI needed.
"""

import os
import re

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("yaml")

from test_nekrs_power import FakeIface, NekRSFlowEnv, _cfg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WING = os.path.join(REPO, "examples", "nekrs", "wing")
CASE = os.path.join(REPO, "third_party", "nek5000", "cases", "naca0012_200k")
U_TAU_CH = 0.0624334733  # drl_action.f


def _rea_params(path):
    p = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) > 1 and re.fullmatch(r"p\d{3}", t[1]):
                p[t[1]] = float(t[0])
    return p


def _wall_params(path):
    with open(path) as f:
        lines = f.read().splitlines()
    i = next(k for k, line in enumerate(lines) if "Global parameters" in line)
    return [float(lines[i + 1 + k].split()[0]) for k in range(5)]


def _x2utau(x):
    """X2Utau of drl/drl_init.f (coefficients read from the Fortran source)."""
    src = open(os.path.join(CASE, "drl", "drl_init.f")).read()
    sub = src[src.index("subroutine X2Utau") :]
    c = [float(v) for v in re.findall(r"pl\(\s*\d+\)\s*=\s*([-0-9.eE+]+)", sub)[:10]]
    return sum(ck * x**k for k, ck in enumerate(c))


class WingIface(FakeIface):
    """Suction-side nodes of a NACA0012 in the control window, 5 spanwise nodes per station."""

    def __init__(self, dt):
        super().__init__(dt=dt)
        x = np.linspace(0.25, 0.86, 40)
        yt = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        dy = np.gradient(yt, x)
        X, Z = np.meshgrid(x, np.linspace(0, 0.1, 5), indexing="ij")
        n = np.stack([-np.repeat(dy, 5), np.ones(X.size), np.zeros(X.size)], 1)
        self.normal = n / np.linalg.norm(n, axis=1, keepdims=True)
        self.xyz = np.stack([X.ravel(), np.repeat(yt, 5), Z.ravel()], 1)
        self.w = np.full(len(self.xyz), 0.061 / len(self.xyz))
        self.vel = np.zeros(len(self.xyz))
        self.info["mu"] = 5e-6

    def boundary_nodes(self, bid, nodes="all"):
        z = np.zeros(len(self.xyz), int)
        return {
            "rank": z,
            "element": np.arange(len(self.xyz)),
            "face": z,
            "ix": z,
            "iy": z,
            "iz": z,
            "xyz": self.xyz.copy(),
            "normal": self.normal.copy(),
        }


def test_environment_dt_checked_against_the_solver(tmp_path):
    cfg, fake = _cfg()
    cfg["environment"]["dt"] = fake.dt
    NekRSFlowEnv(cfg, run_root=str(tmp_path / "ok"), iface=fake).close()
    cfg, fake = _cfg()
    cfg["environment"]["dt"] = 2.0 * fake.dt
    with pytest.raises(ValueError, match="environment.dt"):
        NekRSFlowEnv(cfg, run_root=str(tmp_path / "bad"), iface=fake)


def test_nek5000_case_is_re2e5_with_the_ref1_trip():
    p = _rea_params(os.path.join(CASE, "naca_wing.rea"))
    assert p["p002"] == -2.0e5  # Re_c
    assert p["p012"] == -4.0e-5  # fixed dt
    assert p["p089"] == 31  # solver steps per action
    assert p["p090"] > 0  # zero net mass flux per spanwise line
    tamps, tampt, tdt, modes, _ = _wall_params(os.path.join(CASE, "naca_wing.wall"))
    assert (tamps, tampt, tdt, modes) == (0.0, 10.0, 0.008, 30)
    src = open(os.path.join(CASE, "drl", "drl_action.f")).read()
    assert re.search(r"^\s+uti = 0\.0624334733", src, re.M)


def test_wing_yml_is_the_nek5000_deployment(tmp_path):
    p = _rea_params(os.path.join(CASE, "naca_wing.rea"))
    env = NekRSFlowEnv(os.path.join(WING, "wing.yml"), run_root=str(tmp_path / "w"), iface=WingIface(4e-5))
    assert env.cfg["environment"]["dt"] == -p["p012"]
    assert env.steps_per_action == p["p089"]
    assert env.nu == pytest.approx(-1.0 / p["p002"])
    assert env._frame_kind == "wall"
    assert np.allclose(env.amplitude, U_TAU_CH)
    ut = _x2utau(env.actuators["xyz"][:, 0])
    assert np.allclose(env._obs_div, ut[:, None])
    assert env._znmf_groups is not None
    env.close()
    par = open(os.path.join(WING, "naca_wing.par")).read()
    assert re.search(r"^dt = 4e-5", par, re.M)
    assert re.search(r"^tripModes = 30\b", par, re.M) and re.search(r"^tripInterval = 0\.008\b", par, re.M)


@pytest.mark.parametrize(
    "name, dt, steps",
    [("wing_smoke.yml", 4e-5, 31), ("wing_oc.yml", 2e-5, 10), ("wing_dt8e-5.yml", 8e-5, 25)],
)
def test_wing_variants(tmp_path, name, dt, steps):
    env = NekRSFlowEnv(os.path.join(WING, name), run_root=str(tmp_path / "v"), iface=WingIface(dt))
    assert env.steps_per_action == steps
    env.close()
    with pytest.raises(ValueError, match="environment.dt"):
        NekRSFlowEnv(os.path.join(WING, name), run_root=str(tmp_path / "x"), iface=WingIface(1.5e-5))


def test_wing_hg_fortran_yml_keeps_the_original_projection(tmp_path):
    """wing_hg_fortran.yml = wing.yml with the projection of the original Nek5000 coupling, nothing else changed."""
    a = NekRSFlowEnv(os.path.join(WING, "wing.yml"), run_root=str(tmp_path / "a"), iface=WingIface(4e-5))
    b = NekRSFlowEnv(os.path.join(WING, "wing_hg_fortran.yml"), run_root=str(tmp_path / "b"), iface=WingIface(4e-5))
    assert (a._frame_kind, b._frame_kind) == ("wall", "hg_fortran")
    ca, cb = dict(a.cfg), dict(b.cfg)
    pa, pb = dict(ca.pop("probes")), dict(cb.pop("probes"))
    assert pa.pop("frame") == "wall" and pb.pop("frame") == "hg_fortran" and pa == pb
    assert ca == cb
    a.close()
    b.close()
