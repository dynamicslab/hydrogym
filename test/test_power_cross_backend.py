"""The transpiration power of the m-AIA slot-jet model and of the NekRS wall model is the same computation.

The same synthetic jet pair (points, weights, normals, wall velocity, pressure) is evaluated by
hydrogym.maia.power.SlotJetPower (from probe data) and by hydrogym.nekrs.power.WallPower (from wall-node data):
both must give identical signed / positive / abs powers and per-jet values, for any pressure gauge.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

try:  # hydrogym.nekrs imports mpi4py at module level; WallPower itself never uses MPI
    import mpi4py  # noqa: F401

    from hydrogym.nekrs.power import WallPower
except ImportError:
    # Import WallPower under a temporary mpi4py stub, then remove the stub and every module imported with it, so that
    # later tests still see the real "no mpi4py" state (their importorskip must skip, not pick up the stub).
    _before = set(sys.modules)
    _mpi = types.ModuleType("mpi4py.MPI")
    _mpi.Comm = object
    _mpi.Intercomm = object
    _mpi.COMM_WORLD = None
    _pkg = types.ModuleType("mpi4py")
    _pkg.MPI = _mpi
    sys.modules["mpi4py"] = _pkg
    sys.modules["mpi4py.MPI"] = _mpi
    try:
        from hydrogym.nekrs.power import WallPower
    finally:
        for _name in set(sys.modules) - _before:
            if _name.startswith(("mpi4py", "hydrogym.nekrs")):
                del sys.modules[_name]
        if hasattr(sys.modules.get("hydrogym"), "nekrs"):
            delattr(sys.modules["hydrogym"], "nekrs")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("maia_power_x", ROOT / "hydrogym" / "maia" / "power.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

COMMON = dict(n_dim=2, dX=0.05, Ma=0.1, Re=200.0, reference_length=1.0, z_length=0.05)
ORIGIN = (-30.0, -51.2)


def _jets():
    return mp.SlotJetPower(
        [[0, 0], [0, 0]], [0.5, 0.5], [np.pi / 2, 3 * np.pi / 2], [0.2, 0.2], lattice_origin=ORIGIN, **COMMON
    )


def _probe(model, pfield):
    pts = np.asarray(model.probe_coordinates).reshape(-1, 2)
    p = pfield(pts[:, 0], pts[:, 1])
    return np.stack([0 * p, 0 * p, np.ones_like(p), p], 1).ravel()  # NekRS is incompressible: rho = 1


def _field(gauge):
    return lambda x, y: gauge + 2e-3 * np.sin(3 * x) + 1.5e-3 * y + 4e-4 * np.cos(7 * np.arctan2(y, x))


@pytest.mark.parametrize("gauge", [1.0 / 3.0, 7.0])
def test_slot_jets_equal_wallpower(gauge):
    """Same wall pressure at the same wall points: m-AIA slot-jet forms == NekRS WallPower forms."""
    model = _jets()
    a = np.array([0.03, -0.03])
    data = _probe(model, _field(gauge))
    ev = model.evaluate(data, a)
    p_wall, _ = model.wall_state(data)  # (planes, jets, points)

    ns = len(model.s_unit)
    feet = np.concatenate(
        [
            model.samples[j][:, 0, :]
            - model.dist[0]
            * model.dX
            * np.stack(
                [
                    np.cos(np.arctan2(model.samples[j][:, 0, 1], model.samples[j][:, 0, 0])),
                    np.sin(np.arctan2(model.samples[j][:, 0, 1], model.samples[j][:, 0, 0])),
                ],
                1,
            )
            for j in range(2)
        ]
    )
    vn = np.repeat(a, ns) * np.tile(1.0 - model.s_unit**2, 2)
    ang = np.arctan2(feet[:, 1], feet[:, 0])
    normal = np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1)
    w = np.tile(model.w_unit, 2) * np.repeat(model.half_widths / model.dX, ns)
    xyz = np.stack([feet[:, 0], feet[:, 1], np.zeros(len(feet))], 1)  # one plane: one ZNMF group
    wp = WallPower(
        xyz,
        w,
        normal,
        np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]),
        reference_power=0.5 * model.U**3 * model.l_proj,
        pressure_mean=("z",),
    )
    out = wp.evaluate(p_wall.ravel(), normal * vn[:, None])

    for k in ("signed", "positive", "abs"):
        assert ev["forms"][k] == pytest.approx(out[k], rel=1e-12, abs=1e-15), k
    # per-jet sums cancel a large gauge against O(1e-3) pressure variations: round-off ~1e-12 relative
    np.testing.assert_allclose(ev["per_actuator"], out["per_actuator"], rtol=1e-9, atol=1e-13)
    assert ev["forms"]["abs"] > abs(ev["forms"]["signed"])  # the pressure field makes the forms differ


def test_gauge_independence_of_every_form():
    model = _jets()
    res = [model.evaluate(_probe(model, _field(g)), [0.03, -0.03])["forms"] for g in (0.0, 1.0 / 3.0, 50.0)]
    for k in ("signed", "positive", "abs"):
        np.testing.assert_allclose([r[k] for r in res], res[0][k], rtol=1e-8, atol=1e-12)
