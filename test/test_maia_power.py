"""m-AIA actuation-power models on analytical flows (no solver needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "maia_power", Path(__file__).resolve().parents[1] / "hydrogym" / "maia" / "power.py"
)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

COMMON = dict(n_dim=2, dX=0.05, Ma=0.1, Re=100.0, reference_length=1.0, z_length=0.05)


def probe_data(model, field, n_dim=2):
    """Evaluate field(x, y) -> (u, v, rho, p) at the model's probe points, in the m-AIA probe layout."""
    pts = np.asarray(model.probe_coordinates).reshape(-1, n_dim)
    u, v, rho, p = field(pts[:, 0], pts[:, 1])
    cols = [u, v, rho, p] if n_dim == 2 else [u, v, np.zeros_like(u), rho, p]
    return np.stack(cols, 1).ravel()


ORIGIN = (-30.0, -51.2)


def vortex(cx, cy, A, reach=0.95):
    def field(x, y):
        dx, dy = (x - cx) / COMMON["dX"], (y - cy) / COMMON["dX"]  # cells
        r2 = dx**2 + dy**2
        near = r2 < (reach / COMMON["dX"]) ** 2
        return (
            np.where(near, -A * dy / r2, 0.0),
            np.where(near, A * dx / r2, 0.0),
            np.ones_like(x),
            np.full_like(x, 1 / 3),
        )

    return field


def expected_cp(model, A, cmd):
    return 2 * (4 * np.pi * model.nu * A * cmd * COMMON["dX"]) / (model.U**3 * model.D_cells)


def test_ring_torque_irrotational_vortex():
    # u_t = A / r around each body (lattice velocity, r in cells): tau_rt = -2 mu A / r^2,
    # T = -4 pi mu A on every circle
    centers = [[-1.299, 0.0], [0.0, 0.75], [0.0, -0.75]]
    model = mp.RingTorquePower(centers, [0.5, 0.5, 0.5], lattice_origin=ORIGIN, **COMMON)
    A = 0.02 * 10.0  # u_t = 0.02 at the wall radius (10 cells)
    cmds = np.array([0.1, -0.25, 0.25])
    for k in range(3):
        cp = model.per_body(probe_data(model, vortex(*centers[k], A)), cmds)
        assert cp[k] == pytest.approx(expected_cp(model, A, cmds[k]), rel=5e-3)
        assert model.last_spread[k] < 1e-2  # (the truncated vortex also reaches the neighbours' annuli: not checked)


def test_probes_are_cell_centres():
    model = mp.RingTorquePower([[0.0, 0.75]], [0.5], lattice_origin=ORIGIN, **COMMON)
    pts = np.asarray(model.probe_coordinates).reshape(-1, 2)
    frac = (pts - np.asarray(ORIGIN)) / COMMON["dX"] - 0.5
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)
    r = np.hypot(pts[:, 0], pts[:, 1] - 0.75)
    assert r.min() > 0.5 and r.max() < 0.5 * 1.7 + 3 * COMMON["dX"]


def test_annulus_must_not_reach_another_body():
    with pytest.raises(ValueError, match="reaches body"):
        mp.RingTorquePower([[0.0, 0.75], [0.0, -0.75]], [0.5, 0.5], lattice_origin=ORIGIN, annulus=(1.3, 2.1), **COMMON)


def test_ring_rejects_probes_outside_domain():
    model = mp.RingTorquePower([[0.0, 0.0]], [0.5], lattice_origin=ORIGIN, **COMMON)
    data = probe_data(model, lambda x, y: (0 * x, 0 * x, 0 * x, 0 * x))
    with pytest.raises(RuntimeError, match="outside the fluid domain"):
        model.per_body(data, [0.1])


def test_non_finite_probe_data_is_reported_as_invalid_state():
    """A NaN state (e.g. a truncated restart) must not be misreported as probes outside the probed cells."""
    jets = mp.SlotJetPower(
        [[0, 0], [0, 0]], [0.5, 0.5], [np.pi / 2, 3 * np.pi / 2], [0.2, 0.2], lattice_origin=ORIGIN, **COMMON
    )
    data = probe_data(jets, lambda x, y: (0 * x, 0 * x, np.ones_like(x), np.full_like(x, 1.0 / 3.0)))
    data[5] = np.nan
    with pytest.raises(RuntimeError, match="non-finite probe data"):
        jets.per_jet(data, [0.03, -0.03])
    ring = mp.RingTorquePower([[0.0, 0.0]], [0.5], lattice_origin=ORIGIN, **COMMON)
    data = probe_data(ring, lambda x, y: (0 * x, 0 * x, np.ones_like(x), 0 * x))
    data[-1] = np.inf
    with pytest.raises(RuntimeError, match="non-finite probe data"):
        ring.per_body(data, [0.1])


def test_slot_jets_uniform_pressure():
    # Cylinder_2D: two jets at +-90 deg, slot width 0.2, ZNMF pair (a, -a)
    model = mp.SlotJetPower(
        [[0, 0], [0, 0]], [0.5, 0.5], [np.pi / 2, 3 * np.pi / 2], [0.2, 0.2], lattice_origin=ORIGIN, **COMMON
    )
    p0, a = 1.0 / 3.0, 0.03
    data = probe_data(model, lambda x, y: (0 * x, 0 * x, np.ones_like(x), np.full_like(x, p0)))
    cp = model.per_jet(data, [a, -a])
    h = 0.1 / COMMON["dX"]
    per_jet = 0.5 * a**3 * 2 * h * 16 / 35  # uniform pressure = p_ref: only the kinetic-energy flux remains
    expected = model.coefficient(np.array([per_jet, -per_jet]))
    np.testing.assert_allclose(cp, expected, rtol=1e-12)
    assert cp.sum() == pytest.approx(0.0, abs=1e-12)


def test_slot_jets_pressure_difference_and_gauge_invariance():
    model = mp.SlotJetPower(
        [[0, 0], [0, 0]], [0.5, 0.5], [np.pi / 2, 3 * np.pi / 2], [0.2, 0.2], lattice_origin=ORIGIN, **COMMON
    )
    a, dp = 0.03, 1e-3

    def field(gauge):
        return lambda x, y: (0 * x, 0 * x, np.ones_like(x), gauge + np.where(y > 0, dp, -dp))

    cp1 = model.per_jet(probe_data(model, field(1 / 3)), [a, -a])
    cp2 = model.per_jet(probe_data(model, field(5.0)), [a, -a])
    np.testing.assert_allclose(cp1, cp2, rtol=1e-9)  # independent of the pressure gauge in every form
    h = 0.1 / COMMON["dX"]
    flux = a * 4 * h / 3  # int v_n ds of the blowing jet
    kin = 0.5 * a**3 * 2 * h * 16 / 35
    np.testing.assert_allclose(cp1, model.coefficient(np.array([flux * dp + kin, flux * dp - kin])), rtol=1e-9)


def test_slot_jet_probes_are_fluid_cell_centres():
    model = mp.SlotJetPower([[0, 0]], [0.5], [np.pi / 2], [0.2], lattice_origin=ORIGIN, **COMMON)
    pts = np.asarray(model.probe_coordinates).reshape(-1, 2)
    frac = (pts - np.asarray(ORIGIN)) / COMMON["dX"] - 0.5
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)
    assert np.hypot(pts[:, 0], pts[:, 1]).min() >= 0.5 + 0.5 * COMMON["dX"] - 1e-12
    smp = model.samples[0]  # (n_slot, n_dist, 2): on the wall normals at 2..5 cells
    r = np.hypot(smp[..., 0], smp[..., 1])
    np.testing.assert_allclose(r, np.broadcast_to(0.5 + model.dist[None, :] * COMMON["dX"], r.shape))


def test_wall_pressure_extrapolation():
    # pressure varying with the wall distance d (cells): p = p0 + g d + k d^2 -> the wall value p0 is recovered
    model = mp.SlotJetPower(
        [[0, 0], [0, 0]], [0.5, 0.5], [np.pi / 2, 3 * np.pi / 2], [0.2, 0.2], lattice_origin=ORIGIN, **COMMON
    )
    p0, g, k = 1 / 3, 2e-4, -1e-5

    def field(x, y):
        d = (np.hypot(x, y) - 0.5) / COMMON["dX"]
        return 0 * x, 0 * x, np.ones_like(x), p0 + g * d + k * d**2

    pw, rw = model.wall_state(probe_data(model, field))
    # bilinear interpolation of the curved field: error 0.3 % of the pressure change over the samples (1e-3)
    np.testing.assert_allclose(pw, p0, atol=5e-6)
    np.testing.assert_allclose(rw, 1.0, atol=1e-12)


def test_3d_planes_and_axis_sign():
    props = {"nDim": 3, "lbJetCenter": [0.0, 0.0, 1.0], "lbJetRadius": [0.5], "lbRotationAxis": [0.0, 0.0, -1.0]}
    centers, radii, zc, signs = mp.rotating_bodies(props, 1)
    assert signs[0] == -1.0 and zc[0] == 1.0
    model = mp.RingTorquePower(
        centers,
        radii,
        lattice_origin=ORIGIN,
        axis_signs=signs,
        z_center=1.0,
        n_planes=4,
        **{**COMMON, "n_dim": 3, "z_length": 2.0},
    )
    pts = np.asarray(model.probe_coordinates).reshape(-1, 3)
    np.testing.assert_allclose(np.unique(pts[:, 2]), [0.25, 0.75, 1.25, 1.75])
    assert mp.rotating_bodies({**props, "lbRotationAxis": [1.0, 0.0, 0.0]}, 1) is None


@pytest.mark.parametrize("annulus", [(1.3, 1.7), (1.3, 1.5), (1.5, 1.9), (1.4, 1.6)])
def test_ring_torque_independent_of_annulus(annulus):
    A, cmd = 0.2, 0.25
    model = mp.RingTorquePower([[0.0, 0.0]], [0.5], lattice_origin=ORIGIN, annulus=annulus, **COMMON)
    cp = model.per_body(probe_data(model, vortex(0.0, 0.0, A, reach=5.0)), [cmd])[0]
    assert cp == pytest.approx(expected_cp(model, A, cmd), rel=5e-3)


def test_lattice_origin_from_cdf_header(tmp_path):
    import struct

    def name(s):
        b = s.encode()
        return struct.pack(">q", len(b)) + b + b"\x00" * ((4 - len(b) % 4) % 4)

    hdr = b"CDF\x05" + struct.pack(">q", 0) + struct.pack(">i", 0x0A) + struct.pack(">q", 1) + name("dim0")
    hdr += struct.pack(">q", 3) + struct.pack(">i", 0x0C) + struct.pack(">q", 2)
    hdr += name("centerOfGravity") + struct.pack(">i", 6) + struct.pack(">q", 2) + struct.pack(">dd", 21.2, 0.0)
    hdr += name("lengthLevel0") + struct.pack(">i", 6) + struct.pack(">q", 1) + struct.pack(">d", 102.4)
    f = tmp_path / "grid.Netcdf"
    f.write_bytes(hdr + b"\x00" * 64)
    np.testing.assert_allclose(mp.lattice_origin_from_grid(str(f), 0.05, 2), [-30.0, -51.2])
    with pytest.raises(ValueError):
        mp.lattice_origin_from_grid(str(f), 0.07, 2)


def test_check_cell_size_against_grid_header(tmp_path):
    """The property-file dX must match the grid."""
    import struct

    def name(s):
        b = s.encode()
        return struct.pack(">q", len(b)) + b + b"\x00" * ((4 - len(b) % 4) % 4)

    hdr = b"CDF\x05" + struct.pack(">q", 0) + struct.pack(">i", 0x0A) + struct.pack(">q", 1) + name("dim0")
    hdr += struct.pack(">q", 3) + struct.pack(">i", 0x0C) + struct.pack(">q", 2)
    hdr += name("lengthLevel0") + struct.pack(">i", 6) + struct.pack(">q", 1) + struct.pack(">d", 8.00000000745058)
    hdr += name("maxLevel") + struct.pack(">i", 4) + struct.pack(">q", 1) + struct.pack(">i", 11)
    f = tmp_path / "grid.Netcdf"
    f.write_bytes(hdr + b"\x00" * 64)
    assert mp.grid_finest_cell_size(str(f)) == pytest.approx(8.0 / 2**11)
    assert mp.check_cell_size(str(f), 8.0 / 2**11) == pytest.approx(0.00390625)
    with pytest.raises(ValueError, match="differs from dX"):
        mp.check_cell_size(str(f), 4.5 / 2**11)


def test_annulus_needs_fluid_cells_to_the_wall():
    with pytest.raises(ValueError, match="3 fluid cells"):
        mp.RingTorquePower([[0.0, 0.0]], [0.5], lattice_origin=ORIGIN, annulus=(1.2, 1.5), **COMMON)


def test_rotation_change_flags_transient_actuation():
    assert mp.rotation_change(None, [0.0, -0.25, 0.25], 0.25) == pytest.approx(1.0)  # from rest to the bound
    assert mp.rotation_change([0.0, -0.25, 0.25], [0.0, -0.25, 0.25], 0.25) == 0.0  # steady
    assert mp.rotation_change([0.0, -0.25, 0.25], [0.01, -0.24, 0.25], 0.25) <= mp.ROTATION_STEADY_TOL
    assert mp.rotation_change([0.25, -0.25, 0.25], [-0.25, -0.25, 0.25], 0.25) == pytest.approx(2.0)  # reversal
