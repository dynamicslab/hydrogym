"""Wall-jet actuation power of the m-AIA environments (BC 2007 on flat walls, airfoils, spheres): analytical fields."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "maia_wall_jets", Path(__file__).resolve().parents[1] / "hydrogym" / "maia" / "wall_jets.py"
)
wj = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wj  # (dataclasses look the module up)
_spec.loader.exec_module(wj)


def probe_data(model, field):
    """field(points) -> (u, v[, w], rho, p) at the model's probe cells, in the m-AIA probe layout."""
    pts = np.asarray(model.probe_coordinates).reshape(-1, model.n_dim)
    return np.stack(field(pts), 1).ravel()


def uniform(p0=1.0 / 3.0, rho=1.0, d=2):
    return lambda x: [np.zeros(len(x))] * d + [np.full(len(x), rho), np.full(len(x), p0)]


# -------------------------------------------------------------------------------------------- BC 2007 tagging rule
def cavity_jets(object_class=0):
    props = dict(
        nDim=2,
        lbNoJets=1,
        lbObjectClass=object_class,
        lbJetCenter=[1.175, 0.0],
        lbJetRadius=[0.0],
        lbJetAngles=[np.pi / 2],
        lbJetRanges=[0.02],
    )
    return wj.Bc2007Jets.from_properties(props)


def test_tag_class0_profile_across_jet_direction():
    jets = cavity_jets(0)
    x = np.array([[1.175, 0.002], [1.18, 0.002], [1.185 - 1e-9, 0.002], [1.186, 0.002], [1.175, 0.02]])
    tag, fac = jets.tag(x)
    # profile 1 - d^2 / h^2 across the jet direction (independent of the distance from the wall), along +y
    np.testing.assert_allclose(fac[:3, 1], [1.0, 0.75, 0.0], atol=1e-6)
    np.testing.assert_allclose(fac[:, 0], 0.0, atol=1e-9)
    assert list(tag) == [0, 0, 0, -1, -1]  # outside the box: no tag (box half height 0.01)


def test_tag_class1_circle_includes_wall_offset():
    jets = cavity_jets(1)
    tag, fac = jets.tag(np.array([[1.175, 0.005], [1.18, 0.005]]))
    np.testing.assert_allclose(fac[:, 1], [1 - 0.25, 1 - 0.5], atol=1e-12)  # (dx^2 + dy^2) / h^2


def test_tag_2d_box_decides_tag_profile_decides_factor():
    # m-AIA 2D: the box test sets the tag, the profile test the factor; a later jet's box overrides the tag
    props = dict(
        nDim=2,
        lbNoJets=2,
        lbObjectClass=1,
        lbJetCenter=[0.0, 0.0, 0.0009, 0.009],
        lbJetRadius=[0.0, 0.0],
        lbJetAngles=[np.pi / 2, np.pi / 2],
        lbJetRanges=[0.02, 0.002],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    tag, fac = jets.tag(np.array([[0.0, 0.0085]]))  # in jet 0's circle, in jet 1's box but outside its circle
    assert tag[0] == 1 and fac[0, 1] == pytest.approx(1 - 0.0085**2 / 0.01**2)


def test_tag_cube_slot_and_limits():
    props = dict(
        nDim=3,
        lbNoJets=1,
        lbObjectClass=2,
        lbJetCenter=[0.425, 0.5, 0.0],
        lbJetNormals=[0, 1, 0],
        lbJetSlotAxis=[0],
        lbJetRanges=[1.0],
        lbJetRadius=[0.5],
        lbJetLimits=[0.05, 0.01, 0.49, 0.05, 0.01, 0.49],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    tag, fac = jets.tag(np.array([[0.425, 0.502, 0.3], [0.45, 0.502, 0.0], [0.45, 0.502, 0.495], [0.45, 0.52, 0.0]]))
    np.testing.assert_allclose(fac[:2, 1], [1.0, 0.75], atol=1e-12)
    assert list(tag) == [0, 0, -1, -1]


def test_tag_sphere_arc_width_and_azimuth():
    R = 0.5
    th = np.arccos(0.1 / R)
    props = dict(
        nDim=3,
        lbNoJets=1,
        lbObjectClass=3,
        lbJetCenterSph=[0, 0, 0],
        lbJetRadius=[R],
        lbJetCenter=[0.1, 0.0, R * np.sin(th)],
        lbJetSlotAxis=[0],
        lbJetRanges=[np.pi / 3],
        lbJetLimits=[0.025, 0.5, 0.5, 0.025, 0.5, 0.5],
    )
    jets = wj.Bc2007Jets.from_properties(props)

    def on(theta, phi):
        return R * np.array([np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)])

    pts = np.array([on(th, 0.0), on(th + 0.0125 / R, 0.0), on(th, 0.5), on(th, 0.53), on(th + 0.03 / R, 0.0)])
    tag, fac = jets.tag(pts)
    assert list(tag) == [0, 0, 0, -1, -1]  # azimuthal half range pi/6 = 0.5236, arc half width 0.025
    np.testing.assert_allclose(np.linalg.norm(fac[:3], axis=1), [1.0, 0.75, 1.0], atol=1e-9)
    np.testing.assert_allclose(fac[0] / np.linalg.norm(fac[0]), pts[0] / R, atol=1e-12)  # radial


# ------------------------------------------------------------------------------------------------ jet geometries
def test_flat_wall_cells_are_the_fluid_cells_next_to_the_wall():
    dX = 1.0 / 240
    el = wj.flat_wall_jets(cavity_jets(0), dX, (-1.0 - 100 * dX, -1.0))
    # slot 0.02 = 4.8 cells: 4 cells with centres in (1.165, 1.185), one element per cell, factor of the cell centre
    assert len(el) == 4
    np.testing.assert_allclose(el.points[:, 1], 0.0)
    np.testing.assert_allclose(el.weights, 1.0)
    xc = el.points[:, 0]
    np.testing.assert_allclose(el.factors[:, 1], 1 - (xc - 1.175) ** 2 / 0.01**2)
    frac = (xc - (-1.0 - 100 * dX)) / dX - 0.5
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)


def test_link_model_equals_flat_cells_on_a_face_aligned_wall():
    # square cylinder (D/20): the link-level model on the polygon equals the cell model on the faces
    dX = 0.05
    props = dict(
        nDim=2,
        lbNoJets=2,
        lbJetCenter=[0, 0, 0, 0],
        lbJetRadius=[0.5, 0.5],
        lbJetAngles=[np.pi / 2, 3 * np.pi / 2],
        lbJetRanges=[0.2, 0.2],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    origin = (-16.0, -25.6)
    flat = wj.flat_wall_jets(jets, dX, origin)
    sq = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    link = wj.curve_link_jets(jets, np.stack([sq[:-1], sq[1:]], 1), dX, origin)
    for el in (flat, link):
        fl = np.bincount(el.jet, weights=el.weights * np.sum(el.factors * el.normals, 1))
        np.testing.assert_allclose(fl, [2.75, 2.75], atol=1e-9)  # 1 + 2 * 0.9375 - 2 * 0.0625 ... cells 0.5, 1.5
    np.testing.assert_allclose(link.links["q"], 0.5)  # walls on cell faces: no interpolation terms


def off_face_jet():
    # wall at y = -0.2: 0.7 cells below the fluid centres (y = 0.5), 0.3 above the solid centres (y = -0.5); every
    # cut link of a fluid cell is cut beyond the half link, so the solid cells write them (as in the solver)
    props = dict(
        nDim=2,
        lbNoJets=1,
        lbObjectClass=0,
        lbJetCenter=[0.0, -0.2],
        lbJetRadius=[0.0],
        lbJetAngles=[np.pi / 2],
        lbJetRanges=[6.0],
    )
    return wj.flat_wall_jets(wj.Bc2007Jets.from_properties(props), 1.0, (-50.0, -50.0))


SLOT = {x: 1 - (x / 3.0) ** 2 for x in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)}


def test_off_face_wall_solid_cells_write():
    el = off_face_jet()
    flux = np.sum(el.weights * np.sum(el.factors * el.normals, 1))
    # the source of each link comes from the solid cell across it, scaled by 1 / (2 * 0.7); the diagonal links of
    # the fluid cells just beyond the slot also receive the sources of the slot's solid cells
    assert flux == pytest.approx(sum(SLOT.values()) / (2 * 0.7), rel=1e-9)
    assert np.all(el.links["q"] == pytest.approx(0.7))


def test_interpolation_terms_restore_the_wall_flux():
    # fluid moving with the wall velocity of its own cell: the interpolation terms add (2q - 1) / (2q) of the cell's
    # source, so the solver's mass flux is the full sum of the cell velocities (no 1 / (2q) loss)
    el = off_face_jet()
    m = wj.WallJetPower(el, 1, (-50.0, -50.0), 1.0, 0.1, 10.0)
    a = 0.01

    def field(x):
        f = np.clip(1 - (x[:, 0] / 3.0) ** 2, 0.0, None) * (np.abs(x[:, 0]) < 3.0)
        return [np.zeros(len(x)), a * f, np.ones(len(x)), np.full(len(x), 1 / 3)]

    m.evaluate(probe_data(m, field), [a])
    assert m.last_mass_flux[0] == pytest.approx(a * sum(SLOT.values()), rel=1e-9)
    assert m.mass_flux([a])[0] == pytest.approx(a * sum(SLOT.values()) / 1.4, rel=1e-9)
    # fluid at rest: no interpolation term
    m.evaluate(probe_data(m, uniform()), [a])
    assert m.last_mass_flux[0] == pytest.approx(m.mass_flux([a])[0], rel=1e-12)


def test_sphere_link_flux_matches_surface_integral():
    R, dX = 0.5, 1.0 / 64
    props = dict(
        nDim=3,
        lbNoJets=1,
        lbObjectClass=3,
        lbJetCenterSph=[0, 0, 0],
        lbJetRadius=[R],
        lbJetCenter=[0.0, 0.0, R],
        lbJetSlotAxis=[0],
        lbJetRanges=[2.0],
        lbJetLimits=[0.1, 1, 1, 0.1, 1, 1],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    cont = wj.sphere_jets(jets, dX, cell_offset=0.5, max_phi=512)
    fc = np.sum(cont.weights * np.sum(cont.factors * cont.normals, 1))
    analytic = (4 / 3) * (0.1 / dX) * (R * 2.0 / dX)  # parabola mean 2/3 over the width 2 h, times the arc length
    assert fc == pytest.approx(analytic, rel=0.03)
    # link model with the fluid moving at its cells' wall velocity: the solver's flux equals the surface integral
    el = wj.sphere_link_jets(jets, dX, (-1.0, -1.0, -1.0))
    m = wj.WallJetPower(el, 1, (-1.0, -1.0, -1.0), dX, 0.1, 1.0, wall_distances=(2, 3, 4, 5))
    a = 0.01

    def field(x):
        _, fac = jets.tag(x)
        return [a * fac[:, 0], a * fac[:, 1], a * fac[:, 2], np.ones(len(x)), np.full(len(x), 1 / 3)]

    m.evaluate(probe_data(m, field), [a])
    assert m.last_mass_flux[0] == pytest.approx(a * fc, rel=0.03)


# ------------------------------------------------------------------------------------------------------ the power
def square_model(**kw):
    dX = 0.05
    props = dict(
        nDim=2,
        lbNoJets=2,
        lbJetCenter=[0, 0, 0, 0],
        lbJetRadius=[0.5, 0.5],
        lbJetAngles=[np.pi / 2, 3 * np.pi / 2],
        lbJetRanges=[0.2, 0.2],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    el = wj.flat_wall_jets(jets, dX, (-16.0, -25.6))
    return wj.WallJetPower(el, 2, (-16.0, -25.6), dX, 0.1, 1.0 / dX, **kw)


def test_uniform_pressure_only_kinetic_terms():
    m = square_model()
    a = np.array([0.02, 0.01])
    ev = m.evaluate(probe_data(m, uniform(p0=0.4, rho=1.0)), a)
    fac = m.el.factors[:, 1] * np.sign(m.el.normals[:, 1])
    kin = np.bincount(m.el.jet, weights=0.5 * (a[m.el.jet] * fac) ** 3)
    np.testing.assert_allclose(ev["per_actuator"], m.coefficient(kin), rtol=1e-12)
    assert ev["forms"]["signed"] == pytest.approx(ev["forms"]["abs"]) == pytest.approx(ev["forms"]["positive"])
    # suction on one jet: the kinetic term is negative there, signed < positive <= abs
    ev2 = m.evaluate(probe_data(m, uniform(p0=0.4)), np.array([0.02, -0.01]))
    assert ev2["forms"]["signed"] < ev2["forms"]["positive"] <= ev2["forms"]["abs"]


def test_gauge_invariance_of_all_forms():
    m = square_model()

    def field(p_add):
        return lambda x: [np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), 0.3 + 0.01 * x[:, 0] + p_add]

    a = np.array([0.02, -0.02])
    e0, e1 = m.evaluate(probe_data(m, field(0.0)), a), m.evaluate(probe_data(m, field(5.0)), a)
    for k in ("signed", "positive", "abs"):
        assert e0["forms"][k] == pytest.approx(e1["forms"][k], rel=1e-10, abs=1e-14)


def test_pressure_difference_signed_power():
    # p = p0 + g y: top wall pressure p0 + g/2, bottom p0 - g/2; with (a, -a) (ZNMF) signed P = a Q_unit g / ...
    m = square_model()
    g, a = 0.01, 0.02

    def field(x):
        return [np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), 1 / 3 + g * x[:, 1]]

    ev = m.evaluate(probe_data(m, field), np.array([a, -a]))
    q = m.mass_flux([a, -a])  # top blows +Q, bottom: v_n = -a * factor * (-1)... both into the fluid sign
    p_top, p_bot = 1 / 3 + g * 0.5, 1 / 3 - g * 0.5
    p_ref = 0.5 * (p_top + p_bot)
    kin = np.bincount(
        m.el.jet,
        weights=m.el.weights
        * 0.5
        * np.sum((np.array([a, -a])[m.el.jet, None] * m.el.factors) ** 2, 1)
        * np.sum(np.array([a, -a])[m.el.jet, None] * m.el.factors * m.el.normals, 1),
    )
    expected = m.coefficient(q * (np.array([p_top, p_bot]) - p_ref) + kin)
    np.testing.assert_allclose(ev["per_actuator"], expected, rtol=1e-9)


def test_wall_extrapolation_of_known_pressure():
    # quadratic wall-normal pressure profile: the quadratic fit is exact, the linear one is not (spread > 0)
    m = square_model()

    def field(x):
        dist = (np.abs(x[:, 1]) - 0.5) / 0.05  # cells from the top/bottom faces
        return [
            np.zeros(len(x)),
            np.zeros(len(x)),
            np.ones(len(x)),
            0.3 + 0.002 * dist + 0.001 * dist**2 + np.where(x[:, 1] > 0, 0.01, 0.0),
        ]

    p_w, _ = m.wall_state(probe_data(m, field), degree=2)
    np.testing.assert_allclose(p_w, 0.3 + np.where(m.el.points[:, 1] > 0, 0.01, 0.0), atol=1e-12)
    m.evaluate(probe_data(m, field), np.array([0.02, 0.02]))
    assert m.last_extrapolation_spread > 0


def test_probes_are_cell_centres_outside_the_body():
    m = square_model()
    pts = np.asarray(m.probe_coordinates).reshape(-1, 2)
    frac = (pts - np.array([-16.0, -25.6])) / 0.05 - 0.5
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)
    assert np.all(np.abs(pts[:, 1]) > 0.5)


def test_stencil_inside_body_is_rejected():
    R, dX = 0.5, 1.0 / 64
    props = dict(
        nDim=3,
        lbNoJets=1,
        lbObjectClass=3,
        lbJetCenterSph=[0, 0, 0],
        lbJetRadius=[R],
        lbJetCenter=[0.0, R / np.sqrt(2), R / np.sqrt(2)],
        lbJetSlotAxis=[0],
        lbJetRanges=[1.0],
        lbJetLimits=[0.05, 1, 1, 0.05, 1, 1],
    )
    el = wj.sphere_link_jets(wj.Bc2007Jets.from_properties(props), dX, (-1.0, -1.0, -1.0))
    wj.WallJetPower(el, 1, (-1.0, -1.0, -1.0), dX, 0.1, 1.0, wall_distances=(2, 3, 4, 5))  # fine
    with pytest.raises(ValueError, match="reaches into the body"):
        wj.WallJetPower(el, 1, (-1.0, -1.0, -1.0), dX, 0.1, 1.0, wall_distances=(1.0, 2.0, 3.0))


def test_stl_section_and_inside(tmp_path):
    # extruded unit square as an ASCII STL: the section at z = 0.3 is the square
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    lines = ["solid s"]
    for k in range(4):
        (x0, y0), (x1, y1) = sq[k], sq[(k + 1) % 4]
        for tri in ([(x0, y0, 0), (x1, y1, 0), (x1, y1, 1)], [(x0, y0, 0), (x1, y1, 1), (x0, y0, 1)]):
            lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {a} {b} {c}" for a, b, c in tri]
            lines += ["endloop", "endfacet"]
    lines.append("endsolid s")
    f = tmp_path / "sq.stl"
    f.write_text("\n".join(lines))
    segs = wj.read_stl_section(str(f), 0.3)
    assert np.sum(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)) == pytest.approx(4.0)
    inside = wj.segments_inside(segs)
    assert list(inside(np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.2]]))) == [True, False, False]


def test_near_links_interpolation_term():
    # wall 0.3 cells below the fluid centres (q = 0.3): the fluid cells write; the interpolation adds
    # (1 - 2q) [f_in(next cell out) - f_in(cell)], zero for a uniform flow, (1 - 2q) w (rho' - rho) for a density step
    props = dict(
        nDim=2,
        lbNoJets=1,
        lbObjectClass=0,
        lbJetCenter=[0.0, 0.2],
        lbJetRadius=[0.0],
        lbJetAngles=[np.pi / 2],
        lbJetRanges=[6.0],
    )
    el = wj.flat_wall_jets(wj.Bc2007Jets.from_properties(props), 1.0, (-50.0, -50.0))
    assert np.all(el.links["q"] == pytest.approx(0.3)) and not np.any(el.links["far"])
    m = wj.WallJetPower(el, 1, (-50.0, -50.0), 1.0, 0.1, 10.0)
    m.evaluate(probe_data(m, uniform()), [0.01])
    assert m.last_mass_flux[0] == pytest.approx(m.mass_flux([0.01])[0], rel=1e-12)

    def step(x):  # density 1 at the wall cells (y = 0.5), 1.01 further out
        return [np.zeros(len(x)), np.zeros(len(x)), np.where(x[:, 1] > 1.0, 1.01, 1.0), np.full(len(x), 1 / 3)]

    corr = m.interpolation_flux(probe_data(m, step))
    # per wall cell: three links (weights 1/9, 1/36, 1/36) -> (1 - 0.6) * 0.01 / 6
    np.testing.assert_allclose(corr[np.abs(el.points[:, 0]) < 2.0], 0.4 * 0.01 / 6, rtol=1e-12)


def test_invalid_state_messages_tell_geometry_from_divergence():
    """Square cylinder, sustained suction: a flow that diverges after valid states must not be reported as
    probes outside the fluid domain; NaN probe data must not pass silently."""
    m = square_model()
    a = np.array([0.03, -0.03])
    good = probe_data(m, uniform())
    # zero density at the first evaluation: geometry message
    zero = probe_data(m, uniform(rho=0.0))
    with pytest.raises(RuntimeError, match="outside the fluid domain"):
        square_model().evaluate(zero, a)
    # valid state first, then density <= 0 next to a jet: divergence message
    m.evaluate(good, a)
    broken = good.copy().reshape(-1, 4)
    broken[0, 2] = -1.4
    with pytest.raises(RuntimeError, match="diverged"):
        m.evaluate(broken.ravel(), a)
    # NaN anywhere in the probe data: divergence message, also at the first evaluation
    nan = good.copy()
    nan[3] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        square_model().evaluate(nan, a)
