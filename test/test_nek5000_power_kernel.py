"""Nek5000 NACA0012 wing case: actuation-power kernel and observation frame against the NekRS definitions.

Compiles the cost-form kernel of ``drl/drl_power.f`` and the two velocity projections of ``drl/drl_state.f``
(both extracted verbatim from the case sources) with the flags of the case build, runs them on three MPI ranks
(one without wall nodes) and compares them with

* ``hydrogym.power.transpiration_forms`` and the node terms of ``hydrogym.nekrs.power.WallPower``
  (pi_p = v_n (p - p_ref), pi_k = v_n |u_w|^2 / 2), summed over the ranks as Nek5000's gop does;
* the frames of the NekRS backend (``probes.frame``): ``hg_fortran`` u_t = u n_y + v n_x and ``wall``
  u_t = t . u with the downstream tangent t = (n_y, -n_x) (t_x > 0).

Skipped when no MPI Fortran compiler or MPI launcher is available (NEK_TEST_MPIRUN, NEK_TEST_MPIRUN_ARGS
override the launcher).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hydrogym.power import transpiration_forms

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "third_party" / "nek5000" / "cases" / "naca0012_200k"
DRIVER = Path(__file__).resolve().parent / "nek5000" / "power_kernel_driver.f"
FC = shutil.which("mpif77") or shutil.which("mpifort")
MPIRUN = (
    os.environ.get("NEK_TEST_MPIRUN") or shutil.which("mpirun") or shutil.which("mpiexec") or shutil.which("prterun")
)
FLAGS = ["-std=legacy", "-fdefault-real-8", "-fdefault-double-8", "-O3", "-cpp", "-w"]

pytestmark = pytest.mark.skipif(FC is None or MPIRUN is None, reason="needs mpif77/mpifort and an MPI launcher")


def _kernel_source() -> str:
    src = (CASE / "drl" / "drl_power.f").read_text()
    m = re.search(r"^      subroutine drl_power_kernel\(.*?^      end\n", src, flags=re.S | re.M)
    assert m, "drl_power_kernel not found in drl_power.f"
    return m.group(0)


def _projection_lines():
    src = (CASE / "drl" / "drl_state.f").read_text()
    orig = re.findall(r"^\s*rotV=vx\*body_sin \+ vy\*body_cos\s*$", src, flags=re.M)
    wall = re.findall(r"^\s*rotV=vx\*body_sin - vy\*body_cos\s*$", src, flags=re.M)
    assert len(orig) == 1 and len(wall) == 1, "projection statements of normal_V_project not found"
    return orig[0].rstrip() + "\n", wall[0].rstrip() + "\n"


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    d = tmp_path_factory.mktemp("nek5000_power")
    (d / "kernel.f").write_text(_kernel_source())
    orig, wall = _projection_lines()
    (d / "proj_orig.inc").write_text(orig)
    (d / "proj_wall.inc").write_text(wall)
    shutil.copy(DRIVER, d / "driver.f")
    subprocess.run([FC, *FLAGS, "-I", str(d), "driver.f", "kernel.f", "-o", "pwtest"], cwd=d, check=True)
    return d


def _floats(path):
    return np.array(path.read_text().split(), dtype=float)


def _run(d, nranks):
    cmd = [MPIRUN, *os.environ.get("NEK_TEST_MPIRUN_ARGS", "").split(), "-np", str(nranks), "./pwtest"]
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    subprocess.run(cmd, cwd=d, check=True, env=env, timeout=120)


def test_power_kernel_matches_nekrs_forms(build):
    rng = np.random.default_rng(7)
    rho = 1.0
    ranks = []
    for n in (37, 64, 0):  # rank 2 has no wall nodes
        w = rng.uniform(1e-4, 2e-3, n)
        vn = rng.normal(0, 0.05, n)  # blowing and suction
        u = rng.normal(0, 0.05, (n, 3))
        u[:, 0] += vn  # imposed wall velocity with tangential parts
        u2 = (u**2).sum(axis=1)
        p = rng.normal(0.1, 0.02, n)
        pref = p + rng.normal(0, 0.005, n)
        ranks.append((w, vn, u2, p, pref))
        with open(build / f"nodes_{len(ranks) - 1}.txt", "w") as f:
            f.write(f"{n} {rho:.17e}\n")
            for row in zip(w, vn, u2, p, pref):
                f.write(" ".join(f"{x:.17e}" for x in row) + "\n")
    with open(build / "proj.txt", "w") as f:
        f.write("1\n0 0 0 0\n")
    _run(build, len(ranks))

    w, vn, u2, p, pref = (np.concatenate(a) for a in zip(*ranks))
    pi_p = vn * (p - pref)  # hydrogym.nekrs.power.WallPower.node_terms
    pi_k = 0.5 * vn * u2
    ref = transpiration_forms(pi_p, pi_k, w)
    for r, (wr, vnr, u2r, pr_, prefr) in enumerate(ranks):
        out = _floats(build / f"out_{r}.txt")
        s = out[:6]
        np.testing.assert_allclose(s[0], ref["signed"], rtol=1e-12)
        np.testing.assert_allclose(s[1], ref["positive"], rtol=1e-12)
        np.testing.assert_allclose(s[2], ref["abs"], rtol=1e-12)
        np.testing.assert_allclose(s[3], np.sum(w * (vn * p + pi_k)), rtol=1e-12)  # signed_full_pressure
        np.testing.assert_allclose(s[4], w.sum(), rtol=1e-12)
        np.testing.assert_allclose(s[5], np.sum(w * vn), rtol=1e-12)  # wall flux
        pn = out[6:]
        np.testing.assert_allclose(pn, wr * (vnr * (pr_ - prefr) + 0.5 * vnr * u2r), rtol=1e-12, atol=0)
    assert ref["signed"] <= ref["positive"] <= ref["abs"]


def test_power_kernel_density(build):
    """pi_k carries rho (NekRS omits it; identical for rho = 1, the wing cases)."""
    rho = 2.5
    w, vn, u2, p, pref = (np.array([1e-3]), np.array([0.04]), np.array([0.0025]), np.array([0.2]), np.array([0.1]))
    for r in range(3):
        with open(build / f"nodes_{r}.txt", "w") as f:
            if r == 0:
                f.write(f"1 {rho:.17e}\n" + " ".join(f"{x[0]:.17e}" for x in (w, vn, u2, p, pref)) + "\n")
            else:
                f.write(f"0 {rho:.17e}\n")
    _run(build, 3)
    s = _floats(build / "out_0.txt")[:5]
    np.testing.assert_allclose(s[0], (w * (vn * (p - pref) + 0.5 * rho * vn * u2))[0], rtol=1e-13)


def test_frames_match_nekrs(build):
    rng = np.random.default_rng(3)
    n = 200
    theta = rng.uniform(0.05, np.pi - 0.05, n)  # suction side: n_y = body_sin >= 0 by construction (drl_init.f)
    bcos, bsin = np.cos(theta), np.sin(theta)
    vx, vy = rng.normal(1.0, 0.3, n), rng.normal(0.0, 0.1, n)
    for r in range(3):
        with open(build / f"nodes_{r}.txt", "w") as f:
            f.write(f"0 {1.0:.17e}\n")
    with open(build / "proj.txt", "w") as f:
        f.write(f"{n}\n")
        for row in zip(vx, vy, bsin, bcos):
            f.write(" ".join(f"{x:.17e}" for x in row) + "\n")
    _run(build, 3)
    out = _floats(build / "proj_out.txt")
    ut_orig, ut_wall = out[:n], out[n:]
    nrm = np.stack([bcos, bsin], axis=1)  # wall normal of the case, (body_cos, body_sin)
    # hydrogym/nekrs/env.py transform_probes
    np.testing.assert_allclose(ut_orig, nrm[:, 1] * vx + nrm[:, 0] * vy, rtol=1e-14, atol=1e-15)  # hg_fortran
    tan = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    tan[tan[:, 0] < 0] *= -1  # downstream tangent
    np.testing.assert_allclose(ut_wall, tan[:, 0] * vx + tan[:, 1] * vy, rtol=1e-14, atol=1e-15)  # wall
