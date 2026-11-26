"""Time means of the wing forces in examples/nekrs/wing/fig4d.py: solver_steps.csv of several launches."""

import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "fig4d", os.path.join(HERE, "..", "examples", "nekrs", "wing", "fig4d.py")
)
fig4d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fig4d)

COLS = ["tstep", "time", "dt", "cfl"] + [
    f"{n}_b{b}" for b in (2, 1) for n in ("Fvx", "Fvy", "Fvz", "Fpx", "Fpy", "Fpz")
]


def _write(path, t, dt, fv1, fp1, fv2, fp2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for k in range(len(t)):
        r = dict.fromkeys(COLS, 0.0)
        r.update(tstep=k, time=t[k], dt=dt[k], Fvx_b1=fv1[k], Fpx_b1=fp1[k], Fvx_b2=fv2[k], Fpx_b2=fp2[k])
        rows.append(",".join(repr(float(r[c])) for c in COLS))
    with open(path, "w") as f:
        f.write(",".join(COLS) + "\n" + "\n".join(rows) + "\n")


def test_step_means_restart_and_dt_weights(tmp_path):
    run = str(tmp_path / "run")
    dt = np.r_[np.full(10, 1e-2), np.full(10, 2e-2)]
    t = 1.0 + np.cumsum(dt)
    fv = np.arange(20.0)
    # first launch: a wrong continuation after step 15 (replaced by the restart of the second launch)
    bad = fv.copy()
    bad[15:] = -100.0
    _write(f"{run}/log000000/solver_steps.csv", t, dt, bad, 0 * t + 2.0, 0 * t + 1.0, 0 * t)
    _write(
        f"{run}/log000015/solver_steps.csv", t[14:], dt[14:], fv[14:], 0 * t[14:] + 2.0, 0 * t[14:] + 1.0, 0 * t[14:]
    )
    st = fig4d.load_steps([run], (1, 2))
    assert len(st["t"]) == 20
    np.testing.assert_allclose(st["fv"], fv + 1.0)  # b1 + b2, the later launch wins
    cdf, cdp, lo, hi = fig4d.step_mean(st, t[0] - dt[0], t[-1], h=0.5)
    assert cdf == pytest.approx(((fv + 1.0) * dt).sum() / dt.sum() / 0.5)
    assert cdp == pytest.approx(2.0 / 0.5)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(t[-1])


def test_missing_force_columns(tmp_path):
    run = str(tmp_path / "run")
    os.makedirs(f"{run}/log000000")
    with open(f"{run}/log000000/solver_steps.csv", "w") as f:
        f.write("tstep,time,dt,cfl,dudn_b2\n0,1.0,0.01,0.3,5.0\n")
    assert fig4d.load_steps([run], (1, 2)) is None
    assert fig4d.load_steps([str(tmp_path / "none")], (1, 2)) is None


def _geometry(run, nst=8):
    """Two boundaries, suction side (b1, y > 0) and pressure side (b2), nst stations each along x."""
    x = np.linspace(0.1, 0.9, nst)
    g = {}
    for b, y in ((1, 0.05), (2, -0.05)):
        g.update({f"x_b{b}": x, f"y_b{b}": np.full(nst, y), f"z_b{b}": np.zeros(nst)})
        g[f"wsum_b{b}"] = np.full(nst, 0.01)
        g.update({f"nx_b{b}": np.zeros(nst), f"ny_b{b}": np.full(nst, np.sign(y)), f"nz_b{b}": np.zeros(nst)})
        g[f"span_b{b}"] = 0.1
    os.makedirs(run, exist_ok=True)
    np.savez(os.path.join(run, "walls_geometry.npz"), **g)
    return nst


def test_load_walls_step_layout_and_old_layout(tmp_path):
    """Wall samples every N solver steps (with step, obs_action/obs_time, power) and end-of-action samples."""
    nst = 8
    runs = {}
    for name, layout in (("new", "steps"), ("old", "end")):
        run = str(tmp_path / name)
        _geometry(run, nst)
        t = 1.0 + 1e-3 * np.arange(1, 13)
        tau = 1.0 + 0.1 * np.arange(12)  # traction t_x of every station at each sample
        tr = np.zeros((12, nst, 4))
        tr[:, :, 0] = tau[:, None]
        F = np.zeros((12, 6))
        q = {"time": t, "action": np.repeat(np.arange(1, 5), 3), "b1": tr, "b2": tr, "F1": F, "F2": F}
        if layout == "steps":
            q.update(
                step=10 * np.arange(1, 13),
                obs=np.zeros((4, 3, 2), np.float16),
                obs_action=np.arange(1, 5),
                obs_time=t[2::3],
                power_action=np.arange(1, 5),
                power_time=t[2::3],
                power=np.zeros((4, 4)),
                power_cd=np.zeros(4),
            )
        else:
            q["action"] = np.arange(1, 13)
            q["obs"] = np.zeros((12, 3, 2), np.float16)
        per_sample = {"time", "action", "b1", "b2", "F1", "F2", "step"} | ({"obs"} if layout == "end" else set())
        # two chunks: the per-sample arrays split, the per-action arrays all in the first one
        first = {k: v[:6] if k in per_sample else v for k, v in q.items()}
        second = {k: v[6:] if k in per_sample else v[:0] for k, v in q.items()}
        np.savez(os.path.join(run, "walls_000000_000006.npz"), **first)
        np.savez(os.path.join(run, "walls_000006_000012.npz"), **second)
        wing = fig4d.Wing(run)
        runs[name] = fig4d.load_walls([run], wing)
    for d in runs.values():
        assert len(d["t"]) == 12
        assert d["tau"].mean(0) == pytest.approx(np.full(2 * nst, 1.55), rel=1e-6)
    assert np.array_equal(runs["new"]["t"], runs["old"]["t"])
