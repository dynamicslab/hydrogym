"""
Boundary-layer characterization of deploy.py wing runs (uncontrolled runs): report.md with the
tables of all quantities, summary.json, CSV files per run and figures, from

  - the wall samples walls_*.npz (span-averaged traction and pressure per station, forces; every --wall-steps
    solver steps of deploy.py, or every --walls-every actions in runs of --wall-steps 0; probe observations):
    forces, Cp, Cf, their time statistics and spectra;
  - field_stats.npz of bl_stats.py (if present): profiles, Reynolds stresses, boundary-layer integrals,
    higher moments, budgets, spanwise / streamwise spectra and correlations, pointwise wall statistics, wake;
  - the case files (naca_wing.par / .udf, trip_origin.txt, state.json): trip, numerics, snapshots.

    python bl_report.py runs/zero_trip60 runs/zero_ref1 --skip 500 2420 --out bl_report

Conventions: U_inf = 1, c = 1, rho = 1, nu = 1/200000; wall frame s (downstream tangent), n (wall normal into
the fluid), z (span); upper side y > 0; Cf = tau_s / 0.5, Cp = 1 + (p - p_stag) / 0.5 with p_stag the stagnation
pressure of each sample; budgets as terms of dR_ij/dt (eps positive, listed as -eps); viscous units with
u_tau = sqrt(|tau_w|) of the field statistics at the same station.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import signal, stats

np.seterr(invalid="ignore", divide="ignore")

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--labels", nargs="+")
p.add_argument(
    "--skip", type=int, nargs="+", default=[0], help="actions left out of the wall samples (one value or one per run)"
)
p.add_argument("--stats", nargs="+", help="field_stats.npz per run (default RUN/field_stats.npz)")
p.add_argument("--out", default="bl_report")
p.add_argument(
    "--profile-x",
    type=float,
    nargs="+",
    default=[0.2, 0.3, 0.5, 0.7, 0.9],
    help="x/c of the profile figures and tables",
)
p.add_argument("--spec-x", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9], help="x/c of the spectra")
p.add_argument("--kx-half", type=float, default=0.1, help="half length (c) of the curve segments of Euu(kx)")
args = p.parse_args()
labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
skips = args.skip * len(args.runs) if len(args.skip) == 1 else args.skip
if len(skips) != len(args.runs):
    raise SystemExit("--skip: one value or one per run")
os.makedirs(args.out, exist_ok=True)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
NU = 1 / 200000
COMP = ["ss", "sn", "sz", "nn", "nz", "zz"]
TERMS = ["conv", "prod", "phi", "pdiff", "ttrans", "vdiff", "eps"]


def tfac(n):
    return float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan


def write_csv(path, cols):
    """cols: dict name -> 1D array (same length)."""
    names = list(cols)
    data = np.column_stack([np.asarray(cols[k], dtype=float) for k in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="", fmt="%.6g")


def integral_time(x, dt):
    """Integral time scale(s) of x (T, ...) along axis 0: the autocorrelation integrated to its first zero."""
    x = x - x.mean(0)
    n = len(x)
    f = np.fft.rfft(x, 2 * n, axis=0)
    acf = np.fft.irfft(np.abs(f) ** 2, axis=0)[:n]
    acf = acf / np.where(acf[0] > 0, acf[0], np.nan) / (1 - np.arange(n) / n).reshape((-1,) + (1,) * (x.ndim - 1))
    neg = np.cumsum(acf <= 0, axis=0) > 0
    return dt * np.nansum(np.where(neg, 0, acf), axis=0) - 0.5 * dt


def ts_stats(x, dt):
    """Mean, rms, integral time, 95% interval of the mean (effective samples T / 2 tau), drift of the halves."""
    tau = max(float(integral_time(x, dt)), dt)
    T = len(x) * dt
    neff = max(T / (2 * tau), 1.0)
    h = len(x) // 2
    return {
        "mean": float(x.mean()),
        "rms": float(x.std()),
        "tau_int": tau,
        "T": T,
        "n_eff": neff,
        "ci95": float(1.96 * x.std() / np.sqrt(neff)),
        "half1": float(x[:h].mean()),
        "half2": float(x[h:].mean()),
    }


# ------------------------------------------------------------------------------------------------ case files
def case_info(run):
    info = {}
    par = open(os.path.join(run, "naca_wing.par")).read() if os.path.exists(os.path.join(run, "naca_wing.par")) else ""
    udf = open(os.path.join(run, "naca_wing.udf")).read() if os.path.exists(os.path.join(run, "naca_wing.udf")) else ""
    for k in (
        "polynomialOrder",
        "cubaturePolynomialOrder",
        "dt",
        "timeStepper",
        "viscosity",
        "tripModes",
        "tripInterval",
        "regularization",
    ):
        m = re.search(rf"^\s*{k}\s*=\s*([^#\n]+)", par, re.M)
        if m:
            info[k] = m.group(1).strip()
    trip = {}
    for k in ("tripX", "tripY"):
        m = re.search(rf"{k}\[nWalls\]\s*=\s*\{{([^}}]+)\}}", udf)
        if m:
            trip[k] = [float(v) for v in m.group(1).split(",")]
    for k in ("radX", "radY", "tamps", "tampt", "tdt"):
        m = re.search(rf"\b{k}\s*=\s*([-+0-9.eE]+)", udf)
        if m:
            trip[k] = float(m.group(1))
    m = re.search(r"idum\s*=\s*(-?\d+);", udf)
    trip["ran2_seed"] = int(m.group(1)) if m else None
    trip["modes"] = int(info.get("tripModes", 30))
    if "tripInterval" in info:  # [CASEDATA] tripInterval overrides the default tdt of the udf
        trip["tdt"] = float(info["tripInterval"])
    f = os.path.join(run, "trip_origin.txt")
    if os.path.exists(f):
        trip["time_origin"] = float(open(f).read().split()[0])
    M = trip["modes"]
    trip["signal_rms"] = trip.get("tampt", np.nan) * np.sqrt(M / 2)  # sum of M unit sines with random phases
    trip["shortest_wavelength_c"] = 0.1 / M
    trip["gaussian_area_c2"] = np.pi * trip.get("radX", np.nan) * trip.get("radY", np.nan)
    info["trip"] = trip
    info["state"] = json.load(open(os.path.join(run, "state.json")))
    return info


# ------------------------------------------------------------------------------------------------ wall samples
def load_walls(run, skip):
    files = sorted(glob.glob(os.path.join(run, "walls_*_*.npz")))
    geo = dict(np.load(os.path.join(run, "walls_geometry.npz")))
    parts = [dict(np.load(f)) for f in files]
    parts = [q for q in parts if len(q["action"])]
    # per-sample arrays only (the observations and the power of every action have their own length)
    keys = set.intersection(*(set(q) for q in parts)) - {"obs", "obs_action", "obs_time"}
    keys = {k for k in keys if all(np.ndim(q[k]) and len(q[k]) == len(q["time"]) for q in parts)}
    d = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    obs_parts = [q for q in parts if "obs" in q]
    keep = d["action"] > skip
    d = {k: v[keep] for k, v in d.items()}
    if obs_parts:
        oa = np.concatenate([q.get("obs_action", q["action"]) for q in obs_parts])
        ob = np.concatenate([q["obs"] for q in obs_parts])
        d["obs_action"], d["obs"] = oa[oa > skip], ob[oa > skip].astype(np.float32)
    bids = [b for b in (1, 2) if f"x_b{b}" in geo and f"b{b}" in d]
    w = {
        "x": np.concatenate([geo[f"x_b{b}"] for b in bids]),
        "y": np.concatenate([geo[f"y_b{b}"] for b in bids]),
        "ws": np.concatenate([geo[f"wsum_b{b}"] for b in bids]),
        "Lz": float(geo[f"span_b{bids[0]}"]),
    }
    n = np.stack([np.concatenate([geo[f"nx_b{b}"] for b in bids]), np.concatenate([geo[f"ny_b{b}"] for b in bids])], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.stack([n[:, 1], -n[:, 0]], 1)
    t *= np.sign(t[:, 0] + 1e-300)[:, None]
    w["n"], w["t"] = n, t
    w["side"] = np.where(w["y"] >= 0, 1, -1)
    tr = np.concatenate([d[f"b{b}"] for b in bids], axis=1)  # (samples, stations, [tx, ty, tz, p])
    w["tau_s"] = tr[..., 0] * t[:, 0] + tr[..., 1] * t[:, 1]
    w["tau_z"] = tr[..., 2]
    w["p"] = tr[..., 3]
    w["trac"] = tr[..., :2]
    w["time"], w["action"] = d["time"], d["action"]
    w["F"] = sum(d[f"F{b}"] for b in bids)
    w["dt"] = float(np.median(np.diff(d["time"]))) if len(d["time"]) > 1 else np.nan
    if "obs" in d:
        w["obs"], w["obs_action"] = d["obs"], d["obs_action"]
        af = os.path.join(run, "agents.npz")
        if os.path.exists(af):
            w["agents"] = dict(np.load(af))
    return w


def side_sorted(w, side):
    k = np.flatnonzero(w["side"] == side)
    return k[np.argsort(w["x"][k])]


def forces(w):
    Lz, F = w["Lz"], w["F"]
    q = 0.5 * Lz
    ts = {
        "CD": (F[:, 0] + F[:, 3]) / q,
        "CD_f": F[:, 0] / q,
        "CD_p": F[:, 3] / q,
        "CL": (F[:, 1] + F[:, 4]) / q,
        "CL_f": F[:, 1] / q,
        "CL_p": F[:, 4] / q,
    }
    # pitching moment about c/4 (nose up positive) and a station-sum check of the drag
    pr = w["p"] - w["p_stag"][:, None] + 0.5  # constant offsets integrate to zero on the closed section
    f = w["ws"][None, :, None] * (w["trac"] - pr[..., None] * w["n"][None])
    mz = ((w["x"] - 0.25)[None] * f[..., 1] - w["y"][None] * f[..., 0]).sum(1)
    ts["CM_c4"] = -mz / q
    ts["CD_stations"] = f[..., 0].sum(1) / q
    return ts


# ------------------------------------------------------------------------------------------------ z resampling
def z_resampler(zs, Lz, N):
    """(M, nz) matrix: periodic GLL z-lines (element blocks of N intervals) -> M = nz uniform points, Lagrange
    interpolation within the z elements (linear if the levels are not element blocks)."""
    nz = len(zs)
    zu = np.arange(nz) * Lz / nz
    Wm = np.zeros((nz, nz))
    if nz % N == 0:
        zb = np.r_[zs[::N], Lz]
        for r, z in enumerate(zu):
            e = min(np.searchsorted(zb, z, "right") - 1, len(zb) - 2)
            nodes = np.r_[zs[e * N : e * N + N], zb[e + 1]]
            for j in range(N + 1):
                lj = np.prod([(z - nodes[m]) / (nodes[j] - nodes[m]) for m in range(N + 1) if m != j])
                Wm[r, (e * N + j) % nz] += lj
    else:
        zz = np.r_[zs, Lz]
        for r, z in enumerate(zu):
            j = min(np.searchsorted(zz, z, "right") - 1, nz - 1)
            a = (z - zz[j]) / (zz[j + 1] - zz[j])
            Wm[r, j] += 1 - a
            Wm[r, (j + 1) % nz] += a
    return Wm


def zspec(u, Lz):
    """u (..., M) uniform periodic -> kz, one-sided E(kz) (sum E dk = variance), R(dz) normalized, dz."""
    M = u.shape[-1]
    up = u - u.mean(-1, keepdims=True)
    F = np.fft.rfft(up, axis=-1) / M
    E = np.abs(F) ** 2
    E[..., 1:] *= 2
    if M % 2 == 0:
        E[..., -1] /= 2
    dk = 2 * np.pi / Lz
    c = np.fft.irfft(np.abs(np.fft.rfft(up, axis=-1)) ** 2, n=M, axis=-1) / M
    flat = lambda a: a.reshape(-1, a.shape[-1]).mean(0)
    Em, cm = flat(E), flat(c)
    return np.arange(len(Em)) * dk, Em / dk, cm[: M // 2 + 1] / cm[0], np.arange(M // 2 + 1) * Lz / M


# ------------------------------------------------------------------------------------------------ field stats
def bl_integrals(n, Us, Un, P, p0):
    """delta99 (U_s / U_I = 0.99, U_I = sqrt(2 (p0 - P) - U_n^2) of Griffin et al.), Ue, delta*, theta."""
    UI = np.sqrt(np.maximum(2 * (p0 - P) - Un**2, 0))
    r = Us / np.where(UI > 0, UI, np.nan)
    j = np.flatnonzero((r >= 0.99) & (np.arange(len(n)) > 0))
    out = dict(d99=np.nan, Ue=np.nan, dstar=np.nan, theta=np.nan, H=np.nan, j=None)
    if not len(j) or not np.isfinite(r[: j[0] + 1]).all():
        return out
    j = j[0]
    a = (0.99 - r[j - 1]) / (r[j] - r[j - 1])
    d99 = n[j - 1] + a * (n[j] - n[j - 1])
    Ue = UI[j - 1] + a * (UI[j] - UI[j - 1])
    nn = np.r_[n[:j], d99]
    uu = np.r_[Us[:j], Us[j - 1] + a * (Us[j] - Us[j - 1])] / Ue
    ds, th = np.trapezoid(1 - uu, nn), np.trapezoid(uu * (1 - uu), nn)
    out.update(d99=d99, Ue=Ue, dstar=ds, theta=th, H=ds / th, j=j)
    return out


def interp_side(w, side, q, x0):
    k = side_sorted(w, side)
    return np.interp(x0, w["x"][k], q[k])


def analyse_fields(fs, w, out_dir, res):
    meta = json.loads(str(fs["meta"]))
    p0 = meta["p0_freestream"]
    nvals = fs["nvals"]
    kind, lx, lside = fs["line_kind"], fs["line_x"], fs["line_side"]
    line_of = fs["line_of"]
    starts = np.r_[0, np.cumsum(np.bincount(line_of, minlength=len(kind)))]
    L = {k[2:]: fs[k] for k in fs.files if k.startswith("L_")}
    pf = fs["per_field"]
    nf = pf.shape[0]
    tf = tfac(nf)
    Lz, N = meta["Lz"], meta["N"]
    zs = fs["zs"]
    res["fields"] = {
        "n": nf,
        "files": meta["files"],
        "t": [min(meta["times"]), max(meta["times"])],
        "window_cU": max(meta["times"]) - min(meta["times"]) + (meta["times"][1] - meta["times"][0] if nf > 1 else 0),
        "p0_freestream": p0,
        "p_stag": meta["p_stag"],
        "Lz": Lz,
        "nz": meta["nz"],
        "N": N,
        "elements": meta["elements"],
    }
    # ---------------------------------------------------------------- per wall line: profiles, integrals
    walls = np.flatnonzero(kind == "wall")
    rows, prof = (
        [],
        {
            k: []
            for k in ["side", "x", "n", "n+", "U_s", "U_n", "W", "U+", "P", "Cp_p0"]
            + [f"R_{c}" for c in COMP]
            + [f"R+_{c}" for c in COMP]
            + ["k", "k+", "u_rms+", "v_rms+", "w_rms+", "S_u", "S_v", "S_w", "F_u", "F_v", "F_w"]
            + [f"b_{c}" for c in COMP]
            + [
                "eta",
                "xi",
                "p_rms",
                "p_rms+",
                "pu_s",
                "pu_n",
                "pu_z",
                "U_s_ci95",
                "R_ss_ci95",
                "R_nn_ci95",
                "R_zz_ci95",
                "R_sn_ci95",
            ]
        },
    )
    bud = {
        k: []
        for k in ["side", "x", "n", "n+"]
        + [f"{c}_{t}" for c in ["k", "ss", "nn", "zz", "sn"] for t in TERMS + ["residual"]]
    }
    for i in walls:
        sl = slice(starts[i], starts[i + 1])
        U, P, R = L["U"][sl], L["P"][sl], L["R"][sl]
        dUsdn = L["dU"][sl][0, 1]
        tw = NU * dUsdn
        ut = np.sqrt(abs(tw))
        bi = bl_integrals(nvals, U[:, 0], U[:, 1], P, p0)
        # per field (confidence intervals)
        per = [bl_integrals(nvals, pf[f, sl, 0], pf[f, sl, 1], pf[f, sl, 3], p0) for f in range(nf)]
        ci = {
            k: tf * np.nanstd([q[k] for q in per], ddof=1) / np.sqrt(nf) if nf > 1 else np.nan
            for k in ("d99", "dstar", "theta", "H", "Ue")
        }
        dPds = np.nan
        if bi["j"] is not None:
            dPds = np.interp(bi["d99"], nvals, L["dP"][sl][:, 0])
        k_ = 0.5 * (R[:, 0] + R[:, 3] + R[:, 5])
        kmax_j = int(np.nanargmax(k_[1:])) + 1 if np.isfinite(k_[1:]).any() else 0
        cfw = interp_side(w, lside[i], w["Cf_mean"], lx[i]) if w is not None else np.nan
        dn1 = interp_side_f(fs, lside[i], fs["wall_dn1"], lx[i])
        ds = interp_side_f(fs, lside[i], fs["wall_ds"], lx[i])
        dz = np.diff(np.r_[zs, Lz])
        ut_w = np.sqrt(abs(cfw) / 2) if np.isfinite(cfw) else np.nan
        cfrms = interp_side_f(fs, lside[i], res["_cf_rms_point"], lx[i])
        rows.append(
            dict(
                side=lside[i],
                x=lx[i],
                Cf_field=tw / 0.5,
                Cf_walls=cfw,
                utau=ut,
                utau_walls=ut_w,
                d99=bi["d99"],
                dstar=bi["dstar"],
                theta=bi["theta"],
                H=bi["H"],
                Ue=bi["Ue"],
                Re_theta=bi["Ue"] * bi["theta"] / NU,
                Re_dstar=bi["Ue"] * bi["dstar"] / NU,
                Re_tau=ut * bi["d99"] / NU,
                beta_clauser=bi["dstar"] / tw * dPds if tw else np.nan,
                dPe_ds=dPds,
                k_max=k_[kmax_j],
                n_kmax=nvals[kmax_j],
                n_kmax_plus=nvals[kmax_j] * ut / NU,
                urms_max_plus=np.sqrt(np.nanmax(R[1:, 0])) / ut,
                Cf_rms_point=cfrms,
                dn1_plus=dn1 * ut / NU,
                ds_plus=ds * ut / NU,
                dz_plus_max=dz.max() * ut / NU,
                dz_plus_mean=dz.mean() * ut / NU,
                dt_plus=res["dt_solver"] * ut**2 / NU,
                d99_ci95=ci["d99"],
                dstar_ci95=ci["dstar"],
                theta_ci95=ci["theta"],
                H_ci95=ci["H"],
                Ue_ci95=ci["Ue"],
            )
        )
        # profiles
        npl = nvals * ut / NU
        m3, m4 = L["m3"][sl], L["m4"][sl]
        diag = R[:, [0, 3, 5]]
        kk = np.where(k_ > 0, k_, np.nan)
        b = R / (2 * kk[:, None]) - np.array([1, 0, 0, 1, 0, 1]) / 3
        bf = np.stack([b[:, [0, 1, 2]], b[:, [1, 3, 4]], b[:, [2, 4, 5]]], 1)
        II = np.einsum("pij,pij->p", bf, bf)
        III = np.einsum("pij,pjk,pki->p", bf, bf, bf)
        ciU = tf * pf[:, sl, 0].std(0, ddof=1) / np.sqrt(nf) if nf > 1 else np.full(len(nvals), np.nan)
        ciR = [
            tf * pf[:, sl, 4 + c].std(0, ddof=1) / np.sqrt(nf) if nf > 1 else np.full(len(nvals), np.nan)
            for c in (0, 3, 5, 1)
        ]
        vals = dict(
            side=np.full(len(nvals), lside[i]),
            x=np.full(len(nvals), lx[i]),
            n=nvals,
            **{"n+": npl},
            U_s=U[:, 0],
            U_n=U[:, 1],
            W=U[:, 2],
            **{"U+": U[:, 0] / ut},
            P=P,
            Cp_p0=(P - (p0 - 0.5)) / 0.5,
            k=k_,
            **{
                "k+": k_ / ut**2,
                "u_rms+": np.sqrt(diag[:, 0]) / ut,
                "v_rms+": np.sqrt(diag[:, 1]) / ut,
                "w_rms+": np.sqrt(diag[:, 2]) / ut,
                "p_rms+": np.sqrt(L["pp"][sl]) / ut**2,
            },
            S_u=m3[:, 0] / diag[:, 0] ** 1.5,
            S_v=m3[:, 1] / diag[:, 1] ** 1.5,
            S_w=m3[:, 2] / diag[:, 2] ** 1.5,
            F_u=m4[:, 0] / diag[:, 0] ** 2,
            F_v=m4[:, 1] / diag[:, 1] ** 2,
            F_w=m4[:, 2] / diag[:, 2] ** 2,
            eta=np.sqrt(II / 6),
            xi=np.cbrt(III / 6),
            p_rms=np.sqrt(L["pp"][sl]),
            pu_s=L["pu"][sl][:, 0],
            pu_n=L["pu"][sl][:, 1],
            pu_z=L["pu"][sl][:, 2],
            U_s_ci95=ciU,
            R_ss_ci95=ciR[0],
            R_nn_ci95=ciR[1],
            R_zz_ci95=ciR[2],
            R_sn_ci95=ciR[3],
        )
        for c_, nm in enumerate(COMP):
            vals[f"R_{nm}"] = R[:, c_]
            vals[f"R+_{nm}"] = R[:, c_] / ut**2
            vals[f"b_{nm}"] = b[:, c_]
        for k in prof:
            prof[k].append(vals[k])
        # budgets in viscous units
        sc = NU / ut**4
        bud["side"].append(vals["side"]), bud["x"].append(vals["x"]), bud["n"].append(nvals), bud["n+"].append(npl)
        for cn, sel in (("k", None), ("ss", 0), ("nn", 3), ("zz", 5), ("sn", 1)):
            tot = 0
            for t in TERMS:
                v = L["budget_" + t][sl]
                v = 0.5 * (v[:, 0] + v[:, 3] + v[:, 5]) if sel is None else v[:, sel]
                v = -v if t == "eps" else v
                tot = tot + v
                bud[f"{cn}_{t}"].append(v * sc)
            bud[f"{cn}_residual"].append(tot * sc)
    write_csv(os.path.join(out_dir, "profiles.csv"), {k: np.concatenate(v) for k, v in prof.items()})
    write_csv(os.path.join(out_dir, "budgets_plus.csv"), {k: np.concatenate(v) for k, v in bud.items()})
    blx = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    write_csv(os.path.join(out_dir, "bl_vs_x.csv"), blx)
    res["_blx"], res["_prof"], res["_bud"] = (
        blx,
        {k: np.concatenate(v) for k, v in prof.items()},
        {k: np.concatenate(v) for k, v in bud.items()},
    )

    # ---------------------------------------------------------------- spanwise spectra and correlations
    Wz = z_resampler(zs, Lz, N)
    spec_pts = fs["spec_pts"]
    spec = fs["spec"]  # (fields, pts, z, 3) x, y, z
    pos = {int(q): k for k, q in enumerate(spec_pts)}
    kz_rows = {
        k: []
        for k in [
            "side",
            "x",
            "target",
            "n",
            "n+",
            "kz",
            "kz+",
            "lambda_z+",
            "E_uu",
            "E_vv",
            "E_ww",
            "kzE_uu+",
            "kzE_vv+",
            "kzE_ww+",
        ]
    }
    rz_rows = {k: [] for k in ["side", "x", "target", "n", "n+", "dz", "dz+", "R_uu", "R_vv", "R_ww"]}
    kz_sum = []
    for i in walls:
        if starts[i] + 1 not in pos:
            continue
        r = next(r for r in rows if r["side"] == lside[i] and abs(r["x"] - lx[i]) < 1e-9)
        ut = r["utau"]
        targets = [("n+=15", 15 * NU / ut)] + [
            (f"n/d99={f}", f * r["d99"]) for f in (0.1, 0.3, 0.6, 0.9) if np.isfinite(r["d99"])
        ]
        tv, nv = fs["line_t"][i], fs["line_n"][i]
        for tname, nt in targets:
            j = int(np.argmin(np.abs(nvals[1:] - nt))) + 1
            q = starts[i] + j
            if q not in pos:
                continue
            u = spec[:, pos[q]].astype(float)  # (fields, z, 3)
            us = u[..., 0] * tv[0] + u[..., 1] * tv[1]
            un = u[..., 0] * nv[0] + u[..., 1] * nv[1]
            comps = [zspec(np.einsum("mz,fz->fm", Wz, a), Lz) for a in (us, un, u[..., 2])]
            kz, rz = comps[0][0], comps[0][3]
            m = kz > 0
            for k, v in dict(side=lside[i], x=lx[i], target=tname, n=nvals[j], **{"n+": nvals[j] * ut / NU}).items():
                kz_rows[k].append(np.full(m.sum(), v, dtype=object))
                rz_rows[k].append(np.full(len(rz), v, dtype=object))
            (
                kz_rows["kz"].append(kz[m]),
                kz_rows["kz+"].append(kz[m] * NU / ut),
                kz_rows["lambda_z+"].append(2 * np.pi / kz[m] * ut / NU),
            )
            for c_, nm in enumerate(("uu", "vv", "ww")):
                kz_rows[f"E_{nm}"].append(comps[c_][1][m])
                kz_rows[f"kzE_{nm}+"].append(kz[m] * comps[c_][1][m] / ut**2)
                rz_rows[f"R_{nm}"].append(comps[c_][2])
            rz_rows["dz"].append(rz), rz_rows["dz+"].append(rz * ut / NU)
            if tname == "n+=15":
                Euu = kz[m] * comps[0][1][m]
                kz_sum.append(
                    dict(
                        side=int(lside[i]),
                        x=float(lx[i]),
                        n_plus=float(nvals[j] * ut / NU),
                        lambda_z_plus_peak_uu=float(2 * np.pi / kz[m][np.argmax(Euu)] * ut / NU),
                        Lz_plus=float(Lz * ut / NU),
                    )
                )
    tcode = {}
    if kz_rows["kz"]:
        for d_, fn in ((kz_rows, "spectra_kz.csv"), (rz_rows, "correlations_dz.csv")):
            cols = {k: np.concatenate(v) for k, v in d_.items()}
            names = sorted(set(cols["target"]))
            tcode = {nm: c for c, nm in enumerate(names)}
            cols["target"] = np.array([tcode[t] for t in cols["target"]])
            write_csv(os.path.join(out_dir, fn), cols)
        res["_kz"] = {k: np.concatenate(v) for k, v in kz_rows.items()}
    res["target_codes"] = tcode
    res["kz_n15"] = kz_sum

    # ---------------------------------------------------------------- streamwise spectra and correlations
    cx, cn, cp, ct, cvalid = fs["curve_x"], fs["curve_n"], fs["curve_pts"], fs["curve_t"], fs["curve_valid"]
    curve = fs["curve"]  # (fields, side x n x x, z, 3)
    nx = len(cx)
    kx_rows = {k: [] for k in ["side", "x0", "n", "n+", "kx", "kx+", "lambda_x+", "E_uu", "kxE_uu+"]}
    rx_rows = {k: [] for k in ["side", "x0", "n", "n+", "ds", "ds+", "R_uu"]}
    for si, side in enumerate((1, -1)):
        for ni, nc in enumerate(cn):
            base = (si * len(cn) + ni) * nx
            for x0 in args.spec_x:
                k = np.flatnonzero(np.abs(cx - x0) <= args.kx_half + 1e-9) + base
                if len(k) < 16 or not cvalid[k].all():
                    continue
                r = min((r for r in rows if r["side"] == side), key=lambda r: abs(r["x"] - x0))
                ut = r["utau"]
                u = curve[:, k].astype(float)  # (fields, pts, z, 3)
                us = u[..., 0] * ct[k, 0][None, :, None] + u[..., 1] * ct[k, 1][None, :, None]
                s = np.r_[0, np.cumsum(np.hypot(*np.diff(cp[k], axis=0).T))]
                su = np.linspace(0, s[-1], len(s))
                us = np.stack(
                    [[np.interp(su, s, us[f, :, z]) for z in range(us.shape[2])] for f in range(us.shape[0])]
                )  # (f, z, s)
                up = us - us.mean((0, 1))
                ds = su[1] - su[0]
                fq, Pxx = signal.periodogram(up, fs=1 / ds, window="hann", detrend=False, axis=-1)
                E = Pxx.reshape(-1, Pxx.shape[-1]).mean(0) / (2 * np.pi)
                kx = 2 * np.pi * fq
                m = kx > 0
                for kk_, v in dict(side=side, x0=x0, n=nc, **{"n+": nc * ut / NU}).items():
                    kx_rows[kk_].append(np.full(m.sum(), v))
                (
                    kx_rows["kx"].append(kx[m]),
                    kx_rows["kx+"].append(kx[m] * NU / ut),
                    kx_rows["lambda_x+"].append(2 * np.pi / kx[m] * ut / NU),
                )
                kx_rows["E_uu"].append(E[m]), kx_rows["kxE_uu+"].append(kx[m] * E[m] / ut**2)
                ic = len(su) // 2
                var = (up**2).mean((0, 1))
                Ruu = (up[..., ic : ic + 1] * up).mean((0, 1)) / np.sqrt(var[ic] * var)
                for kk_, v in dict(side=side, x0=x0, n=nc, **{"n+": nc * ut / NU}).items():
                    rx_rows[kk_].append(np.full(len(su), v))
                (
                    rx_rows["ds"].append(su - su[ic]),
                    rx_rows["ds+"].append((su - su[ic]) * ut / NU),
                    rx_rows["R_uu"].append(Ruu),
                )
    if kx_rows["kx"]:
        write_csv(os.path.join(out_dir, "spectra_kx.csv"), {k: np.concatenate(v) for k, v in kx_rows.items()})
        write_csv(os.path.join(out_dir, "correlations_ds.csv"), {k: np.concatenate(v) for k, v in rx_rows.items()})
        res["_kx"] = {k: np.concatenate(v) for k, v in kx_rows.items()}
        res["_rx"] = {k: np.concatenate(v) for k, v in rx_rows.items()}

    # ---------------------------------------------------------------- wall: spanwise spectra of tau and p
    wall = fs["wall"]
    wx, wy = fs["wall_x"], fs["wall_y"]
    ew = {k: [] for k in ["side", "x", "kz", "kz+", "E_tau_s", "E_tau_z", "E_p"]}
    for side in (1, -1):
        for x0 in np.arange(0.2, 0.91, 0.1):
            k = np.flatnonzero(np.sign(wy) == side)
            k = k[np.argmin(np.abs(wx[k] - x0))]
            r = min((r for r in rows if r["side"] == side), key=lambda r: abs(r["x"] - x0))
            a = wall[:, k].astype(float)  # (fields, z, 3)
            comps = [zspec(np.einsum("mz,fz->fm", Wz, a[..., c]), Lz) for c in range(3)]
            kz = comps[0][0]
            m = kz > 0
            for kk_, v in dict(side=side, x=wx[k]).items():
                ew[kk_].append(np.full(m.sum(), v))
            ew["kz"].append(kz[m]), ew["kz+"].append(kz[m] * NU / r["utau"])
            for c, nm in enumerate(("E_tau_s", "E_tau_z", "E_p")):
                ew[nm].append(comps[c][1][m])
    write_csv(os.path.join(out_dir, "wall_spectra_kz.csv"), {k: np.concatenate(v) for k, v in ew.items()})

    # ---------------------------------------------------------------- wake
    wake = np.flatnonzero(kind == "wake")
    wk = {k: [] for k in ["x", "y", "U", "V", "P", "R_xx", "R_xy", "R_yy", "R_zz", "k"]}
    res["wake"] = []
    pinf = p0 - 0.5
    for i in wake:
        sl = slice(starts[i], starts[i + 1])
        y = fs["wake_y"]
        U, P, R = L["U"][sl], L["P"][sl], L["R"][sl]
        for k, v in dict(
            x=np.full(len(y), lx[i]),
            y=y,
            U=U[:, 0],
            V=U[:, 1],
            P=P,
            R_xx=R[:, 0],
            R_xy=R[:, 1],
            R_yy=R[:, 3],
            R_zz=R[:, 5],
            k=0.5 * (R[:, 0] + R[:, 3] + R[:, 5]),
        ).items():
            wk[k].append(v)
        ok = np.isfinite(U[:, 0])
        y, u, Rxx, P = y[ok], U[ok, 0], R[ok, 0], P[ok]
        jm = int(np.argmin(u))
        defic = 1 - u[jm]
        half = 1 - 0.5 * defic
        try:
            yl = np.interp(half, u[jm::-1], y[jm::-1])
            yu = np.interp(half, u[jm:], y[jm:])
        except ValueError:
            yl = yu = np.nan
        res["wake"].append(
            dict(
                x=float(lx[i]),
                y_center=float(y[jm]),
                deficit=float(defic),
                half_width=float((yu - yl) / 2),
                theta_w=float(np.trapezoid(u * (1 - u), y)),
                dstar_w=float(np.trapezoid(1 - u, y)),
                CD_wake=float(2 * np.trapezoid(u * (1 - u) - Rxx + (pinf - P), y)),
                k_max=float(np.nanmax(0.5 * (R[ok, 0] + R[ok, 3] + R[ok, 5]))),
            )
        )
    write_csv(os.path.join(out_dir, "wake_profiles.csv"), {k: np.concatenate(v) for k, v in wk.items()})


def interp_side_f(fs, side, q, x0):
    wx, wy = fs["wall_x"], fs["wall_y"]
    k = np.flatnonzero(np.sign(wy) == side)
    k = k[np.argsort(wx[k])]
    ok = np.isfinite(q[k])
    return float(np.interp(x0, wx[k][ok], q[k][ok])) if ok.any() else np.nan


# ------------------------------------------------------------------------------------------------ per run
summary, R_ = {}, {}
for run, lab, skip, sfile in zip(args.runs, labels, skips, args.stats or [None] * len(args.runs)):
    od = os.path.join(args.out, lab)
    os.makedirs(od, exist_ok=True)
    res = {"run": run, "skip_actions": skip}
    info = case_info(run)
    res["case"] = {k: v for k, v in info.items() if k != "state"}
    res["dt_solver"] = float(info.get("dt", 2e-5))
    w = load_walls(run, skip)
    w["p_stag"] = w["p"][:, w["x"] < 0.01].max(1)
    w["Cp"] = 1 + (w["p"] - w["p_stag"][:, None]) / 0.5
    w["Cf"] = w["tau_s"] / 0.5
    w["Cf_mean"] = w["Cf"].mean(0)
    dt = w["dt"]
    res["walls"] = {
        "samples": int(len(w["time"])),
        "t": [float(w["time"][0]), float(w["time"][-1])],
        "dt_sample": dt,
        "actions": [int(w["action"][0]), int(w["action"][-1])],
        "T": float(w["time"][-1] - w["time"][0] + dt),
    }
    # 2. forces
    ts = forces(w)
    res["forces"] = {k: ts_stats(v, dt) for k, v in ts.items()}
    fq, Pcd = signal.welch(ts["CD"] - ts["CD"].mean(), fs=1 / dt, nperseg=min(len(ts["CD"]), 256))
    _, Pcl = signal.welch(ts["CL"] - ts["CL"].mean(), fs=1 / dt, nperseg=min(len(ts["CL"]), 256))
    res["forces"]["St_peak_CL"] = float(fq[1:][np.argmax(Pcl[1:])]) if len(fq) > 1 else None
    write_csv(os.path.join(od, "forces_timeseries.csv"), {"time": w["time"], "action": w["action"], **ts})
    write_csv(os.path.join(od, "forces_psd.csv"), {"f": fq, "PSD_CD": Pcd, "PSD_CL": Pcl})
    # 3./4. Cp and Cf per station (time statistics of the span averages)
    tau_cf = integral_time(w["Cf"], dt)
    neff = np.maximum(res["walls"]["T"] / (2 * np.maximum(tau_cf, dt)), 1)
    h = len(w["time"]) // 2
    st = {
        "side": w["side"],
        "x": w["x"],
        "y": w["y"],
        "Cp": w["Cp"].mean(0),
        "Cp_rms_spanavg": w["Cp"].std(0),
        "Cf": w["Cf_mean"],
        "Cf_rms_spanavg": w["Cf"].std(0),
        "Cf_ci95": 1.96 * w["Cf"].std(0) / np.sqrt(neff),
        "Cf_half1": w["Cf"][:h].mean(0),
        "Cf_half2": w["Cf"][h:].mean(0),
        "Cf_z": w["tau_z"].mean(0) / 0.5,
    }
    # 8. transition from the mean Cf of each side
    res["transition"] = {}
    for side, nm in ((1, "upper"), (-1, "lower")):
        k = side_sorted(w, side)
        x, cf = w["x"][k], w["Cf_mean"][k]
        # the trip itself gives a local Cf spike at x = 0.1; the transition proper is judged downstream of it
        mt = (x > 0.095) & (x < 0.125)
        cfs = np.array([cf[np.abs(x - x0) <= 0.015].mean() for x0 in x])  # smoothed over +-0.015c
        mm = (x >= 0.125) & (x < 0.8)
        xm = x[mm][np.argmin(cfs[mm])]  # laminar / relaxing minimum
        mpk = (x >= xm) & (x < 0.9)
        xp = x[mpk][np.argmax(cfs[mpk])]  # turbulent peak
        mmin = (x >= xm) & (x <= xp)
        cmin, cmax = cfs[mm].min(), cfs[mpk].max()
        thr = 0.5 * (cmin + cmax)
        up_ = mmin & (cfs >= thr)
        sep = x[(cf < 0) & (x > 0.01)]
        res["transition"][nm] = {
            "x_Cf_trip_peak": float(x[mt][np.argmax(cf[mt])]),
            "Cf_trip_peak": float(cf[mt].max()),
            "x_Cf_min": float(xm),
            "Cf_min_smoothed": float(cmin),
            "x_Cf_peak": float(xp),
            "Cf_peak_smoothed": float(cmax),
            "rise_after_trip": bool(xp > xm),
            "Cf_threshold": float(thr),
            "x_mean_Cf_crosses_threshold": float(x[up_].min()) if up_.any() else None,
            "Cf_negative_x": [float(sep.min()), float(sep.max())] if len(sep) else None,
        }
    # temporal PSD of the span-averaged wall shear
    tp = {k: [] for k in ["side", "x", "f", "PSD_tau_s_spanavg"]}
    for side in (1, -1):
        for x0 in np.arange(0.2, 0.91, 0.1):
            k = side_sorted(w, side)
            k = k[np.argmin(np.abs(w["x"][k] - x0))]
            f_, P_ = signal.welch(
                w["tau_s"][:, k] - w["tau_s"][:, k].mean(), fs=1 / dt, nperseg=min(len(w["time"]), 256)
            )
            for kk_, v in dict(
                side=np.full(len(f_), side), x=np.full(len(f_), w["x"][k]), f=f_, PSD_tau_s_spanavg=P_
            ).items():
                tp[kk_].append(v)
    write_csv(os.path.join(od, "wall_shear_psd_time.csv"), {k: np.concatenate(v) for k, v in tp.items()})
    # 10. temporal velocity PSD at the probes (observations: u_s', v_n' about the span mean, every wall sample)
    if "obs" in w and "agents" in w and len(w["obs"]) > 16:
        ag = w["agents"]
        ob = w["obs"] * ag["obs_div"][None]
        px = ag["probe_xyz"]
        dxy = np.hypot(px[:, 0, None] - w["x"][None], px[:, 1, None] - w["y"][None]).min(1)
        tv = {k: [] for k in ["x", "n", "f", "PSD_u_s", "PSD_v_n"]}
        for x0 in args.spec_x:
            k = np.flatnonzero(np.abs(px[:, 0] - px[np.argmin(np.abs(px[:, 0] - x0)), 0]) < 1e-6)
            f_, P_ = signal.welch(ob[:, k] - ob[:, k].mean(0), fs=1 / dt, nperseg=min(len(ob), 256), axis=0)
            P_ = P_.mean(1)
            for kk_, v in dict(
                x=np.full(len(f_), px[k[0], 0]),
                n=np.full(len(f_), dxy[k].mean()),
                f=f_,
                PSD_u_s=P_[:, 0],
                PSD_v_n=P_[:, 1],
            ).items():
                tv[kk_].append(v)
        write_csv(os.path.join(od, "velocity_psd_time.csv"), {k: np.concatenate(v) for k, v in tv.items()})
        res["probes"] = {
            "samples": int(len(ob)),
            "n_probes": int(len(px)),
            "wall_distance_c": [float(dxy.min()), float(dxy.max())],
        }
    # field statistics
    sfile = sfile or os.path.join(run, "field_stats.npz")
    res["field_stats_file"] = sfile if os.path.exists(sfile) else None
    if res["field_stats_file"]:
        fs = np.load(sfile, allow_pickle=False)
        wall = fs["wall"].astype(float)  # (fields, stations, z, [tau_s, tau_z, p])
        wx = fs["wall_x"]
        ps = wall[..., 2].mean(2)[:, wx < 0.01].max(1)  # stagnation pressure per field
        cfp = wall[..., 0] / 0.5
        res["_cf_rms_point"] = cfp.std((0, 2))
        cp_rms = ((wall[..., 2] - ps[:, None, None]) / 0.5).std((0, 2))
        # 8. intermittency of the pointwise Cf with the threshold of the mean Cf
        gam = np.full(len(wx), np.nan)
        for side, nm in ((1, "upper"), (-1, "lower")):
            k = np.flatnonzero(np.sign(fs["wall_y"]) == side)
            gam[k] = (cfp[:, k] > res["transition"][nm]["Cf_threshold"]).mean((0, 2))
            o = k[np.argsort(wx[k])]
            g, xs_ = gam[o], wx[o]
            for lev in (0.1, 0.5, 0.9):
                m = (xs_ >= res["transition"][nm]["x_Cf_min"]) & (xs_ < 0.8) & (g >= lev)
                res["transition"][nm][f"x_gamma_{lev}"] = float(xs_[m].min()) if m.any() else None
        # pointwise wall statistics at the walls npz stations (same stations, same order)
        if len(wx) == len(w["x"]) and np.allclose(wx, w["x"]):
            st.update(
                {
                    "Cf_field": cfp.mean((0, 2)),
                    "Cf_rms_point": res["_cf_rms_point"],
                    "Cf_z_rms_point": (wall[..., 1] / 0.5).std((0, 2)),
                    "Cp_rms_point": cp_rms,
                    "gamma": gam,
                    "Cf_skewness_point": stats.skew(cfp.transpose(1, 0, 2).reshape(len(wx), -1), axis=1),
                }
            )
        analyse_fields(fs, w, od, res)
        b = res["_blx"]
        for side, nm in ((1, "upper"), (-1, "lower")):
            m = (b["side"] == side) & (b["x"] >= 0.125)
            for lev in (2.2, 1.7):
                q = b["x"][m][b["H"][m] < lev]
                res["transition"][nm][f"x_H_below_{lev}"] = float(q.min()) if len(q) else None
    write_csv(os.path.join(od, "wall_stations.csv"), st)
    res["_st"], res["_w"], res["_ts"] = st, w, ts
    # 18. snapshots
    snaps = []
    for c in info["state"]["chunks"]:
        f = os.path.join(run, c["file"])
        meta_f = os.path.join(run, c["file"].replace("0.f00000", ".nek5000"))
        snaps.append(
            {
                **c,
                "exists": os.path.exists(f),
                "GB": os.path.getsize(f) / 1e9 if os.path.exists(f) else None,
                "paraview": os.path.basename(meta_f) if os.path.exists(meta_f) else None,
            }
        )
    res["snapshots"] = snaps
    R_[lab] = res
    summary[lab] = {k: v for k, v in res.items() if not k.startswith("_")}
    print(
        f"[BL] {lab}: {res['walls']['samples']} wall samples, fields: {res.get('fields', {}).get('n', 0)}", flush=True
    )


# ------------------------------------------------------------------------------------------------ figures
def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, name), dpi=130)
    plt.close(fig)


figs = []
if plt is not None:
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for lab, r in R_.items():
        t, ts = r["_w"]["time"], r["_ts"]
        for a, k in zip(ax, ("CD", "CL")):
            a.plot(t, ts[k], lw=0.6, label=lab)
            a.plot(t, np.cumsum(ts[k]) / np.arange(1, len(t) + 1), "--", lw=1, label=f"{lab} running mean")
            a.set_ylabel(k)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("t U/c")
    fig_save(fig, "fig_forces.png"), figs.append("fig_forces.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for lab, r in R_.items():
        st = r["_st"]
        for side, ls in ((1, "-"), (-1, ":")):
            k = np.flatnonzero(st["side"] == side)
            k = k[np.argsort(st["x"][k])]
            ax[0].plot(st["x"][k], st["Cp"][k], ls, label=f"{lab} {'upper' if side > 0 else 'lower'}")
            ax[1].plot(st["x"][k], st["Cf"][k], ls, label=f"{lab} {'upper' if side > 0 else 'lower'}")
            if "Cf_rms_point" in st:
                ax[2].plot(st["x"][k], st["Cf_rms_point"][k], ls, label=f"{lab} {'upper' if side > 0 else 'lower'}")
    ax[0].invert_yaxis(), ax[0].set_ylabel("Cp"), ax[1].set_ylabel("Cf"), ax[2].set_ylabel("Cf rms (pointwise, fields)")
    ax[1].axhline(0, color="k", lw=0.5)
    for a in ax:
        a.set_xlabel("x/c"), a.set_xlim(0, 1), a.axvline(0.1, color="0.6", lw=0.5)
    ax[1].legend(fontsize=7)
    fig_save(fig, "fig_cp_cf.png"), figs.append("fig_cp_cf.png")

    have = [lab for lab, r in R_.items() if "_blx" in r]
    if have:
        qs = [
            ("d99", "delta99"),
            ("theta", "theta"),
            ("H", "H"),
            ("Re_theta", "Re_theta"),
            ("utau", "u_tau"),
            ("beta_clauser", "Clauser beta"),
            ("urms_max_plus", "max u_rms+"),
            ("dn1_plus", "dn1+ / ds+ / dz+max"),
        ]
        fig, ax = plt.subplots(2, 4, figsize=(17, 7))
        for lab in have:
            b = R_[lab]["_blx"]
            for side, ls in ((1, "-"), (-1, ":")):
                m = b["side"] == side
                for a, (k, nm) in zip(ax.flat, qs):
                    a.plot(b["x"][m], b[k][m], ls, label=f"{lab} {'upper' if side > 0 else 'lower'}")
                    a.set_title(nm, fontsize=9)
                if side == 1:
                    ax.flat[-1].plot(b["x"][m], b["ds_plus"][m], "--", label=f"{lab} ds+")
                    ax.flat[-1].plot(b["x"][m], b["dz_plus_max"][m], "-.", label=f"{lab} dz+ max")
        ax.flat[2].set_ylim(1, 4), ax.flat[5].set_ylim(-2, 10)
        ax.flat[0].legend(fontsize=7), ax.flat[-1].legend(fontsize=6)
        for a in ax.flat:
            a.set_xlabel("x/c"), a.axvline(0.1, color="0.6", lw=0.5)
        fig_save(fig, "fig_bl_vs_x.png"), figs.append("fig_bl_vs_x.png")

        fig, ax = plt.subplots(2, len(args.profile_x), figsize=(4 * len(args.profile_x), 7), squeeze=False)
        for c_, x0 in enumerate(args.profile_x):
            for lab in have:
                pr = R_[lab]["_prof"]
                xs_ = np.unique(pr["x"][pr["side"] == 1])
                xn = xs_[np.argmin(np.abs(xs_ - x0))]
                m = (pr["side"] == 1) & (pr["x"] == xn) & (pr["n+"] > 0)
                (l_,) = ax[0, c_].semilogx(pr["n+"][m], pr["U+"][m], label=lab)
                for k, ls in (("u_rms+", "-"), ("v_rms+", "--"), ("w_rms+", ":")):
                    ax[1, c_].semilogx(pr["n+"][m], pr[k][m], ls, color=l_.get_color(), label=f"{lab} {k}")
                ax[1, c_].semilogx(pr["n+"][m], -pr["R+_sn"][m], "-.", color=l_.get_color(), label=f"{lab} -uv+")
            npl = np.geomspace(1, 1000)
            ax[0, c_].semilogx(npl, np.log(npl) / 0.41 + 5.2, "k:", lw=0.7)
            ax[0, c_].semilogx(npl[:20], npl[:20], "k:", lw=0.7)
            ax[0, c_].set_title(f"upper x/c = {x0}"), ax[0, c_].set_ylim(0, 30), ax[1, c_].set_xlabel("n+")
        (
            ax[0, 0].set_ylabel("U_s+"),
            ax[1, 0].set_ylabel("rms+, -uv+"),
            ax[0, 0].legend(fontsize=7),
            ax[1, 0].legend(fontsize=6),
        )
        fig_save(fig, "fig_profiles.png"), figs.append("fig_profiles.png")

        fig, ax = plt.subplots(1, len(args.profile_x), figsize=(4 * len(args.profile_x), 4), squeeze=False)
        for c_, x0 in enumerate(args.profile_x):
            for lab, ls in zip(have, ("-", "--", ":", "-.")):
                bd = R_[lab]["_bud"]
                xs_ = np.unique(bd["x"][bd["side"] == 1])
                xn = xs_[np.argmin(np.abs(xs_ - x0))]
                m = (bd["side"] == 1) & (bd["x"] == xn) & (bd["n+"] > 0.1)
                for t, col in zip(TERMS + ["residual"], plt.cm.tab10.colors):
                    ax[0, c_].semilogx(
                        bd["n+"][m],
                        bd[f"k_{t}"][m],
                        ls,
                        color=col,
                        label=f"{lab} {'-eps' if t == 'eps' else t}" if c_ == 0 else None,
                    )
            (
                ax[0, c_].set_title(f"k budget (+), upper x/c = {x0}", fontsize=9),
                ax[0, c_].set_xlim(0.5, 1000),
                ax[0, c_].set_xlabel("n+"),
            )
        ax[0, 0].legend(fontsize=5)
        fig_save(fig, "fig_budget_k.png"), figs.append("fig_budget_k.png")

        if any("_kz" in R_[lab] for lab in have):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            for lab in have:
                kz = R_[lab].get("_kz")
                if kz is None:
                    continue
                for x0 in args.spec_x:
                    m = (kz["side"] == 1) & (kz["target"] == "n+=15") & (np.abs(kz["x"].astype(float) - x0) < 0.006)
                    if m.any():
                        ax[0].semilogx(
                            kz["lambda_z+"][m].astype(float), kz["kzE_uu+"][m].astype(float), label=f"{lab} x={x0}"
                        )
                kx = R_[lab].get("_kx")
                if kx is not None:
                    for x0 in args.spec_x:
                        m = (kx["side"] == 1) & (kx["x0"] == x0) & (kx["n"] == kx["n"].min())
                        ax[1].semilogx(kx["lambda_x+"][m], kx["kxE_uu+"][m], label=f"{lab} x={x0}")
            ax[0].set_xlabel("lambda_z+"), ax[0].set_ylabel("kz Euu / u_tau^2 at n+ = 15 (upper)")
            ax[1].set_xlabel("lambda_x+"), ax[1].set_ylabel("kx Euu / u_tau^2 (upper, smallest curve n)")
            ax[0].legend(fontsize=6), ax[1].legend(fontsize=6)
            fig_save(fig, "fig_spectra.png"), figs.append("fig_spectra.png")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for lab in have:
            st = R_[lab]["_st"]
            if "gamma" in st:
                for side, ls in ((1, "-"), (-1, ":")):
                    k = np.flatnonzero(st["side"] == side)
                    k = k[np.argsort(st["x"][k])]
                    ax[0].plot(st["x"][k], st["gamma"][k], ls, label=f"{lab} {'upper' if side > 0 else 'lower'}")
            pr = R_[lab]["_prof"]
            xs_ = np.unique(pr["x"][pr["side"] == 1])
            for x0 in args.profile_x:
                xn = xs_[np.argmin(np.abs(xs_ - x0))]
                m = (pr["side"] == 1) & (pr["x"] == xn) & (pr["n+"] > 0.5)
                ax[1].plot(pr["xi"][m], pr["eta"][m], lw=0.8, label=f"{lab} x={x0}")
        xi = np.linspace(-1 / 6, 1 / 3, 100)
        ax[1].plot(xi, np.sqrt(1 / 27 + 2 * xi**3), "k", lw=0.7)
        ax[1].plot([0, -1 / 6], [0, 1 / 6], "k", lw=0.7), ax[1].plot([0, 1 / 3], [0, 1 / 3], "k", lw=0.7)
        (
            ax[0].set_xlabel("x/c"),
            ax[0].set_ylabel("intermittency gamma (pointwise Cf > threshold)"),
            ax[0].set_xlim(0, 1),
        )
        ax[1].set_xlabel("xi"), ax[1].set_ylabel("eta"), ax[1].set_title("Lumley triangle, upper side")
        ax[0].legend(fontsize=7), ax[1].legend(fontsize=5)
        fig_save(fig, "fig_transition_anisotropy.png"), figs.append("fig_transition_anisotropy.png")

        fig, ax = plt.subplots(1, len(have), figsize=(8 * len(have), 3.5), squeeze=False)
        for c_, lab in enumerate(have):
            fs = np.load(R_[lab]["field_stats_file"])
            gx, gy, gU, gR = fs["gx"], fs["gy"], fs["gU"], fs["gR"]
            m = (gx > -0.05) & (gx < 1.3) & (gy > -0.12) & (gy < 0.12)
            kk = 0.5 * (gR[m, 0] + gR[m, 3] + gR[m, 5])
            tc = ax[0, c_].tricontourf(gx[m], gy[m], np.sqrt(np.maximum(kk, 0)), 40, cmap="viridis")
            ax[0, c_].tricontour(gx[m], gy[m], gU[m, 0], [0], colors="w", linewidths=0.5)
            (
                fig.colorbar(tc, ax=ax[0, c_]),
                ax[0, c_].set_aspect("equal"),
                ax[0, c_].set_title(f"{lab}: sqrt(k), white U = 0"),
            )
        fig_save(fig, "fig_mean_field.png"), figs.append("fig_mean_field.png")


# ------------------------------------------------------------------------------------------------ report.md
def f3(v, spec=".4g"):
    try:
        return "-" if v is None or not np.isfinite(v) else format(v, spec)
    except TypeError:
        return str(v)


labs = list(R_)
md = [
    "# Boundary-layer statistics of the uncontrolled NACA0012 wing (NekRS, Re_c = 200k, AoA 0)",
    "",
    "Generated by `bl_report.py` from: "
    + ", ".join(
        f"`{R_[l]['run']}` (wall samples after {R_[l]['skip_actions']} actions, "
        f"field statistics {'`' + R_[l]['field_stats_file'] + '`' if R_[l]['field_stats_file'] else 'missing'})"
        for l in labs
    ),
    "",
    "CSV files per run in `<label>/`, figures `fig_*.png`, all numbers in `summary.json`.",
    "",
]


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


md += ["## 1. Tripping", ""]
rows = []
for k, nm in (
    ("modes", "spanwise modes M"),
    ("tripX", "x/c"),
    ("tripY", "y/c"),
    ("radX", "Gaussian radius x (c)"),
    ("radY", "Gaussian radius y (c)"),
    ("tamps", "steady amplitude tamps"),
    ("tampt", "unsteady amplitude tampt"),
    ("tdt", "time between random signals tdt (c/U)"),
    ("ran2_seed", "ran2 seed"),
    ("time_origin", "trip time origin"),
    ("signal_rms", "rms of the spanwise signal tampt sqrt(M/2)"),
    ("shortest_wavelength_c", "shortest wavelength Lz/M (c)"),
    ("gaussian_area_c2", "pi radX radY (c^2)"),
):
    rows.append(
        [nm]
        + [
            f3(R_[l]["case"]["trip"].get(k))
            if not isinstance(R_[l]["case"]["trip"].get(k), list)
            else str(R_[l]["case"]["trip"].get(k))
            for l in labs
        ]
    )
md += table(rows, ["quantity"] + labs)
md += [
    "Forcing: f_y = tampt [(1 - b) g_{i-1}(z) + b g_i(z)] exp(-((x - tripX)/radX)^2 - ((y - tripY)/radY)^2) on the y momentum, "
    "g_i(z) = sum_{m=1..M} sin(2 pi m z / Lz + phi_{m,i}) with random phases drawn every tdt, b = p^2 (3 - 2p) the smooth "
    "blend, both sides; where |f| >= 1e-10 it replaces the RT-filter term. The signal is not normalized by M: with 60 modes the "
    "forcing rms is sqrt(2) times that with 30 modes.",
    "",
]

md += ["## 2. Forces (span length Lz, q = 0.5 Lz)", ""]
rows = []
for k in ("CD", "CD_p", "CD_f", "CL", "CL_p", "CL_f", "CM_c4", "CD_stations"):
    rows.append(
        [k]
        + [
            f"{f3(R_[l]['forces'][k]['mean'], '.5f')} ± {f3(R_[l]['forces'][k]['ci95'], '.1e')} (rms {f3(R_[l]['forces'][k]['rms'], '.2e')}, "
            f"tau {f3(R_[l]['forces'][k]['tau_int'], '.3f')}, halves {f3(R_[l]['forces'][k]['half1'], '.5f')} / {f3(R_[l]['forces'][k]['half2'], '.5f')})"
            for l in labs
        ]
    )
rows.append(
    ["window T (c/U), samples"]
    + [f"{R_[l]['walls']['T']:.3f}, {R_[l]['walls']['samples']} (dt {R_[l]['walls']['dt_sample']:.4g})" for l in labs]
)
rows.append(["CL PSD peak f c/U"] + [f3(R_[l]["forces"].get("St_peak_CL")) for l in labs])
md += table(rows, ["coefficient: mean ± 95% (integral time)"] + labs)
md += [
    "CD_stations: drag from the station sums of the wall samples (check of the force integral). CM about c/4, nose up positive. "
    "Time series `forces_timeseries.csv`, Welch PSD `forces_psd.csv`.",
    "",
]

md += [
    "## 3./4. Surface pressure and wall friction",
    "",
    "`wall_stations.csv`: Cp, Cf (time and span mean, 95% interval, the halves of the window), rms over time of the span averages "
    "(`*_rms_spanavg`) and, from the fields, the pointwise rms over span and fields of Cf, Cf_z and Cp, the skewness of Cf, the intermittency. "
    "`wall_spectra_kz.csv`: spanwise spectra of tau_s, tau_z, p at x/c = 0.2..0.9; `wall_shear_psd_time.csv`: temporal PSD of the span-averaged tau_s "
    "(kz = 0 only).",
    "",
]
rows = []
for x0 in args.profile_x:
    row = [f"{x0}"]
    for l in labs:
        st = R_[l]["_st"]
        vals = []
        for side in (1, -1):
            k = np.flatnonzero(st["side"] == side)
            k = k[np.argmin(np.abs(st["x"][k] - x0))]
            vals.append(
                f"Cp {st['Cp'][k]:.3f}, Cf {st['Cf'][k]:.5f}±{st['Cf_ci95'][k]:.1e}"
                + (f", Cf_rms {st['Cf_rms_point'][k]:.5f}" if "Cf_rms_point" in st else "")
            )
        row.append(" / ".join(vals))
    rows.append(row)
md += table(rows, ["x/c (upper / lower)"] + labs)

if any("_blx" in R_[l] for l in labs):
    md += [
        "## 5.-7., 9., 12. Profiles, Reynolds stresses, boundary-layer parameters, higher moments",
        "",
        "`profiles.csv`: every station (x/c = 0.02..0.98 by 0.01, both sides) and n (240 points to 0.1c): U_s, U_n, W, P, all R_ij "
        "(outer and +), k, rms, skewness S, flatness F, anisotropy b_ij, Lumley eta / xi, p_rms, <p'u_i'>, 95% intervals from the "
        "spread of the fields. `bl_vs_x.csv`: delta99, delta*, theta, H, U_e, u_tau (fields and wall samples), Re_theta, Re_delta*, "
        "Re_tau, Clauser beta, max k and u_rms+, resolution in viscous units, with 95% intervals.",
        "",
    ]
    rows = []
    cols = [
        ("d99", ".4f"),
        ("dstar", ".5f"),
        ("theta", ".5f"),
        ("H", ".3f"),
        ("Ue", ".4f"),
        ("utau", ".4f"),
        ("Re_theta", ".0f"),
        ("Re_tau", ".0f"),
        ("beta_clauser", ".2f"),
        ("Cf_field", ".5f"),
        ("Cf_walls", ".5f"),
    ]
    for l in labs:
        b = R_[l].get("_blx")
        if b is None:
            continue
        for side in (1, -1):
            for x0 in args.profile_x:
                m = np.flatnonzero(b["side"] == side)
                i = m[np.argmin(np.abs(b["x"][m] - x0))]
                rows.append(
                    [l, "upper" if side > 0 else "lower", f"{b['x'][i]:.2f}"]
                    + [
                        f3(b[k][i], s) + (f" ± {f3(b[k + '_ci95'][i], '.1e')}" if k + "_ci95" in b else "")
                        for k, s in cols
                    ]
                )
    md += table(rows, ["run", "side", "x/c"] + [k for k, _ in cols])

md += [
    "## 8. Transition",
    "",
    "From the mean Cf of the wall samples: the local spike of the trip (x/c = 0.095..0.125), then downstream of x/c = 0.125 the Cf "
    "minimum (x < 0.8, laminar / relaxing state) and the peak after it (x < 0.9; Cf smoothed over +-0.015c; rise_after_trip false: "
    "no rise after the trip), the threshold halfway between; from the fields the intermittency gamma = fraction of the "
    "pointwise samples (span, fields) with Cf above the threshold (`wall_stations.csv`) and its first x downstream of the minimum with "
    "gamma >= 0.1 (onset), 0.5, 0.9 (completion); from the profiles the first x downstream of 0.125 with the shape factor H below "
    "2.2 and 1.7 (H(x) in `bl_vs_x.csv`).",
    "",
]
rows = []
for k in (
    "x_Cf_trip_peak",
    "Cf_trip_peak",
    "x_Cf_min",
    "Cf_min_smoothed",
    "x_Cf_peak",
    "Cf_peak_smoothed",
    "Cf_threshold",
    "x_mean_Cf_crosses_threshold",
    "x_gamma_0.1",
    "x_gamma_0.5",
    "x_gamma_0.9",
    "x_H_below_2.2",
    "x_H_below_1.7",
    "Cf_negative_x",
):
    rows.append(
        [k]
        + [
            " / ".join(
                str(f3(R_[l]["transition"][s].get(k)))
                if not isinstance(R_[l]["transition"][s].get(k), list)
                else str([round(v, 3) for v in R_[l]["transition"][s][k]])
                for s in ("upper", "lower")
            )
            for l in labs
        ]
    )
md += table(rows, ["upper / lower"] + labs)
md += [
    "Near-trip development: `bl_vs_x.csv` columns k_max, urms_max_plus, n_kmax (per station from x/c = 0.02), Cf_rms_point.",
    "",
]

if any("kz_n15" in R_[l] for l in labs):
    md += [
        "## 10./11. Spectra and correlations",
        "",
        "`spectra_kz.csv` / `correlations_dz.csv`: Euu, Evv, Eww(kz) (one-sided, sum E dk = variance; premultiplied in +) and R(dz) at "
        "n+ = 15 and n/delta99 = 0.1, 0.3, 0.6, 0.9 (target codes "
        + json.dumps(next((R_[l]["target_codes"] for l in labs if R_[l].get("target_codes")), {}))
        + "), "
        "z-lines resampled to uniform points with the element polynomials. `spectra_kx.csv` / `correlations_ds.csv`: Euu(kx) "
        f"(Hann window over x0 ± {args.kx_half}c of wall-parallel curves) and R_uu(ds) about the segment centre. "
        "`velocity_psd_time.csv`: temporal PSD of u_s', v_n' at the observation probes (if recorded), `wall_shear_psd_time.csv` of the span-averaged wall shear.",
        "",
    ]
    rows = [
        [
            l,
            "upper" if q["side"] > 0 else "lower",
            f"{q['x']:.2f}",
            f"{q['n_plus']:.1f}",
            f"{q['lambda_z_plus_peak_uu']:.0f}",
            f"{q['Lz_plus']:.0f}",
        ]
        for l in labs
        for q in R_[l].get("kz_n15", [])
        if any(abs(q["x"] - x0) < 0.006 for x0 in args.spec_x)
    ]
    md += table(rows, ["run", "side", "x/c", "n+", "peak lambda_z+ of kz Euu", "Lz+"])

if any("_bud" in R_[l] for l in labs):
    md += [
        "## 13. Budgets",
        "",
        "`budgets_plus.csv`: k, R_ss, R_nn, R_zz, R_sn budgets (times nu / u_tau^4) per station and n: convection, production, pressure "
        "strain, pressure diffusion, turbulent transport, viscous diffusion, -dissipation and the residual (dR/dt plus the statistical "
        "error). All derivatives are exact element derivatives of the instantaneous fields.",
        "",
    ]
    rows = []
    for l in labs:
        bd = R_[l].get("_bud")
        if bd is None:
            continue
        for x0 in args.profile_x:
            xs_ = np.unique(bd["x"][bd["side"] == 1])
            xn = xs_[np.argmin(np.abs(xs_ - x0))]
            m = (bd["side"] == 1) & (bd["x"] == xn)
            eps = -bd["k_eps"][m]
            res_ = np.abs(bd["k_residual"][m])
            inner = bd["n+"][m] < 30
            rows.append(
                [
                    l,
                    f"{xn:.2f}",
                    f3(eps[0], ".4f"),
                    f3(np.nanmax(bd["k_prod"][m]), ".4f"),
                    f3(np.nanmax(res_[inner]) / np.nanmax(eps), ".2f"),
                    f3(np.nanmax(res_) / np.nanmax(eps), ".2f"),
                ]
            )
    md += table(
        rows,
        [
            "run",
            "x/c upper",
            "eps+ at wall",
            "max P+",
            "max |residual| / max eps (n+ < 30)",
            "max |residual| / max eps (all n)",
        ],
    )

if any("wake" in R_[l] for l in labs):
    md += [
        "## 14. Wake",
        "",
        "`wake_profiles.csv`: U, V, P, R_xx, R_xy, R_yy, R_zz, k on y = -0.3..0.3 at each x/c. CD_wake = 2 int[U(1-U) - R_xx + (p_inf - P)] dy.",
        "",
    ]
    rows = []
    for l in labs:
        for q in R_[l].get("wake", []):
            rows.append(
                [
                    l,
                    f"{q['x']:.2f}",
                    f"{q['deficit']:.4f}",
                    f"{q['y_center']:.4f}",
                    f"{q['half_width']:.4f}",
                    f"{q['theta_w']:.5f}",
                    f"{q['dstar_w']:.5f}",
                    f"{q['CD_wake']:.5f}",
                    f"{q['k_max']:.2e}",
                ]
            )
    md += table(rows, ["run", "x/c", "deficit", "y centre", "half width", "theta_w", "delta*_w", "CD_wake", "k max"])

md += ["## 15. Numerical resolution", ""]
rows = []
for k in ("polynomialOrder", "cubaturePolynomialOrder", "dt", "timeStepper", "viscosity", "regularization"):
    rows.append([k] + [str(R_[l]["case"].get(k, "-")) for l in labs])
for l in labs:
    if "fields" in R_[l]:
        rows.append(
            ["elements, z levels, Lz"]
            + [
                f"{R_[q]['fields']['elements']}, {R_[q]['fields']['nz']}, {R_[q]['fields']['Lz']:.4g}"
                if "fields" in R_[q]
                else "-"
                for q in labs
            ]
        )
        break
for k, nm in (
    ("dn1_plus", "first point n+ (max over x>0.15)"),
    ("ds_plus", "station spacing s+ (max, x>0.15)"),
    ("dz_plus_max", "z spacing z+ max"),
    ("dz_plus_mean", "z spacing z+ mean"),
    ("dt_plus", "dt+ (max)"),
):
    row = [nm]
    for l in labs:
        b = R_[l].get("_blx")
        row.append(f3(np.nanmax(b[k][b["x"] > 0.15])) if b is not None else "-")
    rows.append(row)
md += table(rows, ["quantity"] + labs)
md += [
    "No grid or time-step study was run; the resolution numbers are from the u_tau of the fields, per station in `bl_vs_x.csv`. "
    "The budget residual (section 13) is the measure of how far the statistics are from converged.",
    "",
]

md += ["## 16. Statistical convergence", ""]
rows = []
for l in labs:
    r = R_[l]
    fl = r.get("fields")
    rows.append(
        [
            l,
            f"{r['walls']['t'][0]:.3f}-{r['walls']['t'][1]:.3f} ({r['walls']['T']:.3f} c/U, {r['walls']['samples']} samples)",
            f"{fl['n']} fields, t {fl['t'][0]:.3f}-{fl['t'][1]:.3f} (~{fl['window_cU']:.2f} c/U)" if fl else "-",
            f"{r['forces']['CD']['tau_int']:.3f} / {r['forces']['CD']['n_eff']:.0f}",
            f"{100 * (r['forces']['CD']['half2'] / r['forces']['CD']['half1'] - 1):+.2f}%",
        ]
    )
md += table(
    rows,
    [
        "run",
        "wall-sample window",
        "field statistics",
        "CD integral time (c/U) / effective samples",
        "CD drift 2nd vs 1st half",
    ],
)
md += [
    "Intervals: time series with T / (2 tau_int) effective samples; field statistics with the Student t factor over the fields "
    "(0.15 c/U apart, treated as independent; spanwise averaging within a field is included in each field's value).",
    "",
]

md += [
    "## 17. Normalization",
    "",
    "c = 1, U_inf = 1, rho = 1, nu = 1/200000 (Re_c = 200 000), time in c/U_inf. Wall frame s (downstream tangent), n (normal into the fluid), z. "
    "Cf = tau_s / (0.5 rho U_inf^2), Cp = 1 + (p - p_stag) / 0.5 per sample (Cp_p0 in `profiles.csv` with the upstream total pressure p0 of the "
    "field), forces per span Lz. Viscous units with the local u_tau = sqrt(|nu dU_s/dn|_wall) of the field statistics; "
    "delta99 from U_s / U_I = 0.99, U_I = sqrt(2 (p0 - P) - U_n^2) (Griffin et al. 2021), delta* and theta integrated to delta99 with U_e = U_I(delta99), "
    "Clauser beta = delta* / tau_w dP_e/ds.",
    "",
]

md += [
    "## 18. Snapshots",
    "",
    "Restart fields (`chk<actions>0.f00000`, Nek5000 format with coordinates, x, u, p; open the `.nek5000` file in ParaView / VisIt):",
    "",
]
rows = []
for l in labs:
    for s in R_[l]["snapshots"]:
        rows.append([l, str(s["actions"]), f"{s['time']:.4f}", s["file"], f3(s["GB"], ".2f"), str(s["paraview"])])
md += table(rows, ["run", "actions", "t", "file", "GB", "ParaView"])
md += ["## Figures", ""] + [f"![{f}]({f})" for f in figs]
open(os.path.join(args.out, "report.md"), "w").write("\n".join(md) + "\n")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


json.dump(clean(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print(f"[BL] {args.out}/report.md, summary.json, {len(figs)} figures written")
