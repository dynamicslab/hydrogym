"""
Wing deployment results (paper Fig. 4) from the walls_*.npz of deploy.py runs: skin friction Cf(x/c) of the
suction side, time-averaged over the samples after --skip, and the drag history and mean, total Cd and its
friction part Cd,f, with the change against the first run (the reference, normally the uncontrolled one).

    python postprocess.py runs/zero runs/oc runs/ub runs/td3_s0 --skip 500 --out fig4.png

Cf = (t . e_s) / (0.5 U_inf^2) with t the viscous traction mu (grad u + grad u^T) . n and e_s the downstream
tangent of the section; Cd = F_x / (0.5 U_inf^2 c L_z) with F = sum over the wing (bid 1 + 2) of w (-p n + t),
n into the fluid. U_inf = 1, c = 1, rho = 1.
"""

import argparse
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--labels", nargs="+")
p.add_argument("--skip", type=int, default=0, help="discard the samples of the first actions (transient)")
p.add_argument("--bids", type=int, nargs="+", default=[1, 2])
p.add_argument("--out", default="wing_results.png")
args = p.parse_args()
labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]


def load(run):
    files = sorted(glob.glob(os.path.join(run, "walls_*_*.npz")))
    if not files:
        raise SystemExit(f"{run}: no walls_*.npz")
    geo = dict(np.load(os.path.join(run, "walls_geometry.npz")))
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([q[k] for q in parts if len(q["action"])]) for k in parts[0]}
    keep = d["action"] > args.skip
    return geo, {k: v[keep] for k, v in d.items()}


def suction_cf(geo, d):
    """Span- and time-averaged Cf of the upper surface (y > 0), sorted by x."""
    xs, cf = [], []
    for b in args.bids:
        x, y = geo[f"x_b{b}"], geo[f"y_b{b}"]
        n = np.stack([geo[f"nx_b{b}"], geo[f"ny_b{b}"]], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        # downstream tangent of the upper surface: n rotated by -90 degrees, oriented with +x
        t = np.stack([n[:, 1], -n[:, 0]], axis=1)
        t *= np.sign(t[:, 0] + 1e-300)[:, None]
        tr = d[f"b{b}"].mean(axis=0)  # (stations, [tx, ty, tz, p])
        up = y > 0
        xs.append(x[up])
        cf.append(((tr[:, 0] * t[:, 0] + tr[:, 1] * t[:, 1]) / 0.5)[up])
    x, cf = np.concatenate(xs), np.concatenate(cf)
    o = np.argsort(x)
    return x[o], cf[o]


def drag(geo, d):
    """Cd,f and Cd time series: F{b} = (viscous x, y, z, pressure x, y, z)."""
    lz = max(float(geo[f"span_b{b}"]) for b in args.bids)
    ff = sum(d[f"F{b}"] for b in args.bids)
    return ff[:, 0] / (0.5 * lz), (ff[:, 0] + ff[:, 3]) / (0.5 * lz)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
summary = {}
ref = None
for run, lab in zip(args.runs, labels):
    geo, d = load(run)
    x, cf = suction_cf(geo, d)
    cdf, cd = drag(geo, d)
    ax[0].plot(x, cf, label=lab)
    ax[1].plot(d["time"], cd, label=f"{lab} Cd")
    ax[1].plot(d["time"], cdf, "--", label=f"{lab} Cd,f")
    s = {"samples": int(len(d["time"])), "t": [float(d["time"][0]), float(d["time"][-1])],
         "Cd": float(cd.mean()), "Cd_f": float(cdf.mean())}
    if 2 in args.bids:  # friction drag of the control window alone
        s["Cd_f_window"] = float(d["F2"][:, 0].mean() / (0.5 * float(geo["span_b2"])))
    if ref is None:
        ref = s
    else:
        s["dCd_%"] = 100 * (s["Cd"] / ref["Cd"] - 1)
        s["dCd_f_%"] = 100 * (s["Cd_f"] / ref["Cd_f"] - 1)
        if "Cd_f_window" in s:
            s["dCd_f_window_%"] = 100 * (s["Cd_f_window"] / ref["Cd_f_window"] - 1)
    summary[lab] = s
    print(lab, json.dumps(s))
ax[0].axvspan(0.25, 0.86, color="0.9", zorder=0)
ax[0].set_xlabel("x/c")
ax[0].set_ylabel("Cf (suction side)")
ax[0].set_xlim(0, 1)
ax[0].legend()
ax[1].set_xlabel("t U/c")
ax[1].set_ylabel("drag coefficient")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
    json.dump(summary, f, indent=1)
print(f"{args.out} written")
