"""
Friction velocity u_tau(x) of the suction side from an uncontrolled deploy.py run (span- and time-averaged wall
shear stress, u_tau = sqrt(tau_w), rho = 1), fitted with a polynomial in x/c over the control window, and an env
YAML that uses it instead of the polynomial of drl_init.f (probe height and observation scaling; the action amplitude
stays the one of the base YAML):

    python fit_utau.py runs/zero_trip60 --skip 500 --out wing_utau_trip60.yml
    python fit_utau.py runs/zero_ref1 --skip 2420 --out wing_utau_ref1.yml
    python fit_utau.py runs/zero_trip60 --skip 500 --out wing_utau_dtp_trip60.yml --dt-plus 0.67  # also dt+

The YAML has `base: wing.yml` and overrides only wall_units.u_tau (and steps_per_action with --dt-plus).
"""

import argparse
import glob
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--skip", type=int, default=0, help="discard the samples of the first actions (transient)")
p.add_argument("--range", type=float, nargs=2, default=[0.2, 0.9], help="x/c range of the fit (window 0.25-0.86)")
p.add_argument("--degree", type=int, default=6)
p.add_argument("--base", default="wing.yml")
p.add_argument("--dt-plus", type=float, help="also set steps_per_action from this dt+ (channel: 0.67)")
p.add_argument("--out", required=True)
args = p.parse_args()

import yaml

geo = dict(np.load(os.path.join(args.run, "walls_geometry.npz")))
parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(args.run, "walls_*_*.npz")))]
parts = [q for q in parts if len(q["action"])]
action = np.concatenate([q["action"] for q in parts])
keep = action > args.skip
if keep.sum() < 2:
    raise SystemExit(f"{args.run}: only {keep.sum()} samples after {args.skip} actions")
if keep.sum() < 50:
    print(f"WARNING: only {keep.sum()} samples: the fit follows a snapshot, not the mean flow")

xs, taus, sds = [], [], []
for b in (1, 2):
    if f"x_b{b}" not in geo:
        continue
    x, y = geo[f"x_b{b}"], geo[f"y_b{b}"]
    n = np.stack([geo[f"nx_b{b}"], geo[f"ny_b{b}"]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.stack([n[:, 1], -n[:, 0]], axis=1)
    t *= np.sign(t[:, 0] + 1e-300)[:, None]
    tr = np.concatenate([q[f"b{b}"] for q in parts])[keep]  # (samples, stations, [tx, ty, tz, p])
    tau = tr[:, :, 0] * t[:, 0] + tr[:, :, 1] * t[:, 1]
    up = y > 0
    xs.append(x[up])
    taus.append(tau.mean(0)[up])
    # uncertainty of the time mean from the two halves of the samples
    h = len(tau) // 2
    sds.append(0.5 * np.abs(tau[:h].mean(0) - tau[h:].mean(0))[up])
x, tau, sd = (np.concatenate(v) for v in (xs, taus, sds))
o = np.argsort(x)
x, tau, sd = x[o], tau[o], sd[o]
m = (x >= args.range[0]) & (x <= args.range[1])
if np.any(tau[m] <= 0):
    raise SystemExit(f"reversed mean flow (tau_w <= 0) at x/c {x[m][tau[m] <= 0]}: u_tau undefined there")
ut = np.sqrt(tau[m])
c = np.polynomial.polynomial.polyfit(x[m], ut, args.degree)
fit = np.polynomial.polynomial.polyval(x[m], c)

base = yaml.safe_load(open(args.base))
old = np.polynomial.polynomial.polyval(x[m], base["wall_units"]["u_tau"]["polynomial"])
print(
    f"{args.run}: {keep.sum()} samples (actions {action[keep].min()}-{action[keep].max()}), "
    f"{m.sum()} stations in x/c {args.range[0]}-{args.range[1]}"
)
print(
    f"fit residual rms {np.sqrt(np.mean((fit - ut) ** 2)) / ut.mean():.2%} of the mean u_tau; "
    f"time-mean uncertainty (halves) {np.mean(sd[m] / tau[m]) / 2:.2%} of u_tau"
)
for lo in np.arange(0.25, 0.86, 0.05):
    k = (x[m] >= lo) & (x[m] < lo + 0.05)
    if k.any():
        print(
            f"  x/c {lo:.2f}-{lo + 0.05:.2f}: u_tau {ut[k].mean():.4f} (fit {fit[k].mean():.4f}), "
            f"old polynomial {old[k].mean():.4f}, ratio actual/old {ut[k].mean() / old[k].mean():.3f}"
        )

out = {
    "base": os.path.relpath(args.base, os.path.dirname(os.path.abspath(args.out)) or "."),
    "wall_units": {"u_tau": {"axis": "x", "polynomial": [float(v) for v in c]}},
}
if args.dt_plus:
    out["environment"] = {"steps_per_action": {"dt_plus": args.dt_plus}}
with open(args.out, "w") as f:
    f.write(
        f"# u_tau(x) fitted to {os.path.normpath(args.run)} (actions > {args.skip}, x/c {args.range[0]}-"
        f"{args.range[1]}, degree {args.degree}) by fit_utau.py\n"
    )
    yaml.safe_dump(out, f, sort_keys=False)
print(f"{args.out} written")
