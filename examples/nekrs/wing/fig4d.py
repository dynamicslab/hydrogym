"""
Drag reductions of controlled wing runs against an uncontrolled run, from the walls_*.npz of deploy.py:

- local skin-friction reduction on the suction side over 0.25 <= x/c <= 0.858 (x-uniform mean of c_f);
- reduction of the friction drag C_d,f and of the total drag C_d of the wing, integrated over the complete surface,
  as time means over every solver step (force monitors of solver_steps.csv in the log* directories of deploy.py;
  without them, from the wall samples, with a warning);
- the same with the friction integrated over the stations 0.0197 <= x/c <= 0.995 of both sides (the range of the
  Nek5000 statistics of Wang, Atzori & Vinuesa, JFM 2025) and divided by their uncontrolled C_d,f and C_d
  (--nek5000-coeffs);
- c_f(x) of the suction side, normalised with U_inf and with the edge velocity U_e(x).

The wall samples (all other metrics) are taken by deploy.py after every --wall-steps-th solver step (default 10),
also within the actions, so their mean over the window is the time mean. Runs made with --wall-steps 0 have one
sample after the last solver step of every --walls-every-th action. A policy that changes its action in steps makes
the friction vary within the action, so these end-of-action samples are not the time mean
of the strip and station metrics. dCDf_end and dCD_end in OUT.json are C_d,f and C_d from the wall samples, for
comparison with the step means. The station metrics (dCDf_nek, dCD_nek) use the wall samples.

Each controlled run is paired with the uncontrolled run over the same absolute time and averaged over a window
t0 <= t - t_start <= t1 of control time. Runs of one controller from different start states are realisations: the
table gives their mean and 95 % interval (t distribution); --diff A-B gives the paired differences per start state.

    python fig4d.py --ref runs/zero --t0 1 --t1 11 --diff TD3-OC --diff TD3-UB --out fig4d \
        TD3=runs/td3:28 TD3=runs/td3_r2:33.2 OC=runs/oc:28 OC=runs/oc_r2:33.2 UB=runs/ub:28 UB=runs/ub_r2:33.2

A spec is LABEL=RUN:T_START (t_start: the time at which the run starts its control); RUN may be DIR+DIR2+... for a
run continued in further directories (joined by time, as --ref). --ref takes one or more runs
of the uncontrolled trajectory (e.g. a run and its continuation), joined by time. Writes OUT.json, OUT.md and
OUT_cf.csv, and OUT.png if matplotlib is installed.

c_f = 2 tau_w / (rho U_inf^2) with tau_w the downstream component of the viscous traction (span average);
c_f,e = c_f (U_inf / U_e)^2. U_e(x): --ue FILE (columns x/c and U_e/U_inf, e.g. the edge velocity of the reference
data), else the inviscid estimate from the wall pressure of the uncontrolled run, U_e = sqrt(2 (p_0 - p_w)) with p_0
the stagnation pressure (largest wall pressure). U_inf = c = rho = 1.
"""

import argparse
import glob
import json
import os

import numpy as np

NEK5000_COEFFS = (0.010486795573193472, 0.011837434331973491)  # C_d,f and C_d of the uncontrolled flow (Ref1)
METRICS = ("DR_strip", "dCDf", "dCD", "dCDf_nek", "dCD_nek")
HEADER = {
    "DR_strip": "0.25-0.858 (%)",
    "DR25": "0.25-0.86 (%)",  # power_forms.py
    "dCDf": "dC_d,f (%)",
    "dCD": "dC_d (%)",
    "dCDf_nek": "dC_d,f stations (%)",
    "dCD_nek": "dC_d stations (%)",
}
_trapz = getattr(np, "trapezoid", None) or np.trapz


class Wing:
    """Station geometry of the wing (walls_geometry.npz): both boundaries, span-integrated weights."""

    def __init__(self, run, bids=(1, 2)):
        g = dict(np.load(os.path.join(run, "walls_geometry.npz")))
        self.bids = bids
        cat = lambda k: np.concatenate([g[f"{k}_b{b}"] for b in bids])  # noqa: E731
        self.x, self.y, self.ws = cat("x"), cat("y"), cat("wsum")
        n = np.stack([cat("nx"), cat("ny")], axis=1)
        self.n = n / np.linalg.norm(n, axis=1, keepdims=True)
        t = np.stack([self.n[:, 1], -self.n[:, 0]], axis=1)  # downstream tangent
        self.t = t * np.sign(t[:, 0] + 1e-300)[:, None]
        self.span = max(float(g[f"span_b{b}"]) for b in bids)
        self.up = self.y > 0
        self.s25 = self.up & (self.x >= 0.25) & (self.x <= 0.86)
        self.stations = (self.x >= 0.0197) & (self.x <= 0.995)
        self.iss = np.nonzero(self.up)[0][np.argsort(self.x[self.up], kind="stable")]  # suction side by x
        self.raw = g

    def max_dx(self, run):
        g = np.load(os.path.join(run, "walls_geometry.npz"))
        return max(float(np.abs(g[f"x_b{b}"] - self.raw[f"x_b{b}"]).max()) for b in self.bids)


def load_steps(runs, bids):
    """
    Force monitors of every solver step (solver_steps.csv of the log* directories of deploy.py), unique in time;
    a later launch (restart from a chunk) replaces the steps an earlier one computed after that chunk.
    None if a run has no such file or the file lacks the force columns of the boundaries.
    """
    parts = []
    for run in runs:
        files = sorted(glob.glob(os.path.join(run, "log*", "solver_steps.csv")))
        if not files:
            return None
        for f in files:
            with open(f) as fh:
                names = fh.readline().strip().split(",")
            need = ["time", "dt"] + [f"F{q}x_b{b}" for b in bids for q in ("v", "p")]
            if any(n not in names for n in need):
                return None
            a = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            if not len(a):
                continue
            col = {n: a[:, names.index(n)] for n in need}
            fv = sum(col[f"Fvx_b{b}"] for b in bids)
            fp = sum(col[f"Fpx_b{b}"] for b in bids)
            parts.append(np.column_stack([col["time"], col["dt"], fv, fp]))
    if not parts:
        return None
    a = np.concatenate(parts[::-1])  # latest launch first: np.unique keeps the first occurrence
    _, u = np.unique(np.round(a[:, 0], 9), return_index=True)
    a = a[u]
    return {"t": a[:, 0], "dt": a[:, 1], "fv": a[:, 2], "fp": a[:, 3]}


def step_mean(st, lo, hi, h):
    """dt-weighted mean C_d,f and C_d,p of the solver steps that end in (lo, hi], and the covered interval."""
    m = (st["t"] > lo + 1e-9) & (st["t"] <= hi + 1e-9)
    if not m.any():
        return None
    w = st["dt"][m]
    cdf = float((st["fv"][m] * w).sum() / w.sum() / h)
    cdp = float((st["fp"][m] * w).sum() / w.sum() / h)
    return cdf, cdp, float(st["t"][m][0] - w[0]), float(st["t"][m][-1])


def load_walls(runs, wing):
    """Wall samples of one or more runs (later runs continue earlier ones), unique in time, and the solver steps."""
    series = {k: [] for k in ("t", "action", "cdf", "cdp", "cdf_st", "ff25", "tau", "p", "cd_forces")}
    pw = {k: [] for k in ("t", "power", "cd")}
    for run in runs:
        files = sorted(glob.glob(os.path.join(run, "walls_*_*.npz")))
        if not files:
            raise SystemExit(f"{run}: no walls_*.npz")
        for f in files:
            q = np.load(f)
            if len(q["action"]):
                tr = np.concatenate([q[f"b{b}"] for b in wing.bids], axis=1)  # (samples, stations, [tx, ty, tz, p])
                tau = tr[:, :, 0] * wing.t[:, 0] + tr[:, :, 1] * wing.t[:, 1]
                fx = tr[:, :, 0] * wing.ws
                h = 0.5 * wing.span
                series["t"].append(q["time"])
                series["action"].append(q["action"])
                series["cdf"].append(fx.sum(1) / h)
                series["cdp"].append((-tr[:, :, 3] * wing.n[:, 0] * wing.ws).sum(1) / h)
                series["cdf_st"].append(fx[:, wing.stations].sum(1) / h)
                series["ff25"].append((tau[:, wing.s25] * wing.ws[wing.s25]).sum(1))
                series["tau"].append(tau.astype(np.float32))
                series["p"].append(tr[:, :, 3].astype(np.float32))
                series["cd_forces"].append(sum(q[f"F{b}"][:, 0] + q[f"F{b}"][:, 3] for b in wing.bids) / h)
            if "power" in q.files and len(q["power"]):
                pw["t"].append(q["power_time"])
                pw["power"].append(q["power"])
                pw["cd"].append(q["power_cd"])
    d = {k: np.concatenate(v) for k, v in series.items()}
    _, u = np.unique(np.round(d["t"], 7), return_index=True)  # sorted by time, the first sample of a repeated time
    d = {k: v[u] for k, v in d.items()}
    if pw["t"]:
        p = {k: np.concatenate(v) for k, v in pw.items()}
        _, u = np.unique(np.round(p["t"], 7), return_index=True)
        d["power"] = {k: v[u] for k, v in p.items()}
    d["steps"] = load_steps(runs, wing.bids)
    return d


def pair(t_ref, t, tol):
    """Index of the nearest reference sample of every time t, and whether it lies within tol."""
    j = np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1)
    j = np.where(np.abs(t_ref[j - 1] - t) < np.abs(t_ref[j] - t), j - 1, j)
    return j, np.abs(t_ref[j] - t) <= tol


def trap_mean(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return float(_trapz(y[m], x[m]) / (x[m][-1] - x[m][0]))


def window(c, ref, t_start, t0, t1, tol):
    """Sample indices of the run (i) and of the paired reference (r) in the window, and coverage information."""
    tc = c["t"] - t_start
    m = (tc >= t0 - 1e-9) & (tc <= t1 + 1e-9)
    j, ok = pair(ref["t"], c["t"], tol)
    ok &= m
    i, r = np.nonzero(ok)[0], j[ok]
    if not len(i):
        raise SystemExit(f"no paired samples in t_c = [{t0}, {t1}] (run: t_c {tc.min():.3f}-{tc.max():.3f})")
    info = dict(n=int(len(i)), n_window=int(m.sum()), window=[float(tc[i][0]), float(tc[i][-1])])
    gap = float(np.max(np.diff(tc[i]))) if len(i) > 1 else np.inf
    spacing = float(np.median(np.diff(tc[m]))) if m.sum() > 1 else np.inf
    info["complete"] = bool(
        info["window"][0] <= t0 + 2 * spacing and info["window"][1] >= t1 - 2 * spacing and gap <= 2.5 * spacing
    )
    return i, r, info


def metrics(c, ref, i, r, wing, nek5000, twin=None):
    """
    Metrics of the samples i of run c against the paired reference samples r. twin: absolute time window of the step
    means of C_d,f and C_d (clipped to the steps both runs have); None: wall samples only.
    """
    e = {}
    cf, cf0 = 2 * c["tau"][i].mean(0), 2 * ref["tau"][r].mean(0)
    xs = wing.x[wing.iss]
    e["DR_strip"] = 100 * (1 - trap_mean(xs, cf[wing.iss], 0.25, 0.858) / trap_mean(xs, cf0[wing.iss], 0.25, 0.858))
    cdf0, cdp0 = ref["cdf"][r].mean(), ref["cdp"][r].mean()
    cdf, cdp = c["cdf"][i].mean(), c["cdp"][i].mean()
    cdp_w, cdp0_w = cdp, cdp0  # wall samples, for the station metric (also from the wall samples)
    e["dCDf_end"] = -100 * (cdf - cdf0) / cdf0
    e["dCD_end"] = -100 * (cdf + cdp - cdf0 - cdp0) / (cdf0 + cdp0)
    e["cd_sampling"] = "wall samples"
    if twin is not None and c.get("steps") is not None and ref.get("steps") is not None:
        st, st0 = c["steps"], ref["steps"]
        lo = max(twin[0], st["t"][0] - st["dt"][0], st0["t"][0] - st0["dt"][0])
        hi = min(twin[1], st["t"][-1], st0["t"][-1])
        h = 0.5 * wing.span
        a, b = (step_mean(st, lo, hi, h), step_mean(st0, lo, hi, h)) if hi > lo else (None, None)
        if a is not None and b is not None:
            (cdf, cdp), (cdf0, cdp0) = a[:2], b[:2]
            e["cd_sampling"] = "every solver step"
            e["cd_window"] = [lo, hi]
            if hi - lo < twin[1] - twin[0] - 0.01:
                print(f"warning: solver steps cover t = {lo:.3f}-{hi:.3f} only, not {twin[0]:.3f}-{twin[1]:.3f}")
    if e["cd_sampling"] != "every solver step":
        print("warning: C_d,f and C_d from the wall samples, not from every solver step")
    e["dCDf"] = -100 * (cdf - cdf0) / cdf0
    e["dCD"] = -100 * (cdf + cdp - cdf0 - cdp0) / (cdf0 + cdp0)
    e.update(CDf=float(cdf), CD=float(cdf + cdp), CDf0=float(cdf0), CD0=float(cdf0 + cdp0))
    e["dCD_forces"] = -100 * (c["cd_forces"][i].mean() / ref["cd_forces"][r].mean() - 1)  # check of dCD
    dq = c["cdf_st"][i].mean() - ref["cdf_st"][r].mean()
    e["dCDf_nek"] = -100 * dq / nek5000[0]
    e["dCD_nek"] = -100 * (dq + cdp_w - cdp0_w) / nek5000[1]
    return e, cf, cf0


def edge_velocity(args, wing, ref, r):
    """U_e at the suction-side stations: from --ue, else from the wall pressure of the reference samples r."""
    xs = wing.x[wing.iss]
    if args.ue:
        rows = []
        with open(args.ue) as f:
            for line in f:  # two columns, separated by commas or blanks; header lines are skipped
                try:
                    rows.append([float(v) for v in line.replace(",", " ").split()[:2]])
                except ValueError:
                    continue
        a = np.array([r for r in rows if len(r) == 2])
        a = a[np.argsort(a[:, 0])]
        return np.interp(xs, a[:, 0], a[:, 1], left=np.nan, right=np.nan), args.ue
    p = ref["p"][r].mean(0)
    return np.sqrt(np.maximum(2 * (p.max() - p[wing.iss]), 0.0)), "wall pressure of the uncontrolled run (Bernoulli)"


def t95(n):
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except ImportError:
        table = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262]
        return table[n - 2] if n - 2 < len(table) else 1.96


def summary(values):
    v = np.asarray(values, float)
    ci = t95(len(v)) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return dict(mean=float(v.mean()), ci95=float(ci), n=int(len(v)), values=v.tolist())


def parse_spec(s):
    label, rest = s.split("=", 1)
    run, t_start = rest.rsplit(":", 1)
    return label, run, float(t_start)


def common_args(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="LABEL=RUN:T_START")
    p.add_argument("--ref", nargs="+", required=True, help="uncontrolled run(s), joined by time")
    p.add_argument("--t0", type=float, default=1.0, help="window start, control time t - t_start (c/U_inf)")
    p.add_argument("--t1", type=float, default=11.0, help="window end")
    p.add_argument("--pair-tol", type=float, default=2e-3, help="largest time difference of paired samples")
    p.add_argument("--diff", action="append", default=[], help="A-B: paired differences per start state")
    p.add_argument(
        "--nek5000-coeffs",
        type=float,
        nargs=2,
        default=NEK5000_COEFFS,
        metavar=("CDF", "CD"),
        help="uncontrolled C_d,f and C_d of the station metric (default: Nek5000 statistics of Ref1)",
    )
    p.add_argument("--out", default="fig4d")
    return p


def evaluate(args, wing, ref, extra=None):
    """Per-run metrics (and extra(c, ref, i, r) values), grouped by label."""
    runs, labels = {}, {}
    for s in args.specs:
        label, run, t_start = parse_spec(s)
        parts = run.split("+")  # a run continued in further directories: RUN+RUN_CONTINUATION
        c = load_walls(parts, wing)
        i, r, info = window(c, ref, t_start, args.t0, args.t1, args.pair_tol)
        twin = (t_start + args.t0, t_start + args.t1)
        e, cf, cf0 = metrics(c, ref, i, r, wing, args.nek5000_coeffs, twin)
        e.update(info, label=label, t_start=t_start, geometry_max_dx=max(wing.max_dx(d) for d in parts))
        if extra is not None:
            e.update(extra(c, ref, i, r, t_start))
        e["_cf"], e["_cf0"], e["_r"] = cf, cf0, r
        runs[run] = e
        labels.setdefault(label, []).append(run)
        if not info["complete"]:
            print(f"warning: {run}: samples cover t_c = {info['window']} only, not [{args.t0}, {args.t1}]")
    return runs, labels


def group_tables(runs, labels, diffs, keys):
    stats_ = {lab: {k: summary([runs[r][k] for r in rs]) for k in keys} for lab, rs in labels.items()}
    for d in diffs:
        a, b = d.split("-", 1)
        ta = {runs[r]["t_start"]: r for r in labels.get(a, [])}
        tb = {runs[r]["t_start"]: r for r in labels.get(b, [])}
        both = sorted(set(ta) & set(tb))
        if not both:
            print(f"--diff {d}: no common start state")
            continue
        stats_[f"{a} - {b} (points)"] = {k: summary([runs[ta[t]][k] - runs[tb[t]][k] for t in both]) for k in keys}
    return stats_


def markdown(stats_, keys, title):
    head = "| case | " + " | ".join(HEADER.get(k, k) for k in keys) + " | n |"
    lines = [title, "", head, "|" + "---|" * (len(keys) + 2)]
    for lab, s in stats_.items():
        cells = []
        for k in keys:
            m, ci = s[k]["mean"], s[k]["ci95"]
            cells.append(f"{m:.2f}" + (f" ± {ci:.2f}" if np.isfinite(ci) else ""))
        lines.append(f"| {lab} | " + " | ".join(cells) + f" | {s[keys[0]]['n']} |")
    return "\n".join(lines) + "\n"


def jsonable(runs):
    return {r: {k: v for k, v in e.items() if not k.startswith("_")} for r, e in runs.items()}


def main():
    p = common_args(__doc__)
    p.add_argument("--ue", help="edge velocity U_e(x)/U_inf: file with columns x/c and U_e (default: wall pressure)")
    args = p.parse_args()

    wing = Wing(args.ref[0])
    ref = load_walls(args.ref, wing)
    runs, labels = evaluate(args, wing, ref)
    stats_ = group_tables(runs, labels, args.diff, METRICS)
    title = (
        f"Window {args.t0} <= t_c <= {args.t1} c/U_inf; reductions in per cent against the paired uncontrolled run; "
        "mean ± 95 % interval over the start states"
    )
    md = markdown(stats_, list(METRICS), title)

    xs = wing.x[wing.iss]
    ue, ue_src = edge_velocity(args, wing, ref, np.unique(np.concatenate([e["_r"] for e in runs.values()])))
    inv = np.where(ue > 1e-2, 1 / np.maximum(ue, 1e-2) ** 2, np.nan)  # (U_inf / U_e)^2, none at stagnation
    cols, names = [xs, ue], ["x_c", "U_e"]
    for lab, rs in labels.items():
        cf = np.mean([runs[r]["_cf"][wing.iss] for r in rs], axis=0)
        cf0 = np.mean([runs[r]["_cf0"][wing.iss] for r in rs], axis=0)
        cols += [cf, cf * inv, cf0, cf0 * inv]
        names += [f"cf_{lab}", f"cfe_{lab}", f"cf0_{lab}", f"cfe0_{lab}"]
    np.savetxt(f"{args.out}_cf.csv", np.column_stack(cols), delimiter=",", header=",".join(names), comments="")

    res = dict(window=[args.t0, args.t1], reference=args.ref, edge_velocity=ue_src, groups=stats_, runs=jsonable(runs))
    with open(f"{args.out}.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{args.out}.md", "w") as f:
        f.write(md)
    print(md)
    for r, e in runs.items():
        print(
            f"{r}: n {e['n']}, t_c {e['window'][0]:.3f}-{e['window'][1]:.3f}, dC_d {e['dCD']:.2f} % "
            f"({e['cd_sampling']}; wall samples {e['dCD_end']:.2f} %, "
            f"their integrated forces {e['dCD_forces']:.2f} %), "
            f"geometry max |dx| {e['geometry_max_dx']:.1e}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{args.out}.json, .md, _cf.csv written (no matplotlib: no figure)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ok = np.isfinite(inv) & (xs >= 0.05)  # without the leading-edge peak
    first = True
    for lab, rs in labels.items():
        cf = np.mean([runs[r]["_cf"][wing.iss] for r in rs], axis=0)
        if first:
            cf0 = np.mean([runs[r]["_cf0"][wing.iss] for r in rs], axis=0)
            ax[0].plot(xs[ok], (cf0 * inv)[ok], "k", label="uncontrolled")
            first = False
        ax[0].plot(xs[ok], (cf * inv)[ok], label=lab)
    ax[0].axvspan(0.25, 0.86, color="0.9", zorder=0)
    ax[0].set(xlabel="x/c", ylabel="c_f,e (suction side)", xlim=(0.05, 1))
    ax[0].legend()
    labs = list(stats_)
    for k, key in enumerate(("DR_strip", "dCDf", "dCD")):
        m = [stats_[lab][key]["mean"] for lab in labs]
        ci = [stats_[lab][key]["ci95"] for lab in labs]
        ax[1].bar(np.arange(len(labs)) + 0.25 * (k - 1), m, 0.25, yerr=np.nan_to_num(ci), label=HEADER[key])
    ax[1].set_xticks(np.arange(len(labs)), labs, rotation=20, fontsize=8)
    ax[1].set_ylabel("reduction (%)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"{args.out}.json, .md, _cf.csv, .png written")


if __name__ == "__main__":
    main()
