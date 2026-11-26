"""
Distribution of the actuation of wing runs (deploy.py), recomputed from the observations stored with the wall
samples: for every recorded observation (walls_*.npz `obs`, the input of the next action) the controller of the run
gives each agent's policy output a, and the env's wall_velocity gives the applied velocity:

    v_cmd = clip(a, -1, 1) * amplitude              (amplitude per agent, agents.npz)
    v_w   = v_cmd - mean of v_cmd over each spanwise line, weighted by flux_per_unit (zero net mass flux;
            znmf_group and flux_per_unit of actuators.csv; not for --policy ub, which has a net flux)

Statistics over all agents and the selected actions of the window t0 <= t - t_start < t1 (unweighted samples), per
run and pooled over the runs of one label (realisations):
  - a in units of the bound --unit (default 0.0624334733 U, the amplitude of wing.yml): a = v_cmd / unit, so for
    opposition control the commanded -v'(y+ = 15) in that unit;
  - share at the bound |a| >= 0.95, split into blowing (a >= 0.95) and suction (a <= -0.95); share with |a| < 0.5;
  - mean, rms and kurtosis of a (kurtosis 1 for a switch between two values, 1.8 uniform, 3 Gaussian);
  - v_w / u_tau(x) (u_tau(x): the observation scaling of the agent): rms and the shares beyond +1 and -1.

    python action_hist.py --t0 1 --t1 3.5 --out action_hist \
        TD3=runs/td3:28 TD3=runs/td3_r2:33.2 N1=runs/n1:28 OC=runs/oc:28

A spec is LABEL=RUN:T_START[@POLICY]; POLICY (as deploy.py --policy: zero | oc | ub | td3:<zip> | td3:<released id>)
defaults to the policy of RUN/state.json. Runs must be made with the observations recorded (deploy.py without
--no-obs). Writes OUT.json, OUT.md and OUT.png (matplotlib).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EA = (np.arange(-150, 152) - 0.5) * 0.01  # a: bins centred on k * 0.01
EV = (np.arange(-200, 202) - 0.5) * 0.02  # v_w / u_tau: bins centred on k * 0.02
COUNTS = ("n", "bound", "blowing", "suction", "inside", "v_gt1", "v_lt1")


def group_mean(values, labels, weights):
    """Weighted mean over each label, per point (hydrogym.nekrs.layout.group_mean)."""
    n = int(labels.max()) + 1
    return (np.bincount(labels, weights * values, n) / np.bincount(labels, weights, n))[labels]


def make_controller(policy, ag, alpha, ub_velocity, use_numpy):
    """Policy output a per agent from the observations o [agents, fields], as deploy.py act()."""
    obs_div, amp = ag["obs_div"], np.asarray(ag["amplitude"], float)
    fields = [str(f) for f in ag["fields"]]
    if policy == "zero":
        return lambda o: np.zeros(len(o))
    if policy == "ub":
        return lambda o: np.broadcast_to(ub_velocity / amp, (len(o),))
    if policy == "oc":
        iv = fields.index("v")
        return lambda o: -alpha * o[:, iv] * obs_div[:, iv] / amp
    if not policy.startswith("td3:"):
        raise SystemExit(f"unknown policy {policy}")
    path, model = policy[4:], None
    if not os.path.exists(path):  # a released policy id
        sys.path.insert(0, REPO)
        from policies.load_policy import entry, load_policy

        e = entry(path)
        zips = [f["path"] for f in e["files"] if f["role"] == "weights_sb3"]
        if zips and not use_numpy:
            path = os.path.join(REPO, "policies", zips[0])
        else:
            model = load_policy(e["id"])
    if model is not None:
        x = np.asarray(ag["xyz"], float)[:, 0]
        if model.format == "wall_blocks":
            return lambda o: np.asarray(model(o, x), float).reshape(-1)
        return lambda o: np.asarray(model(o), float).reshape(-1)
    import gymnasium
    from stable_baselines3 import TD3

    co = {
        "observation_space": gymnasium.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        "action_space": gymnasium.spaces.Box(-1, 1, (1,), np.float32),
        "lr_schedule": lambda _: 0.0,
        "learning_rate": 0.0,
        "replay_buffer_class": None,
    }
    sb3 = TD3.load(path, device="cpu", custom_objects=co)
    return lambda o: np.asarray(sb3.predict(o, deterministic=True)[0], np.float64).reshape(-1)


def run_one(args, run, t_start, policy):
    with open(os.path.join(run, "state.json")) as f:
        policy = policy or json.load(f)["policy"]
    agf = os.path.join(run, "agents.npz")
    if not os.path.exists(agf):
        raise SystemExit(f"{run}: no agents.npz (run made with --no-obs?)")
    ag = dict(np.load(agf))
    csv = sorted(glob.glob(os.path.join(run, "log*", "actuators.csv")))[0]  # as the env built them (first launch)
    with open(csv) as f:
        hdr = f.readline().strip().split(",")
    cols = ["x", "y", "z", "flux_per_unit"] + (["znmf_group"] if "znmf_group" in hdr else [])
    tab = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=[hdr.index(c) for c in cols], ndmin=2)
    dxyz = float(np.abs(tab[:, :3] - ag["xyz"]).max())
    if dxyz > 1e-6:
        raise SystemExit(f"{run}: the agents of actuators.csv and agents.npz differ ({dxyz:.1e})")
    fpu = tab[:, 3]
    grp = tab[:, 4].astype(int) if "znmf_group" in hdr and policy != "ub" else None
    amp = np.broadcast_to(np.asarray(ag["amplitude"], float), (len(fpu),))
    ut = ag["obs_div"][:, 0]
    ctrl = make_controller(policy, ag, args.alpha, args.ub_velocity, args.numpy)

    files = sorted(glob.glob(os.path.join(run, "walls_*_*.npz")), key=lambda f: int(os.path.basename(f).split("_")[1]))
    sel = []
    for f in files:
        with np.load(f) as z:
            if "obs" not in z.files:
                continue
            tc = (z["obs_time"] if "obs_time" in z.files else z["time"]) - t_start
        sel += [(f, i) for i in np.nonzero((tc >= args.t0) & (tc < args.t1))[0]]
    if not sel:
        raise SystemExit(f"{run}: no recorded observations in t_c = [{args.t0}, {args.t1})")
    stride = max(1, int(np.ceil(len(sel) / args.max_records)))
    sel = sel[::stride]

    H = {"a": np.zeros(len(EA) - 1), "v": np.zeros(len(EV) - 1)}
    M = {"a": np.zeros(5), "v": np.zeros(5)}  # n, sum x, x^2, x^3, x^4
    C = dict.fromkeys(COUNTS, 0)
    tcs, cur, z = [], None, None
    for f, i in sel:
        if f != cur:
            z, cur = np.load(f), f
            obs, tt = z["obs"], (z["obs_time"] if "obs_time" in z.files else z["time"])
        a = ctrl(obs[i].astype(np.float32))
        cmd = np.clip(a, -1, 1) * amp
        vel = cmd - group_mean(cmd, grp, fpu) if grp is not None else cmd
        aa, vv = cmd / args.unit, vel / ut
        for k, x, E in (("a", aa, EA), ("v", vv, EV)):
            H[k] += np.histogram(x, E)[0]  # values outside the bins (OC: |a| > 1.5) are left out
            M[k] += [x.size, x.sum(), (x**2).sum(), (x**3).sum(), (x**4).sum()]
        C["n"] += aa.size
        C["bound"] += int((np.abs(aa) >= 0.95).sum())
        C["blowing"] += int((aa >= 0.95).sum())
        C["suction"] += int((aa <= -0.95).sum())
        C["inside"] += int((np.abs(aa) < 0.5).sum())
        C["v_gt1"] += int((vv > 1).sum())
        C["v_lt1"] += int((vv < -1).sum())
        tcs.append(float(tt[i]) - t_start)
    print(
        f"{run}: policy {policy}, {len(fpu)} agents, {len(sel)} actions (every {stride}), t_c {min(tcs):.3f}-"
        f"{max(tcs):.3f}, u_tau(x) {ut.min():.4f}-{ut.max():.4f}, ZNMF {'on' if grp is not None else 'off'}",
        flush=True,
    )
    return dict(
        policy=policy,
        t_start=t_start,
        n_agents=len(fpu),
        n_actions=len(sel),
        stride=stride,
        window=[min(tcs), max(tcs)],
        u_tau_range=[float(ut.min()), float(ut.max())],
        hist={k: v.tolist() for k, v in H.items()},
        moments={k: v.tolist() for k, v in M.items()},
        counts=C,
    )


def statistics(rs):
    """Statistics of the pooled samples of the runs rs (per-run records of run_one)."""
    M = {k: np.sum([r["moments"][k] for r in rs], 0) for k in ("a", "v")}
    C = {k: sum(r["counts"][k] for r in rs) for k in COUNTS}
    out = {}
    for k in ("a", "v"):
        n, s1, s2, s3, s4 = M[k][0], *(M[k][1:] / M[k][0])
        var = s2 - s1**2
        m4 = s4 - 4 * s1 * s3 + 6 * s1**2 * s2 - 3 * s1**4
        out[k] = dict(mean=float(s1), rms=float(np.sqrt(s2)), kurtosis=float(m4 / var**2) if var > 0 else float("nan"))
    n = C["n"]
    out["a"].update(
        at_bound=100 * C["bound"] / n,
        blowing=100 * C["blowing"] / n,
        suction=100 * C["suction"] / n,
        inside_0p5=100 * C["inside"] / n,
    )
    out["v"].update(beyond_plus=100 * C["v_gt1"] / n, beyond_minus=100 * C["v_lt1"] / n)
    out.update(n_runs=len(rs), n_samples=int(n), n_actions=int(sum(r["n_actions"] for r in rs)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="LABEL=RUN:T_START[@POLICY]")
    p.add_argument("--t0", type=float, default=1.0, help="window start, control time t - t_start (c/U_inf)")
    p.add_argument("--t1", type=float, default=3.5, help="window end")
    p.add_argument("--max-records", type=int, default=1100, help="use every k-th recorded action, at most this many")
    p.add_argument("--unit", type=float, default=0.0624334733, help="unit of a (the bound of wing.yml)")
    p.add_argument("--alpha", type=float, default=1.0, help="opposition control gain (deploy.py --alpha)")
    p.add_argument("--ub-velocity", type=float, default=1e-3, help="uniform blowing velocity (deploy.py)")
    p.add_argument("--numpy", action="store_true", help="released policies: the numpy loader, not the SB3 zip")
    p.add_argument("--out", default="action_hist")
    args = p.parse_args()

    runs, labels = {}, {}
    for s in args.specs:
        label, rest = s.split("=", 1)
        rest, _, policy = rest.partition("@")
        run, t_start = rest.rsplit(":", 1)
        runs[run] = run_one(args, run, float(t_start), policy or None)
        labels.setdefault(label, []).append(run)
    res = {
        lab: dict(runs=rs, pooled=statistics([runs[r] for r in rs]), per_run={r: statistics([runs[r]]) for r in rs})
        for lab, rs in labels.items()
    }

    head = "| controller | at bound (%) | blowing (%) | suction (%) | abs(a) < 0.5 (%) | kurtosis | v_w,rms/u_tau |"
    lines = [
        f"Window {args.t0} <= t_c < {args.t1} c/U_inf; a in units of {args.unit:g} U_inf, at the bound: abs(a) >= 0.95;"
        " v_w after ZNMF, divided by the local u_tau(x)",
        "",
        head + " beyond +u_tau (%) | beyond -u_tau (%) | n |",
        "|" + "---|" * 10,
    ]
    for lab, r in res.items():
        a, v = r["pooled"]["a"], r["pooled"]["v"]
        lines.append(
            f"| {lab} | {a['at_bound']:.1f} | {a['blowing']:.1f} | {a['suction']:.1f} | {a['inside_0p5']:.1f} | "
            f"{a['kurtosis']:.2f} | {v['rms']:.2f} | {v['beyond_plus']:.1f} | {v['beyond_minus']:.1f} | "
            f"{r['pooled']['n_runs']} |"
        )
    md = "\n".join(lines) + "\n"
    with open(f"{args.out}.md", "w") as f:
        f.write(md)
    with open(f"{args.out}.json", "w") as f:
        json.dump(
            dict(
                window=[args.t0, args.t1],
                unit=args.unit,
                bins=dict(a=EA.tolist(), v=EV.tolist()),
                controllers=res,
                runs=runs,
            ),
            f,
        )
    print(md)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{args.out}.json, .md written (no matplotlib: no figure)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for lab, rs in labels.items():
        for k, E, a_ in (("a", EA, ax[0]), ("v", EV, ax[1])):
            h = np.sum([runs[r]["hist"][k] for r in rs], 0)
            d = h / h.sum() / np.diff(E)
            a_.stairs(np.where(d > 0, d, np.nan), E, label=lab)
    ax[0].set(
        xlabel=f"policy output a (units of {args.unit:g} U)",
        ylabel="probability density",
        yscale="log",
        xlim=(-1.1, 1.1),
    )
    ax[1].set(
        xlabel="applied wall velocity v_w / u_tau(x), after ZNMF",
        ylabel="probability density",
        yscale="log",
        xlim=(-3, 3),
    )
    for x in (-1, 1):
        ax[0].axvline(x, color="0.6", lw=0.6, ls=":")
        ax[1].axvline(x, color="0.6", lw=0.6, ls=":")
    labs = list(res)
    y = np.arange(len(labs))
    for k, (key, name) in enumerate(
        (("at_bound", "abs(a) >= 0.95"), ("beyond_plus", "v_w > +u_tau"), ("beyond_minus", "v_w < -u_tau"))
    ):
        part = "a" if key == "at_bound" else "v"
        ax[2].barh(y + 0.27 * (k - 1), [res[lab]["pooled"][part][key] for lab in labs], 0.27, label=name)
    ax[2].set_yticks(y, labs)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("share of the samples (%)")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"{args.out}.json, .md, .png written")


if __name__ == "__main__":
    main()
