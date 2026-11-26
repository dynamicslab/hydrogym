"""Analysis of the PPO/GPPO runs (measures defined in README.md): learning curves, final performance,
sample-efficiency measures S1 / S2, the normalisation / gradient-term decomposition, roll-out tables and figures.
numpy / scipy / matplotlib only (no JAX).

usage: python analyse.py <dir with runs/ and eval/> <out dir for results.json + figures>
"""

import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

VARIANTS = ("ppo_mb", "ppo_ro", "gppo_b0", "gppo_b2")
LABEL = {
    "ppo_mb": "P0 PPO (minibatch norm)",
    "ppo_ro": "P1 PPO + rollout norm",
    "gppo_b0": "G0 GPPO beta=0",
    "gppo_b2": "G2 GPPO beta=2",
}
COLOR = {"ppo_mb": "#1f77b4", "ppo_ro": "#17becf", "gppo_b0": "#ff7f0e", "gppo_b2": "#d62728"}
NU, NB, NB_S1 = 100, 10000, 2000  # updates per run; bootstrap resamples (final effects, S1)
RNG = np.random.default_rng(12345)


def load_run(d):
    algo = "gppo" if d.parent.name.startswith("gppo") else "ppo"
    m = d / f"{algo}_metrics.npz"
    if m.exists():
        z = np.load(m)
        recs = {k: z[k] for k in z.files}
        return recs, True
    log = d / "train.log"
    if not log.exists():
        return None, False
    by_u = {}
    for line in log.read_text().splitlines():
        if line.startswith(("PPO {", "GPPO {")):
            r = ast.literal_eval(line.split(" ", 1)[1])
            by_u[r["update"]] = r
    if not by_u:
        return None, False
    us = sorted(by_u)
    return {k: np.array([by_u[u].get(k, np.nan) for u in us]) for k in by_u[us[-1]]}, False


def ma10(x):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[9:] = (c[10:] - c[:-10]) / 10
    return out


def first_reach(curve, thr, above):
    # tolerance 1e-9 relative: R* (mean of updates 91-100) and the MA10 curve at update 100 are the same number
    # computed in different orders; without it U_ppo_mb can come out censored (101) by round-off
    tol = 1e-9 * max(abs(thr), 1.0)
    ok = curve >= thr - tol if above else curve <= thr + tol
    idx = np.where(ok[9:])[0]
    return int(idx[0] + 10) if len(idx) else 101


def boot_ci(f, groups, nb=None):
    """f(dict variant->array of per-seed rows) -> scalar; percentile CI over seed resamples within each group."""
    vals = []
    for _ in range(NB if nb is None else nb):
        g = {k: v[RNG.integers(0, len(v), len(v))] for k, v in groups.items()}
        vals.append(f(g))
    vals = np.asarray(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def main(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"measures": "examples/jax/kolmogorov_gppo/README.md", "runs": {}}
    curves = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for d in sorted((data / "runs" / v).glob("s*")):
            recs, done = load_run(d)
            if recs is None:
                continue
            seed = int(d.name[1:])
            curves[v][seed] = recs
            res["runs"][f"{v}_s{seed}"] = {"done": done, "updates": int(len(recs["mean_reward"]))}
    complete = {v: {s: r for s, r in curves[v].items() if len(r["mean_reward"]) >= NU} for v in VARIANTS}
    arr = {
        q: {v: np.array([complete[v][s][q][:NU] for s in sorted(complete[v])]) for v in VARIANTS if complete[v]}
        for q in ("mean_reward", "mean_dissipation", "mean_control_cost")
    }
    seeds = {v: sorted(complete[v]) for v in VARIANTS}
    res["seeds"] = seeds
    if not all(len(seeds[v]) >= 2 for v in VARIANTS):
        print("not enough complete runs yet", {v: len(seeds[v]) for v in VARIANTS})
    R = arr["mean_reward"]
    D = arr["mean_dissipation"]
    fin = {q: {v: a[:, 90:100].mean(1) for v, a in arr[q].items()} for q in arr}
    res["final"] = {
        q: {
            v: {
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                "per_seed": dict(zip(map(str, seeds[v]), map(float, x))),
            }
            for v, x in fin[q].items()
        }
        for q in fin
    }
    # pooling rule (README): P1 vs G0 final reward Welch p > 0.05 -> pool
    pool = None
    if "ppo_ro" in R and "gppo_b0" in R and len(seeds["ppo_ro"]) > 1 and len(seeds["gppo_b0"]) > 1:
        p = stats.ttest_ind(fin["mean_reward"]["ppo_ro"], fin["mean_reward"]["gppo_b0"], equal_var=False).pvalue
        pool = bool(p > 0.05)
        res["pool_P1_G0"] = {"welch_p_final_reward": float(p), "pooled": pool}

    def groupN(g):
        return np.concatenate([g["ppo_ro"], g["gppo_b0"]]) if pool else g["gppo_b0"]

    # final-performance differences
    res["effects_final"] = {}
    for q in ("mean_reward", "mean_dissipation", "mean_control_cost"):
        if not all(v in fin[q] for v in VARIANTS):
            continue
        g = {v: fin[q][v] for v in VARIANTS}
        eff = {
            "total_G2_minus_P0": lambda g: g["gppo_b2"].mean() - g["ppo_mb"].mean(),
            "normalisation_N_minus_P0": lambda g: groupN(g).mean() - g["ppo_mb"].mean(),
            "gradient_G2_minus_N": lambda g: g["gppo_b2"].mean() - groupN(g).mean(),
            "P1_minus_P0": lambda g: g["ppo_ro"].mean() - g["ppo_mb"].mean(),
            "G2_minus_G0": lambda g: g["gppo_b2"].mean() - g["gppo_b0"].mean(),
            "G2_minus_P1": lambda g: g["gppo_b2"].mean() - g["ppo_ro"].mean(),
        }
        res["effects_final"][q] = {}
        for name, f in eff.items():
            a_, b_ = {
                "total_G2_minus_P0": ("gppo_b2", "ppo_mb"),
                "P1_minus_P0": ("ppo_ro", "ppo_mb"),
                "G2_minus_G0": ("gppo_b2", "gppo_b0"),
                "G2_minus_P1": ("gppo_b2", "ppo_ro"),
            }.get(name, (None, None))
            e = {"estimate": float(f(g)), "ci95_boot": boot_ci(f, g)}
            if a_:
                e["welch_p"] = float(stats.ttest_ind(g[a_], g[b_], equal_var=False).pvalue)
                common = sorted(set(seeds[a_]) & set(seeds[b_]))
                if len(common) > 1:
                    da = np.array([fin[q][a_][seeds[a_].index(s)] - fin[q][b_][seeds[b_].index(s)] for s in common])
                    e["paired_mean"] = float(da.mean())
                    e["paired_p"] = float(stats.ttest_1samp(da, 0).pvalue)
            res["effects_final"][q][name] = e
    # S1 / S2 sample-efficiency
    if all(v in R for v in VARIANTS):

        def U_reward(g, v):
            rstar = g["ppo_mb"][:, 90:100].mean()
            grp = groupN(g) if v == "N" else g[v]
            return first_reach(np.nanmean(np.array([ma10(x) for x in grp]), 0), rstar, True)

        gR = {v: R[v] for v in VARIANTS}
        s1 = {"R_star": float(R["ppo_mb"][:, 90:100].mean())}
        for v in list(VARIANTS) + ["N"]:
            s1[f"U_{v}"] = U_reward(gR, v)
        for v in ("gppo_b2", "N", "ppo_ro", "gppo_b0"):
            f = lambda g, v=v: 1 - U_reward(g, v) / U_reward(g, "ppo_mb")
            s1[f"reduction_{v}_vs_P0"] = {"estimate": float(f(gR)), "ci95_boot": boot_ci(f, gR, nb=NB_S1)}
        f = lambda g: 1 - U_reward(g, "gppo_b2") / U_reward(g, "N")
        s1["reduction_G2_vs_N"] = {"estimate": float(f(gR)), "ci95_boot": boot_ci(f, gR, nb=NB_S1)}
        res["S1_reward"] = s1
        dstar = 0.9 * float(np.mean([D[v][:, 0].mean() for v in VARIANTS]))
        s2 = {"D_star": dstar}
        for v in VARIANTS:
            ud = [first_reach(ma10(x), dstar, False) for x in D[v]]
            s2[v] = {
                "U_D_per_seed": dict(zip(map(str, seeds[v]), ud)),
                "median": float(np.median(ud)),
                "fraction_reached": float(np.mean(np.array(ud) <= 100)),
                "U_D_mean_curve": first_reach(np.nanmean([ma10(x) for x in D[v]], 0), dstar, False),
            }
        res["S2_dissipation"] = s2
    # first-update common batch, wall clock
    res["first_update"] = {
        v: {
            str(s): [float(curves[v][s]["mean_reward"][0]), float(curves[v][s]["mean_dissipation"][0])]
            for s in curves[v]
        }
        for v in VARIANTS
    }
    wall = {}
    for v in VARIANTS:
        w = [np.asarray(curves[v][s]["wall_s"])[1:] for s in curves[v] if "wall_s" in curves[v][s]]
        if w:
            w = np.concatenate(w)
            wall[v] = {"median_s_per_update": float(np.median(w)), "mean_s_per_update": float(np.mean(w))}
            gw = [np.asarray(curves[v][s]["grad_wall_s"])[1:] for s in curves[v] if "grad_wall_s" in curves[v][s]]
            if gw:
                wall[v]["median_grad_s"] = float(np.median(np.concatenate(gw)))
    res["wall_clock"] = wall
    plot_curves(curves, out)
    ev = analyse_eval(data / "eval", out)
    if ev:
        res["eval"] = ev
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {k: res[k] for k in res if k in ("pool_P1_G0", "S1_reward", "S2_dissipation", "wall_clock")}, indent=1
        )
    )
    return res


def plot_curves(curves, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, q, nm in zip(
        axs,
        ("mean_reward", "mean_dissipation", "mean_control_cost"),
        ("batch reward", "batch dissipation", "batch control cost sum a^2"),
    ):
        for v in VARIANTS:
            rows = [ma10(np.asarray(r[q])[:NU]) for r in curves[v].values() if len(r[q]) >= 10]
            if not rows:
                continue
            L = min(len(x) for x in rows)
            A = np.array([x[:L] for x in rows])
            u = np.arange(1, L + 1)
            m = np.nanmean(A, 0)
            ax.plot(u, m, color=COLOR[v], label=f"{LABEL[v]} (n={len(rows)})")
            if len(rows) > 1:
                ax.fill_between(u, np.nanmin(A, 0), np.nanmax(A, 0), color=COLOR[v], alpha=0.12)
        ax.set_xlabel("update (40 transitions each)")
        ax.set_title(f"{nm}, trailing 10-update mean")
    axs[0].legend(fontsize=7)
    fig.suptitle("Kolmogorov Re 40 preset: trained PPO / GPPO, mean over seeds, band = min-max")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=130)
    plt.close(fig)


def summarise(recs, thr):
    eps = recs["eps_snap"]
    exc = eps > thr
    per = {
        "dissipation": recs["dissipation"].mean(1),
        "reward": recs["reward"].mean(1),
        "control_cost": recs["control_cost"].mean(1),
        "tke": recs["tke"].mean(1),
        "actuation_power": recs["actuation_power"].mean(1),
        "actuation_power_abs": recs["actuation_power_abs"].mean(1),
        "forcing_power": recs["forcing_power"].mean(1),
        "extreme_rate": exc.mean((1, 2)),
    }
    s = {
        k: {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}
        for k, v in per.items()
    }
    n = exc.sum()
    s["extreme_mean_excess"] = float(np.where(exc, eps - thr, 0).sum() / n) if n else 0.0
    s["extreme_count"] = int(n)
    s["mean_action"] = recs["action"].mean((0, 1)).tolist()
    s["n_episodes"] = int(eps.shape[0])
    return s


def split(blob):
    """-> {(name, 'bank'|'reset'|'reset_sto'): recs subset}"""
    groups = {}
    for i, (nm, st) in enumerate(blob["names"]):
        key = (nm, "bank" if st.startswith("bank") else ("reset_sto" if "_k" in st else "reset"))
        groups.setdefault(key, []).append(i)
    return {k: {q: v[idx] for q, v in blob["recs"].items()} for k, idx in groups.items()}


def analyse_eval(ev, out):
    if not (ev / "unc.npz").exists():
        return None
    u = np.load(ev / "unc.npz")
    eps = u["eps_snap"]  # (traj, windows, 4)
    k0 = 125
    sample = eps[:, k0:].reshape(-1)
    mu, sd = float(sample.mean()), float(sample.std())
    thr = mu + 3 * sd
    per_traj = eps[:, k0:].reshape(eps.shape[0], -1)
    r = {
        "uncontrolled": {
            "eps_mean": mu,
            "eps_std": sd,
            "threshold": thr,
            "extreme_rate": float((sample > thr).mean()),
            "per_traj_mean": per_traj.mean(1).tolist(),
            "window_dissipation_mean": float(u["dissipation"][:, k0:].mean()),
            "window_reward_mean": float(u["reward"][:, k0:].mean()),
            "forcing_power_mean": float(u["forcing_power"][:, k0:].mean()),
            "tke_mean": float(u["tke"][:, k0:].mean()),
            "time_units": int(per_traj.size),
        }
    }
    table = {}
    for f in (
        "released",
        "meanconst",
        "grid2",
        "trained_ppo_mb",
        "trained_ppo_ro",
        "trained_gppo_b0",
        "trained_gppo_b2",
    ):
        p = ev / f"{f}.pkl"
        if not p.exists():
            continue
        blob = pickle.load(open(p, "rb"))
        for key, recs in split(blob).items():
            table[f"{key[0]}|{key[1]}"] = summarise(recs, thr)
        if f == "grid2":
            pts = blob["points"]
            bank_r = {nm: table[f"{nm}|bank"]["reward"]["mean"] for nm in {n for n, _ in blob["names"]}}
            best = max(bank_r, key=bank_r.get)
            bestd = min(bank_r, key=lambda n: table[f"{n}|bank"]["dissipation"]["mean"])
            r["best_constant"] = {
                "by_reward": best,
                "action": pts[int(best[5:])].tolist(),
                "by_dissipation": bestd,
                "action_d": pts[int(bestd[5:])].tolist(),
                "cost_gpu_s": float(blob["wall_s"])
                + sum(float(pickle.load(open(q, "rb"))["wall_s"]) for q in ev.glob("grid1_*.pkl")),
            }
    r["episodes"] = table
    # trained policies (secondary measure): per seed the bank mean / reset value, then over seeds
    rt = {}
    for st in ("bank", "reset"):
        per = {}
        for v in VARIANTS:
            keys = sorted(k for k in table if k.startswith(f"{v}_s") and k.endswith(f"|{st}"))
            if keys:
                per[v] = {
                    q: np.array([table[k][q]["mean"] for k in keys])
                    for q in ("dissipation", "reward", "control_cost", "extreme_rate")
                }
                rt.setdefault(v, {})[st] = {
                    q: {
                        "mean": float(x.mean()),
                        "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                        "per_seed": dict(zip([k.split("|")[0][len(v) + 2 :] for k in keys], map(float, x))),
                    }
                    for q, x in per[v].items()
                }
        if "gppo_b2" in per and "ppo_mb" in per:
            for q in ("dissipation", "reward", "control_cost"):
                g = {v: per[v][q] for v in per}
                f = lambda g: g["gppo_b2"].mean() - g["ppo_mb"].mean()
                rt.setdefault("G2_minus_P0", {}).setdefault(st, {})[q] = {
                    "estimate": float(f(g)),
                    "ci95_boot": boot_ci(f, g),
                    "welch_p": float(stats.ttest_ind(g["gppo_b2"], g["ppo_mb"], equal_var=False).pvalue),
                }
    if rt:
        r["trained"] = rt
    plot_eval(ev, table, r, thr, out)
    return r


def plot_eval(ev, table, r, thr, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (ev / "released.pkl").exists():
        g = split(pickle.load(open(ev / "released.pkl", "rb")))
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, (st, i) in zip(axs, (("reset", 0), ("bank", 0))):
            for nm, c in (("uncontrolled", "k"), ("ppo_s30", "#1f77b4"), ("gppo_s30", "#d62728")):
                x = g[(nm, st)]["eps_snap"][i].reshape(-1)
                ax.plot(np.arange(1, len(x) + 1), x, color=c, lw=0.8, label=nm)
            ax.axhline(thr, color="gray", ls="--", lw=0.8, label="uncontrolled mean + 3 sd")
            ax.set_ylabel("dissipation")
            ax.set_title("from the training reset state" if st == "reset" else "from bank state 0 (attractor)")
        axs[0].legend(fontsize=8)
        axs[1].set_xlabel("time units (100 actions x 4)")
        fig.tight_layout()
        fig.savefig(out / "rollout_dissipation.png", dpi=130)
        plt.close(fig)
    names = [
        n
        for n in (
            "uncontrolled",
            "ppo_s30",
            "gppo_s30",
            "ppo_s30_meanconst",
            "gppo_s30_meanconst",
            r.get("best_constant", {}).get("by_reward"),
            r.get("best_constant", {}).get("by_dissipation"),
        )
        if n and f"{n}|bank" in table
    ]
    if names:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, q in zip(axs, ("dissipation", "reward", "extreme_rate")):
            for j, st in enumerate(("bank", "reset")):
                vals = [table.get(f"{n}|{st}", {}).get(q, {"mean": np.nan, "sem": 0}) for n in names]
                ax.bar(
                    np.arange(len(names)) + 0.4 * j,
                    [v["mean"] for v in vals],
                    0.4,
                    yerr=[v["sem"] for v in vals],
                    label=st,
                )
            ax.set_xticks(np.arange(len(names)) + 0.2)
            ax.set_xticklabels([n.replace("_meanconst", "\nmean-const") for n in names], rotation=45, fontsize=7)
            ax.set_title(q)
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(out / "baselines.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
