"""
Actuation power and net savings of controlled wing runs (drag reduction against actuation power), from the
walls_*.npz of deploy.py: the power of every action that the env reports (power section of the YAML; signed,
positive (Hasegawa & Kasagi 2011) and absolute (Bewley et al. 2001) forms with p' about the spanwise mean, and the
signed form with the full pressure) and the drag of fig4d.py, over a window of control time, paired with an
uncontrolled run.

Per form k (C_P,k = P_k / (0.5 rho U_inf^3 c span), the normalisation of C_D):
  P_k / (U_inf F_f0)  power in per cent of the uncontrolled friction power of the suction-side strip
                      0.25 <= x/c <= 0.86 (F_f0: its streamwise friction force);
  DR25 = 100 (1 - F_f / F_f0)                  friction reduction on the strip (per cent);
  S25_k = DR25 - P_k / (U_inf F_f0)            net saving on the strip (per cent);
  S_k = 100 (1 - (C_D + C_P,k) / C_D0)         net saving of the whole wing (per cent), C_D of the force monitors.

The sampling of the power is the one of the run (deploy.py --power-sampling: mean = every solver step of an action,
end = after its last step); run a controller once per sampling to report both.

    python power_forms.py --ref runs/zero --t0 1 --t1 3.5 --out pareto \
        TD3=runs/td3:28 TD3=runs/td3_r2:33.2 N1=runs/n1:28 N1=runs/n1_r2:33.2 OC=runs/oc:28

Specs, --ref and the window as in fig4d.py. Writes OUT.json and OUT.md, and OUT.png if matplotlib is installed.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4d import Wing, common_args, evaluate, group_tables, jsonable, load_walls, markdown, pair  # noqa: E402

FORMS = ("signed", "positive", "abs", "full_pressure")  # columns of the env's power record


def power_values(args, wing, ref):
    ref_power = ref.get("power")

    def extra(c, ref_, i, r, t_start):
        if "power" not in c:
            raise SystemExit("a run without power samples (power section of the YAML)")
        pw = c["power"]
        tc = pw["t"] - t_start
        m = (tc >= args.t0 - 1e-9) & (tc <= args.t1 + 1e-9)
        cp = pw["power"][m].mean(0)
        cd = float(pw["cd"][m].mean())
        if ref_power is not None:  # C_D0 of the force monitors at the same times
            j, ok = pair(ref_power["t"], pw["t"][m], args.pair_tol)
            cd0 = float(ref_power["cd"][j[ok]].mean())
        else:
            cd0 = float(ref_["cd_forces"][r].mean())
        ff0 = ref_["ff25"][r].mean()
        dr25 = 100 * (1 - c["ff25"][i].mean() / ff0)
        e = dict(DR25=float(dr25), n_power=int(m.sum()), CD_monitor=cd, CD0_monitor=cd0)
        for k, form in enumerate(FORMS):
            e[f"CP_{form}"] = float(cp[k])
            e[f"P25_{form}"] = float(100 * cp[k] * 0.5 * args.power_span / ff0)
            e[f"S25_{form}"] = float(dr25 - e[f"P25_{form}"])
            e[f"S_{form}"] = float(100 * (1 - (cd + cp[k]) / cd0))
        return e

    return extra


def main():
    p = common_args(__doc__)
    p.add_argument("--power-span", type=float, default=0.1, help="span of the power reference (power.reference.span)")
    p.set_defaults(out="power_forms")
    args = p.parse_args()

    wing = Wing(args.ref[0])
    ref = load_walls(args.ref, wing)
    runs, labels = evaluate(args, wing, ref, extra=power_values(args, wing, ref))
    keys = ["DR25", "dCD"] + [f"P25_{f}" for f in FORMS[:3]] + [f"S25_{f}" for f in FORMS[:3]]
    keys += [f"S_{f}" for f in FORMS[:3]]
    stats_ = group_tables(runs, labels, args.diff, keys)
    md = markdown(
        stats_,
        keys,
        f"Window {args.t0} <= t_c <= {args.t1} c/U_inf; per cent; P25: power / (U_inf F_f0) of the strip "
        "0.25-0.86; S25 = DR25 - P25; S: net saving of the whole wing",
    )
    with open(f"{args.out}.json", "w") as f:
        json.dump(dict(window=[args.t0, args.t1], reference=args.ref, groups=stats_, runs=jsonable(runs)), f, indent=1)
    with open(f"{args.out}.md", "w") as f:
        f.write(md)
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
        pa = [runs[r]["P25_abs"] for r in rs]
        for a, key in ((ax[0], "dCD"), (ax[1], "DR25")):
            v = [runs[r][key] for r in rs]
            (h,) = a.plot(np.mean(pa), np.mean(v), "o", label=lab)
            a.plot(pa, v, ".", color=h.get_color(), alpha=0.5)
    labs = list(labels)
    for k, form in enumerate(("signed", "abs")):
        v = [np.mean([runs[r][f"S25_{form}"] for r in labels[lab]]) for lab in labs]
        ax[2].bar(np.arange(len(labs)) + 0.4 * (k - 0.5), v, 0.4, label=form)
    ax[0].set(xlabel="P_abs / (U F_f0) (%)", ylabel="wing C_D reduction (%)")
    ax[1].set(xlabel="P_abs / (U F_f0) (%)", ylabel="DR25 (%)")
    ax[2].set_xticks(np.arange(len(labs)), labs, rotation=20, fontsize=8)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("net saving S25 (%)")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"{args.out}.json, .md, .png written")


if __name__ == "__main__":
    main()
