#!/usr/bin/env python3
"""Plot a released-policy roll-out (``run_policy.py --store_data`` output) in the paper style.

    python plot_policy.py run.npz                    # -> run.png
    python plot_policy.py run.npz --readable         # also run_readable.png, run_actions_readable.png
    python plot_policy.py run.npz --out figs/cyl200 --ylim 0.6 1.6
    python plot_policy.py run.npz --out figs/excerpt --max-action 160   # the first 160 actions only

One policy against the uncontrolled flow from the same start state, from control onset (PNG):

- ``<out>.png``: the paper style and panel size (axes box 2.2559 x 0.9331 in), with tick labels, axis labels and a
  small legend above the panel; ``<out>`` is the first npz without ``.npz`` (``--out`` sets it);
- ``--readable`` adds ``<out>_readable.png`` (a larger version with a title: policy, environment, summary of the
  run) and ``<out>_actions_readable.png`` (the actions of the policy; several actuators in shades of the policy
  colour, from dark to light; the uncontrolled run has zero action).

Where the run file holds it (Kolmogorov), ``--readable`` also plots the window-mean turbulent kinetic energy
(``<out>_tke_readable.png``). ``run_policy.py`` makes the same figures
from memory at the end of a run. The y-range is the case's range (``policy_cases.py``), extended where the data
does not fit in it; ``--ylim`` sets it exactly.

Time axis: from control onset in convective units, except for

- the cavity: the TKE (trailing-window variance) starts one window after control onset, so time is normalised to
  0-1 over the plotted data (0 = first, 1 = last point). The TKE is normalised by the mean uncontrolled TKE over the
  plotted samples (1 = the mean of the uncontrolled curve shown). The cavity Re 4200 figure starts 37 TKE samples
  (0.7 D/U) after the first TKE sample (``tke_skip`` in ``policy_cases.py``): there the policies' first, tiny actions
  have not yet changed the TKE, and the controlled and uncontrolled curves coincide; its default roll-out is 37
  actions longer (237) so that the plotted part is as long as before, and its summary's ``episode_end_ratio`` is the
  last plotted point (end of action 237, ``episode_end``; other TKE cases: end of action 200);
- the gust: the default figure shows an excerpt of the gust passage (``excerpt`` in ``policy_cases.py``) on a
  normalised 0-1 time axis; the readable figures show the whole run in convective units.

``--max-action N`` shows the first N controlled actions instead (gust: no excerpt; TKE: normalised as for the whole
run); ``--xlim`` is in the units of the time axis. The summary numbers are always those of the whole run.

The figure settings (quantity, axis labels and limits, summary window, label and colour of the policy) come from
``policy_cases.py`` and the policy id in ``meta``; other fields of ``meta`` are not used.

Colours: uncontrolled #2E2E2E; PPO #1f77b4, GPPO #2ca02c, TD3 #9467bd, DDPG #d62728 (from the variant of the policy
id); other controllers #8C564B.

Several npz files of the same policy (e.g. the uncontrolled and the controlled roll-out run separately with
``run_policy.py --runs``) are merged.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNCONTROLLED = "#2E2E2E"


def set_publication_style(scale=1.0):
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.linewidth": 1.0 * scale,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 4 * scale,
            "xtick.minor.size": 2 * scale,
            "ytick.major.size": 4 * scale,
            "ytick.minor.size": 2 * scale,
            "xtick.major.width": 0.8 * scale,
            "xtick.minor.width": 0.6 * scale,
            "ytick.major.width": 0.8 * scale,
            "ytick.minor.width": 0.6 * scale,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "legend.numpoints": 1,
            "legend.scatterpoints": 1,
            "axes.formatter.use_mathtext": True,
            "mathtext.default": "regular",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def _pin_ticks(ax):
    """fix the current major/minor tick positions so that resizing the figure does not change them"""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
        axis.set_minor_locator(FixedLocator(axis.get_minorticklocs()))
        axis.get_offset_text().set_visible(False)


PANEL = (2.2559, 0.9331)  # size of the paper panels (the axes box), inches
PAPER_FONT = 6.5  # font size of the paper panels, pt
PAPER_SCALE = 0.35  # line widths and tick sizes of the paper panels, relative to set_publication_style(1)


def fit_ylim(lim, curves, x_lim=None):
    """the y-range lim if the data (inside x_lim) fits in it, else extended to the data with a 5 % margin (lim None:
    the data range with a 5 % margin)"""
    ys = []
    for t, y, *_ in curves:
        t, y = np.asarray(t, float).ravel(), np.asarray(y, float).ravel()
        if x_lim is not None:
            y = y[(t >= x_lim[0]) & (t <= x_lim[1])]
        ys.append(y[np.isfinite(y)])
    ys = np.concatenate(ys) if ys else np.zeros(0)
    if lim is None:  # the data range with a 5 % margin
        lo, hi = (float(ys.min()), float(ys.max())) if ys.size else (0.0, 1.0)
        pad = 0.05 * (hi - lo or 1.0)
        return lo - pad, hi + pad
    lo, hi = float(lim[0]), float(lim[1])
    if ys.size and (ys.min() < lo or ys.max() > hi or lo == hi):
        pad = 0.05 * (max(hi, ys.max()) - min(lo, ys.min()) or 1.0)
        lo, hi = (min(lo, float(ys.min()) - pad), max(hi, float(ys.max()) + pad))
    return lo, hi


def paper_panel(curves, x_lim, y_lim, path, xlabel, ylabel):
    """paper panel: the paper style and panel size (PANEL), with tick labels, axis labels and a small legend above
    the panel. curves = [(t, y, colour, label), ...]"""
    set_publication_style(PAPER_SCALE)
    rcParams.update({"font.size": PAPER_FONT, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    n = len(curves)
    ncol = n if n <= 4 else -(-n // 2)
    rows = -(-n // ncol)
    W, H = PANEL[0] + 0.65, PANEL[1] + 0.45 + 0.12 * rows
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.5 / W, 0.35 / H, PANEL[0] / W, PANEL[1] / H])
    for t, y, color, label in curves:
        ax.plot(
            t,
            y,
            color=color,
            linewidth=0.8,
            alpha=0.9,
            label=label,
        )
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5 * PAPER_SCALE)
    ax.set_axisbelow(True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(xlabel, labelpad=1.0)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(which="both", pad=1.5)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=ncol,
        fontsize=PAPER_FONT - 1,
        handlelength=1.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.savefig(f"{path}.png")
    plt.close(fig)


def readable_panel(curves, x_lim, y_lim, path, xlabel, ylabel, title, fig_size=(4.6, 2.8), legend_loc="best"):
    """a larger version with tick labels, axis labels, a small frameless legend and a title"""
    set_publication_style(1.0)
    fig, ax = plt.subplots(figsize=fig_size)
    for t, y, color, label in curves:
        ax.plot(t, y, color=color, linewidth=1.5, alpha=0.9, linestyle="-", label=label)
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        fontsize=8 if len(curves) <= 2 else 7, loc=legend_loc, handlelength=1.8, ncol=1 if len(curves) <= 2 else 2
    )
    ax.set_title(title, fontsize=6.5, loc="left")
    fig.canvas.draw()
    _pin_ticks(ax)
    plt.tight_layout(pad=0.3)
    fig.savefig(f"{path}.png")
    plt.close(fig)


def shade(color, i, n):
    """i-th of n shades of a colour (i = 0: the colour itself, then lighter), for several actuators"""
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    f = 0.0 if n == 1 else 0.6 * i / (n - 1)
    return tuple(rgb + (1.0 - rgb) * f)


def load_runs(files):
    R, meta = {}, None
    for f in files:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k == "meta":
                m = json.loads(str(z["meta"]))
                if meta is not None and m["policy"] != meta["policy"]:
                    raise SystemExit(f"{f}: policy {m['policy']} != {meta['policy']}")
                meta = meta or m
            else:
                R[k] = z[k]
    for tag in ("unc", "ctl"):
        if not any(k.startswith(tag + "_") for k in R):
            raise SystemExit(f"no '{tag}' roll-out in {files} (run_policy.py --runs unc,ctl)")
    return R, meta


def summary_text(s, quantity):
    if not s:
        return ""
    if quantity == "drag" and "DR" in s:
        t = f"DR {100 * s['DR']:.1f} % (actions {s['window'][0]}-{s['window'][1]})"
        return t + (f", $C_P$ {s['C_P']:.3g}" if "C_P" in s else "")
    if quantity == "lift":
        return f"mean |C_L - C_L0| ratio {s['dCL_ratio']:.3f}, peak {s['peak_change_pct']:+.1f} %"
    if quantity == "dissipation":
        return f"mean dissipation {s['dissipation']:.3f} (uncontrolled {s['dissipation_uncontrolled']:.3f})"
    parts = [f"{k.replace('_', ' ')} {v:.3g}" for k, v in s.items() if "ratio" in k]
    return ", ".join(parts)


NORMALISED_TIME = "Normalised time"


def normalised_time(curves):
    """curves with the time mapped linearly onto 0-1 over their data (0 = first, 1 = last point)"""
    t0 = min(float(np.asarray(t)[0]) for t, *_ in curves)
    t1 = max(float(np.asarray(t)[-1]) for t, *_ in curves)
    return [((np.asarray(t, float) - t0) / ((t1 - t0) or 1.0), y, *rest) for t, y, *rest in curves]


def plot_run(R, meta, out, readable=False, xlim=None, ylim=None, max_action=None, actions=True):
    """the figures of a run (R: arrays of a run file, meta: its meta) with the prefix out: <out>.png and, with
    readable, <out>_readable.png and <out>_actions_readable.png (actions False: without the latter); returns
    (files, summary)"""
    import policy_cases as PC

    c = PC.plot_case(meta["policy"])[1]
    label, color = PC.algorithm_family(meta["policy"])
    s = PC.summary(R, c)
    tu, yu = PC.paper_series(R, "unc", c)
    tc, yc = PC.paper_series(R, "ctl", c)
    curves = [(tu, yu, UNCONTROLLED, "Uncontrolled"), (tc, yc, color, label)]
    dt = float(R["dt_action"])
    x_run = xlim or (0.0, float(max(tu[-1], tc[-1])))  # the whole run, convective time from control onset
    if max_action and max_action >= len(yc):  # the run is not longer than the excerpt: show the whole run
        max_action = None
    if max_action and not xlim:  # excerpt: the first N actions after control onset
        x_run = (0.0, max_action * dt)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    xlabel = c.get("xlabel") or (r"Time $tU_\infty/c$" if c["quantity"] == "lift" else r"Time $tU_\infty/D$")
    role = f" ({meta['file_role']})" if meta.get("file_role") else ""
    if int(R["start"]) > 0:
        role += f", control from action {int(R['start'])}"
    if max_action:
        role += f", first {max_action} of {len(yc)} actions"
    title = f"{meta['policy']}{role}\n{meta.get('environment', '')}\n" + summary_text(s, c["quantity"])
    written = []

    # the default figure: the whole run, or the case's excerpt (gust) / the cavity TKE on a normalised time axis
    main, x_main, xlabel_main = curves, x_run, xlabel
    ex = None if max_action else PC.excerpt_indices(c, R, min(len(yu), len(yc)))
    if ex is not None:
        main = normalised_time([(t[ex[0] : ex[1] + 1], y[ex[0] : ex[1] + 1], *r) for t, y, *r in curves])
    elif c["quantity"] == "tke":
        if max_action:  # the TKE samples up to the end of action max_action
            main = [(t[t <= max_action * dt + 1e-9], y[t <= max_action * dt + 1e-9], *r) for t, y, *r in curves]
        skip = PC.tke_skip(c, min(len(y) for _, y, *_ in main))  # the first TKE samples, where both runs coincide
        if skip:
            main = [(t[skip:], y[skip:], *r) for t, y, *r in main]
        main = normalised_time(main)
    if main is not curves:
        x_main, xlabel_main = xlim or (0.0, 1.0), NORMALISED_TIME

    def panels(cur, lim, name, ylabel, x_lim, x_label, default=True, both=True, legend_room=False, loc="best"):
        """<out><name>.png (default) and, with readable (and both), <out><name>_readable.png"""
        fixed = ylim is not None and name == ""
        lim = tuple(lim) if fixed else fit_ylim(lim, cur, x_lim)  # the case range, extended if the data needs it
        if default:
            paper_panel(cur, x_lim, lim, f"{out}{name}", x_label, ylabel)
            written.append(f"{out}{name}.png")
        if readable and both:
            room = (lim[0], lim[1] + 0.4 * (lim[1] - lim[0])) if legend_room else lim  # space for the legend
            readable_panel(cur, x_lim, room, f"{out}{name}_readable", x_label, ylabel, title, legend_loc=loc)
            written.append(f"{out}{name}_readable.png")

    y_case = ylim or c["ylim"]
    if ex is not None:  # gust: the excerpt in the default figure, the whole run in the readable figure
        panels(main, y_case, "", c["ylabel"], x_main, xlabel_main, both=False)
        panels(curves, y_case, "", c["ylabel"], x_run, xlabel, default=False)
    else:
        panels(main, y_case, "", c["ylabel"], x_main, xlabel_main)
    k0 = int(R["start"])
    if readable and actions:  # the actions of the policy (normalised to [-1, 1] for m-AIA); uncontrolled = 0
        A = np.asarray(R["ctl_action"], float)[k0:]
        A = A.reshape(len(A), -1)
        ta = np.linspace(0, len(A) * dt, len(A))
        cur = [(ta, np.zeros(len(A)), UNCONTROLLED, "Uncontrolled")]
        for i in range(A.shape[1]):
            cur.append((ta, A[:, i], shade(color, i, A.shape[1]), f"{label} $a_{i + 1}$"))
        amax = float(np.abs(A).max()) if A.size else 1.0
        lim = c.get("action_ylim") or ((-1.1, 1.1) if amax > 0.5 else (-1.2 * amax - 1e-12, 1.2 * amax + 1e-12))
        panels(cur, lim, "_actions", "Action $a_i$", x_run, xlabel, False, legend_room=True, loc="upper center")
    if readable and "ctl_tke" in R and "unc_tke" in R:  # window-mean turbulent kinetic energy (Kolmogorov roll-outs)
        cur = []
        for tag, col, lab in (("unc", UNCONTROLLED, "Uncontrolled"), ("ctl", color, label)):
            y = np.asarray(R[f"{tag}_tke"], float)[k0:]
            cur.append((np.linspace(0, len(y) * dt, len(y)), y, col, lab))
        lim = c.get("tke_ylim") or (0.0, 1.05 * max(float(np.max(v[1])) for v in cur))
        panels(cur, lim, "_tke", "Turbulent kinetic energy", x_run, xlabel, default=False)
    return written, s


def plot_comparison(runs, out, readable=False):
    """Roll-outs of several policies of one case from the same start state in one figure: the uncontrolled run (of
    the first run file) and the controlled run of each policy. runs = [(meta, R), ...]. Returns the files."""
    import policy_cases as PC

    c = PC.plot_case(runs[0][0]["policy"])[1]
    curves = [(*PC.paper_series(runs[0][1], "unc", c), UNCONTROLLED, "Uncontrolled")]
    for meta, R in runs:
        label, color = PC.algorithm_family(meta["policy"])
        curves.append((*PC.paper_series(R, "ctl", c), color, label))
    x_lim = (0.0, max(float(t[-1]) for t, *_ in curves))
    lim = fit_ylim(c["ylim"], curves, x_lim)
    xlabel = c.get("xlabel") or r"Time $tU_\infty/D$"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    paper_panel(curves, x_lim, lim, out, xlabel, c["ylabel"])
    written = [f"{out}.png"]
    if readable:
        title = "Roll-outs: " + ", ".join(m["policy"] for m, _ in runs) + f"\n{runs[0][0].get('environment', '')}"
        readable_panel(curves, x_lim, lim, f"{out}_readable", xlabel, c["ylabel"], title)
        written.append(f"{out}_readable.png")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", nargs="+", help="run_policy.py output(s) of one policy")
    ap.add_argument("--out", default=None, help="output prefix (default: the first npz without .npz)")
    ap.add_argument("--readable", action="store_true", help="also the readable figures and the figure of the actions")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--no-actions", action="store_true", help="with --readable: skip the figure of the actions")
    ap.add_argument(
        "--max-action",
        type=int,
        default=None,
        help="show only the first N controlled actions (default: the whole run, gust: its excerpt; summary unchanged)",
    )
    a = ap.parse_args()
    R, meta = load_runs(a.npz)
    out = a.out or os.path.splitext(a.npz[0])[0]
    written, s = plot_run(R, meta, out, a.readable, a.xlim, a.ylim, a.max_action, not a.no_actions)
    for w in written:
        print(f"wrote {w}")
    if s:
        print("summary:", json.dumps(s))


if __name__ == "__main__":
    main()
