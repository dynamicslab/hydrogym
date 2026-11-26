"""
Is the wing in the training distribution of the channel policies? From a deploy.py run (walls_*.npz with the
recorded "obs", agents.npz) and channel samples (client.py --record-every, obs_samples.npz):

  1. wall units: the actual suction-side u_tau(x) of the run (span- and time-averaged wall traction) against the
     u_tau polynomial of wing.yml that scales the observations and sets the probe height (y+ = 15), and the action
     amplitude in actual wall units; the control interval dt+ = (action interval) u_tau^2 / nu;
  2. the observations per x/c bin (mean, rms, skewness, u-v correlation, distance to the channel distribution),
     as the policy sees them and rescaled with the actual u_tau.

    python obs_shift.py runs/zero_trip60 --channel ../channel/runs/obs_zero ../channel/runs/obs_td3_s0 --skip 500

Only the samples after --skip actions are used. Without recorded obs only part 1 is done.
"""

import argparse
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--channel", nargs="*", default=[], help="run dirs with obs_samples.npz (client.py --record-every)")
p.add_argument("--skip", type=int, default=0, help="discard the samples of the first actions (transient)")
p.add_argument("--conf", default="wing.yml")
p.add_argument(
    "--bins", type=float, nargs="+", default=list(np.round(np.arange(0.25, 0.8601, 0.05), 3)), help="x/c bin edges"
)
p.add_argument("--out", default=None, help="figure (default RUN/obs_shift.png)")
args = p.parse_args()
out = args.out or os.path.join(args.run, "obs_shift.png")

import yaml

cfg = yaml.safe_load(open(args.conf))
nu = float(cfg["wall_units"]["nu"])
poly = np.asarray(cfg["wall_units"]["u_tau"]["polynomial"], dtype=float)
u_tau_poly = lambda x: np.polynomial.polynomial.polyval(x, poly)
amp = cfg["actuators"].get("amplitude", 1.0)  # action 1: a velocity, or {u_tau: factor} x the polynomial
amp_at = lambda x: float(amp["u_tau"]) * u_tau_poly(x) if isinstance(amp, dict) else np.full(len(x), float(amp))

# ---------------------------------------------------------------- wing samples
geo = dict(np.load(os.path.join(args.run, "walls_geometry.npz")))
parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(args.run, "walls_*_*.npz")))]
parts = [q for q in parts if len(q["action"])]
if not parts:
    raise SystemExit(f"{args.run}: no walls samples")
action = np.concatenate([q["action"] for q in parts])
time_ = np.concatenate([q["time"] for q in parts])
keep = action > args.skip
if not keep.any():
    raise SystemExit(f"{args.run}: no samples after {args.skip} actions (last {action.max()})")
print(f"[SHIFT] {args.run}: {keep.sum()} wall samples, actions {action[keep].min()}-{action[keep].max()}")


def suction_tau(b):
    """x, wall shear stress (t . e_s, rho = 1) of the upper-surface stations of boundary b, time-averaged."""
    x, y = geo[f"x_b{b}"], geo[f"y_b{b}"]
    n = np.stack([geo[f"nx_b{b}"], geo[f"ny_b{b}"]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.stack([n[:, 1], -n[:, 0]], axis=1)
    t *= np.sign(t[:, 0] + 1e-300)[:, None]
    tr = np.concatenate([q[f"b{b}"] for q in parts])[keep].mean(axis=0)
    up = y > 0
    return x[up], (tr[:, 0] * t[:, 0] + tr[:, 1] * t[:, 1])[up]


xs, taus = zip(*(suction_tau(b) for b in (1, 2) if f"x_b{b}" in geo))
xw, tau = np.concatenate(xs), np.concatenate(taus)
o = np.argsort(xw)
xw, tau = xw[o], tau[o]
u_tau_act = np.sign(tau) * np.sqrt(np.abs(tau))
# control interval from the times of the recorded observations (end of every --walls-every-th action); runs of
# --wall-steps 0 record them with the wall samples
if all("obs_time" in q for q in parts):
    oa_ = np.concatenate([q["obs_action"] for q in parts])
    ot_ = np.concatenate([q["obs_time"] for q in parts])
    oa_, ot_ = oa_[oa_ > args.skip], ot_[oa_ > args.skip]
else:
    oa_, ot_ = action[keep], time_[keep]
da = np.diff(oa_)
if (da == 0).any():  # several wall samples per action (--wall-steps) and no observations: time per action overall
    dt_action = float((ot_[-1] - ot_[0]) / (oa_[-1] - oa_[0])) if oa_[-1] > oa_[0] else np.nan
else:
    dt_action = float(np.median(np.diff(ot_)[da > 0] / da[da > 0])) if len(da) and (da > 0).any() else np.nan

edges = np.asarray(args.bins)
centres = 0.5 * (edges[1:] + edges[:-1])
summary = {"run": args.run, "skip": args.skip, "dt_action": dt_action, "bins": []}
for lo, hi, c in zip(edges[:-1], edges[1:], centres):
    m = (xw >= lo) & (xw < hi)
    ua = float(u_tau_act[m].mean()) if m.any() else np.nan
    up = float(u_tau_poly(xw[m]).mean()) if m.any() else np.nan
    summary["bins"].append(
        {
            "x": [float(lo), float(hi)],
            "u_tau_actual": ua,
            "u_tau_poly": up,
            "ratio": ua / up,  # actual / assumed
            "probe_y_plus": 15 * ua / up,  # the probes sit at 15 nu / u_tau_poly
            "amplitude_plus": float(amp_at(xw[m]).mean()) / ua if m.any() else np.nan,  # action 1 in actual wall units
            "dt_plus_action": dt_action * ua**2 / nu,
            "reversed_flow_frac": float(np.mean(tau[m] < 0)) if m.any() else np.nan,
        }
    )


# ---------------------------------------------------------------- observations
def stats(o):
    """o: (samples, fields). Moments per field and the u-v correlation."""
    o = o.astype(np.float64)
    mu, sd = o.mean(0), o.std(0)
    sk = ((o - mu) ** 3).mean(0) / np.maximum(sd, 1e-300) ** 3
    s = {"mean": mu.tolist(), "rms": np.sqrt((o**2).mean(0)).tolist(), "std": sd.tolist(), "skew": sk.tolist()}
    if o.shape[1] > 1:
        s["corr_uv"] = float(np.corrcoef(o[:, 0], o[:, 1])[0, 1])
    return s


def wasserstein(a, b, q=np.linspace(0.005, 0.995, 199)):
    """1D Wasserstein-1 distance approximated by the quantile functions."""
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


channels = {}
for d in args.channel:
    c = dict(np.load(os.path.join(d, "obs_samples.npz")))
    co = c["obs"].reshape(-1, c["obs"].shape[-1]).astype(np.float32)
    ut_assumed = float(np.mean(c["obs_div"][:, 0]))
    ut_act = np.sqrt(float(c["nu"]) * np.asarray(c["dUdy"], dtype=float))
    ch = {
        "label": os.path.basename(os.path.normpath(d)),
        "policy": str(c["policy"]),
        "samples": int(len(c["k"])),
        "u_tau_assumed": ut_assumed,
        "u_tau_actual": float(ut_act.mean()),
        "dt_plus_action": float(c["steps_per_action"]) * float(c["dt"]) * ut_assumed**2 / float(c["nu"]),
        "obs": stats(co),
        "action_rms": float(np.sqrt(np.mean(c["action"].astype(np.float64) ** 2))),
    }
    channels[ch["label"]] = (ch, co[np.random.default_rng(0).choice(len(co), min(len(co), 2_000_000), replace=False)])
    print(f"[SHIFT] channel {ch['label']}: " + json.dumps({k: v for k, v in ch.items() if k != "obs"}))
summary["channel"] = [ch for ch, _ in channels.values()]

agents_file = os.path.join(args.run, "agents.npz")
obs_parts = [q for q in parts if "obs" in q]
if obs_parts and os.path.exists(agents_file):
    ag = dict(np.load(agents_file))
    obs = np.concatenate([q["obs"] for q in obs_parts])  # (samples, probes, fields)
    a_obs = np.concatenate([q.get("obs_action", q["action"]) for q in obs_parts])
    obs = obs[a_obs > args.skip]
    fields = [str(f) for f in ag["fields"]]
    npa = obs.shape[1] // len(ag["xyz"])  # probes per actuator
    xa = np.repeat(ag["xyz"][:, 0], npa)
    ya = np.repeat(ag["xyz"][:, 1], npa)
    up_a = np.repeat(u_tau_poly(ag["xyz"][:, 0]), npa)
    ua_a = np.interp(xa, xw, u_tau_act)
    print(f"[SHIFT] {len(obs)} obs samples of {obs.shape[1]} probes, fields {fields}")
    for b, (lo, hi) in zip(summary["bins"], zip(edges[:-1], edges[1:])):
        m = (xa >= lo) & (xa < hi) & (ya > 0)
        if not m.any():
            continue
        ob = obs[:, m, :].reshape(-1, obs.shape[2]).astype(np.float32)
        # as normalized with the actual u_tau (the probe height stays the one of the polynomial)
        resc = (obs[:, m, :].astype(np.float32) * (up_a[m] / ua_a[m])[None, :, None]).reshape(-1, obs.shape[2])
        b["agents"] = int(m.sum() // npa)
        b["obs"] = stats(ob)
        b["obs_actual_u_tau"] = stats(resc)
        for lab, (ch, co) in channels.items():
            b[f"W1_{lab}"] = [wasserstein(ob[:, i], co[:, i]) for i in range(ob.shape[1])]
            lo_q, hi_q = np.quantile(co, [0.005, 0.995], axis=0)
            b[f"outside_{lab}"] = float(np.mean(np.any((ob < lo_q) | (ob > hi_q), axis=1)))
else:
    print("[SHIFT] no recorded obs in this run: wall units only")
    fields = ["u", "v"]

for b in summary["bins"]:
    line = (
        f"x/c {b['x'][0]:.2f}-{b['x'][1]:.2f}: u_tau actual/poly {b['ratio']:.3f}, probe y+ {b['probe_y_plus']:.1f}, "
        f"dt+ {b['dt_plus_action']:.3f}"
    )
    if "obs" in b:
        line += " | obs rms " + ", ".join(f"{f} {r:.2f}" for f, r in zip(fields, b["obs"]["rms"]))
        line += " | rescaled " + ", ".join(f"{f} {r:.2f}" for f, r in zip(fields, b["obs_actual_u_tau"]["rms"]))
        line += "".join(f" | out of {lab} {b['outside_' + lab]:.1%}" for lab in channels)
    print(line)

with open(os.path.splitext(out)[0] + ".json", "w") as f:
    json.dump(summary, f, indent=1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

has_obs = any("obs" in b for b in summary["bins"])
fig, ax = plt.subplots(1, 3 if has_obs else 2, figsize=(16 if has_obs else 11, 4.2))
ax[0].plot(xw, u_tau_act, label="actual (this run)")
ax[0].plot(xw, u_tau_poly(xw), "--", label="wing.yml polynomial")
ax[0].axvspan(edges[0], edges[-1], color="0.92", zorder=0)
ax[0].set_xlim(0, 1)
ax[0].set_ylim(bottom=min(0, u_tau_act.min()))
ax[0].set_xlabel("x/c")
ax[0].set_ylabel("u_tau (suction side)")
ax[0].legend()
ax[1].plot(centres, [b["dt_plus_action"] for b in summary["bins"]], "o-", label="wing")
for ch, _ in channels.values():
    ax[1].axhline(ch["dt_plus_action"], color="k", ls=":", label=f"channel ({ch['label']})")
    break
ax[1].set_xlabel("x/c")
ax[1].set_ylabel("dt+ per action")
ax[1].legend()
if has_obs:
    bb = [b for b in summary["bins"] if "obs" in b]
    xc = [0.5 * sum(b["x"]) for b in bb]
    for i, f in enumerate(fields):
        (l,) = ax[2].plot(xc, [b["obs"]["rms"][i] for b in bb], "o-", label=f"{f}' rms (policy input)")
        ax[2].plot(
            xc,
            [b["obs_actual_u_tau"]["rms"][i] for b in bb],
            "s--",
            color=l.get_color(),
            label=f"{f}' rms (actual u_tau)",
        )
        for ch, _ in channels.values():
            ax[2].axhline(ch["obs"]["rms"][i], color=l.get_color(), ls=":", lw=1)
    ax[2].set_xlabel("x/c")
    ax[2].set_ylabel("obs rms (dotted: channel)")
    ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"[SHIFT] {out} and {os.path.splitext(out)[0]}.json written")
