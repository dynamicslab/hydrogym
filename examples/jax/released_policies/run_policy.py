#!/usr/bin/env python3
"""Roll out the released Kolmogorov policies (PPO and GPPO) and the uncontrolled flow in the HydroGym JAX solver.

    python run_policy.py                          # -> kolmogorov_rollout_ppo_vs_gppo.png
    python run_policy.py --readable --store_data  # also the readable and per-policy figures, npz and json files
    python run_policy.py --policy kolmogorov_re40/gppo/seed30 --actions 20 --out runs/gppo

Default figure (PNG, paper style): ``kolmogorov_rollout_ppo_vs_gppo.png``, the roll-outs, window-mean dissipation
against time, of the uncontrolled flow and of each policy from the same start state.

Roll-outs: the preset ``KolmogorovFlow.extreme_events_re40()`` (Re 40, 64 x 64, 4 time units per action, actions in
[-0.4, 0.15]), the environment of the released policies; the uncontrolled flow and each policy from the
deterministic reset state of the preset. The policy network and its loading are those of
``examples/jax/kolmogorov_gppo`` (``ppo.ActorCritic``, ``evaluate.load_params`` on ``params.msgpack``); the
deterministic action is ``mid + half * tanh(mean)``, as in ``evaluate.py``. The numpy forward pass of
``policies/load_policy.py`` is evaluated alongside as a check. A summary per policy is printed.

``--readable`` adds the larger version (``kolmogorov_rollout_ppo_vs_gppo_readable.png``) and, per policy, the
roll-out figures against the uncontrolled flow: ``kolmogorov_ppo.png`` and ``kolmogorov_ppo_readable.png``
(window-mean dissipation), ``kolmogorov_ppo_actions_readable.png`` (actions) and
``kolmogorov_ppo_tke_readable.png`` (window-mean turbulent kinetic energy). ``--store_data`` writes, per policy,
``kolmogorov_ppo.npz`` and ``kolmogorov_ppo_policy_run.json``: per action and roll-out (``unc_`` / ``ctl_``) the
action, the window-mean dissipation, the window-mean turbulent kinetic energy, the control cost sum(a_i^2) and the
reward; ``meta`` (JSON) the policy, environment, dimension, action count, action interval and the physical settings.
CPU is enough (about 2 s per action and roll-out). Keep ``jax_enable_x64`` off (float32).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXAMPLES))
POLICY_DIR = os.path.join(REPO, "policies")
if not os.path.isfile(os.path.join(POLICY_DIR, "load_policy.py")):
    # examples copied out of the source tree: the policies/ of the installed (editable) HydroGym
    import hydrogym

    POLICY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(hydrogym.__file__))), "policies")
sys.path.insert(0, POLICY_DIR)
sys.path.insert(0, os.path.join(EXAMPLES, "kolmogorov_gppo"))
sys.path.insert(0, os.path.join(os.path.dirname(EXAMPLES), "maia", "released_policies"))  # figure settings, plotting

import policy_cases as PC  # noqa: E402
from load_policy import _file, entry, load_policy  # noqa: E402

ENVIRONMENT = "KolmogorovFlow.extreme_events_re40()"


def run_meta(pid, env, p, n):
    """meta of the npz: policy, environment, dimension, protocol and the physical settings (no plot styling)"""
    return dict(
        policy=pid,
        environment=ENVIRONMENT,
        dim=2,
        n_actions=int(n),
        dt_action=float(p.action_time),
        Re=float(env.flow.Re),
        min_action=float(p.min_action),
        max_action=float(p.max_action),
    )


def run_record(pid, n):
    """policy_run.json: policy, environment, dimension and the settings of the run"""
    return dict(policy=pid, environment=ENVIRONMENT, dim=2, settings=dict(actions=int(n)))


POLICIES = ("kolmogorov_re40/ppo/seed30", "kolmogorov_re40/gppo/seed30")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--policy", action="append", default=None, help="policy id (repeat for several; default: PPO and GPPO)"
    )
    ap.add_argument("--actions", type=int, default=100, help="actions per roll-out (one episode = 100)")
    ap.add_argument("--out", default="kolmogorov", help="prefix of the output files")
    ap.add_argument("--readable", action="store_true", help="also the readable and per-policy roll-out figures")
    ap.add_argument("--store_data", action="store_true", help="also write the npz and policy_run.json of each policy")
    a = ap.parse_args()
    import evaluate as EV  # examples/jax/kolmogorov_gppo: preset, reset, network and loading of the package
    import jax
    import jax.numpy as jnp
    print(f"JAX devices: {jax.devices()}, x64: {jax.config.jax_enable_x64}", flush=True)  # a CPU fallback is silent otherwise
    import plot_policy  # matplotlib: fail before the roll-outs, not after them

    pids = a.policy or list(POLICIES)
    prefix = PC.run_files(a.out)[0]
    env = EV.env()
    p = env.default_params
    net, tmpl = EV.net_and_template()
    mid, half = (p.max_action + p.min_action) / 2, (p.max_action - p.min_action) / 2
    obs0, state0 = EV.reset_state()

    def run(params, controlled):
        def body(carry, _):
            o, s = carry
            dist, _ = net.apply(params, o)
            act = mid + half * jnp.tanh(dist.mean()) if controlled else jnp.zeros(4)
            o2, s2, rec = EV.step_record(env, p, s, act)
            rec = {k: rec[k] for k in ("action", "dissipation", "tke", "reward", "control_cost")}
            rec["obs"] = o
            return (o2, s2), rec

        _, recs = jax.lax.scan(body, (obs0, state0), None, length=a.actions)
        return recs

    run_jit = jax.jit(run, static_argnums=1)

    def rollout(tag, params, controlled):
        t0 = time.time()
        rec = {k: np.asarray(v, np.float64) for k, v in run_jit(params, controlled).items()}
        print(
            f"[{tag}] {a.actions} actions in {time.time() - t0:.1f} s, mean dissipation {rec['dissipation'].mean():.4f}"
        )
        return rec

    unc = rollout("uncontrolled", tmpl, False)
    unc.pop("obs")
    written, runs = [], []
    for pid in pids:
        variant = pid.split("/")[1]
        rec = rollout(variant, EV.load_params(_file(entry(pid), "weights_flax"), tmpl), True)  # params.msgpack
        obs = rec.pop("obs")
        R = dict(start=0, n_actions=a.actions, dt_action=float(p.action_time))
        R.update({f"unc_{k}": v for k, v in unc.items()})
        R.update({f"ctl_{k}": v for k, v in rec.items()})
        # check: the numpy forward pass of the policy package on the observations of the controlled roll-out
        pi = load_policy(pid)
        d_np = float(np.abs(np.stack([pi(o) for o in obs]) - R["ctl_action"]).max())
        meta = run_meta(pid, env, p, a.actions)
        runs.append((meta, R))
        if a.store_data:
            written += PC.write_run_files(f"{prefix}_{variant}.npz", R, meta, run_record(pid, a.actions))
        if a.readable:
            written += plot_policy.plot_run(R, meta, f"{prefix}_{variant}", readable=True)[0]
        d, du = float(R["ctl_dissipation"].mean()), float(R["unc_dissipation"].mean())
        summary = dict(
            dissipation=d,
            dissipation_uncontrolled=du,
            dissipation_ratio=d / du,
            tke=float(R["ctl_tke"].mean()),
            control_cost=float(R["ctl_control_cost"].mean()),
            reward=float(R["ctl_reward"].mean()),
            mean_action=R["ctl_action"].mean(0).tolist(),
            numpy_loader_max_abs_diff=d_np,
        )
        print(f"summary {pid}:", json.dumps(summary))
    name = "_vs_".join(pid.split("/")[1] for pid in pids)
    written += plot_policy.plot_comparison(runs, f"{prefix}_rollout_{name}", readable=a.readable)
    for f in written:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
