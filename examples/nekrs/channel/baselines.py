"""
Reference controllers from every initial condition of the library: one episode per snapshot index, so TD3
episode rewards can be compared with zero actuation / opposition control from the same start state.
Usage: GPU=3 NP=1 RUN=runs/base_zero SCRIPT=baselines.py ./run.sh --conf channel_paper.yml --policy zero
"""

import argparse
import json
import os
import time

import numpy as np

from hydrogym.nekrs import NekRSFlowEnv

p = argparse.ArgumentParser()
p.add_argument("--conf", default="channel_paper.yml")
p.add_argument("--nproc", type=int)
p.add_argument("--run-root", default="runs/baselines")
p.add_argument("--policy", choices=["zero", "oppo"], default="zero")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--indices", type=int, nargs="*", help="snapshot indices (default: all)")
p.add_argument("--actions", type=int, help="actions per episode (default: episode_actions)")
args = p.parse_args()

env = NekRSFlowEnv(args.conf, run_root=args.run_root, nproc=args.nproc)
try:
    npl = len(env.probe_fields)
    iv = env.probe_fields.index("v")
    n = args.actions or env.episode_actions
    indices = args.indices if args.indices else range(env.cfg["initial_conditions"]["snapshots"]["count"])
    out = open(os.path.join(args.run_root, "baselines.jsonl"), "w")
    for idx in indices:
        obs, info = env.reset(options={"index": idx})
        r, d, t0 = [], [], time.time()
        for k in range(n):
            o = obs.reshape(-1, npl)
            if args.policy == "zero":
                a = np.zeros(env.n_actuators)
            else:
                a = np.clip(-args.alpha * o[:, iv] * env.obs_scale[iv] / env.amplitude, -1, 1)
            obs, rew, term, trunc, info = env.step(a)
            r.append(float(rew))
            d.append(float(info["dUdy"]))
            if term:
                break
        m = len(r)
        rec = {"policy": args.policy, "index": idx, "actions": m, "mean_reward": float(np.mean(r)),
               "last_third_reward": float(np.mean(r[2 * m // 3 :])), "mean_dUdy": float(np.mean(d)),
               "last_third_dUdy": float(np.mean(d[2 * m // 3 :])), "reward": r, "dUdy": d,
               "wall_s": round(time.time() - t0, 1)}
        print("[BASE] " + json.dumps({k: v for k, v in rec.items() if k not in ("reward", "dUdy")}), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()
finally:
    env.close()
