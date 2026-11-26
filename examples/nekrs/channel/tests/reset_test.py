"""
Episode-reset test (run with SCRIPT=tests/reset_test.py ./run.sh):
  T0  5 random actions from the start of the run (field file rs6tcf0.f00004, NekRS start-up)
  T1  reset to snapshot 0 (library built on the first reset), same actions
  T2  reset to snapshot 0 again                  -> T2 == T1 up to the solver tolerance
  T3  snapshot 0 read back from disk into slot 7 -> T3 == T1
  T4  reset from rs6tcf0.f00004 (frozen history) -> close to T0 (different start-up)
  random selection of the library over 4 resets
"""

import argparse
import json
import os

import numpy as np

from hydrogym.nekrs import NekRSFlowEnv

ap = argparse.ArgumentParser()
ap.add_argument("--conf", default="tests/reset.yml")
ap.add_argument("--nproc", type=int)
ap.add_argument("--run-root", default="runs/reset")
ap.add_argument("--actions", type=int, default=5)
args = ap.parse_args()

env = NekRSFlowEnv(args.conf, run_root=args.run_root, nproc=args.nproc)
rng = np.random.default_rng(0)
A = rng.uniform(-1, 1, (args.actions, env.n_actuators))


def run(obs0):
    obs, rew, dudy, cfl = [obs0], [], [], []
    for a in A:
        o, r, _, _, info = env.step(a)
        obs.append(o)
        rew.append(r)
        dudy.append(info["dUdy"])
        cfl.append(info["cfl_max"])
    return {"obs": np.array(obs), "reward": np.array(rew), "dUdy": np.array(dudy), "cfl": np.array(cfl)}


def diff(a, b):
    return {
        "obs_max_abs": float(np.abs(a["obs"] - b["obs"]).max()),
        "obs_rel_l2": float(np.linalg.norm(a["obs"] - b["obs"]) / np.linalg.norm(a["obs"])),
        "reward_max_abs": float(np.abs(a["reward"] - b["reward"]).max()),
        "dUdy_rel": float(np.abs(a["dUdy"] / b["dUdy"] - 1).max()),
    }


res = {}
try:
    T0 = run(env._observe()[0])
    obs, info = env.reset(options={"index": 0})
    T1 = run(obs)
    obs, info = env.reset(options={"index": 0})
    T2 = run(obs)
    res["snapshot_repeat"] = diff(T1, T2)

    env.iface.read_snapshot(7, env.ic_snap["save"] + "_000")
    env.iface.reset_to_snapshot(7)
    T3 = run(env._observe()[0])
    res["snapshot_disk"] = diff(T1, T3)

    env.iface.reset_from_file("rs6tcf0.f00004")
    T4 = run(env._observe()[0])
    res["file_vs_start"] = diff(T0, T4)
    res["file_reward"] = T4["reward"].tolist()
    res["start_reward"] = T0["reward"].tolist()

    idx = []
    for k in range(4):
        _, info = env.reset(seed=123 if k == 0 else None)
        idx.append(info["index"])
    res["random_indices"] = idx
    res["T1_reward"] = T1["reward"].tolist()
    res["cfl_max"] = float(max(T["cfl"].max() for T in (T0, T1, T2, T3, T4)))
finally:
    print("[RESET] " + json.dumps(res), flush=True)
    with open(os.path.join(args.run_root, "reset_test.json"), "w") as f:
        json.dump(res, f, indent=1)
    env.close()
