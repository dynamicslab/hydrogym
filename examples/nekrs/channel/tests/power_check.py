"""
Actuator-power check on the solver (run with SCRIPT=tests/power_check.py ./run.sh [--actions 6]):
  P1  random actions from snapshot 0 with power_sampling mean (one solver step per request, power after each)
      and, after every action, the wall data of the actuated boundary: the flux and maximum of the wall velocity
      returned by WALLS against the ACTUA diagnostics, the pressure of the traction and the pressure-only gathers
  P2  reset to snapshot 0, the same actions with power_sampling end (all steps of an action in one request)
      -> observations and rewards of P1 and P2 equal up to the solver tolerance of a snapshot reset (the pressure
         projection history is not part of the snapshot, see reset_test.py), and the end-of-action power of P2 equal
         to the instantaneous power of the P1 wall data to the same accuracy
Reference result on GH200 GPUs (6 actions of 13 steps, 5760 actuators): ACTUA and WALLS agree on the flux (~1e-17) and
the maximum wall velocity (to the last bit), the two pressure gathers agree to the last bit; P1 - P2: observations
4.8e-7, rewards 4.3e-10; C_P of random actions, step mean vs end of action: signed 2.5e-3 vs 2.0e-4, abs 9.5e-3 vs
7.9e-4 (the end sample misses the pressure response to the jumps of the wall velocity).
"""

import argparse
import json
import os

import numpy as np

from hydrogym.nekrs import NekRSFlowEnv

ap = argparse.ArgumentParser()
ap.add_argument("--conf", default="tests/power.yml")
ap.add_argument("--nproc", type=int)
ap.add_argument("--run-root", default="runs/power")
ap.add_argument("--actions", type=int, default=6)
args = ap.parse_args()

env = NekRSFlowEnv(args.conf, run_root=args.run_root, nproc=args.nproc)
A = np.random.default_rng(0).uniform(-1, 1, (args.actions, env.n_actuators))
bid = env._power_bids[0]
KEYS = ("actuation_power", "actuation_power_positive", "actuation_power_abs", "actuation_power_full_pressure")


def run(sampling, checks=None):
    env.power_sampling = sampling
    obs, _ = env.reset(options={"index": 0})
    out = {"obs": [obs], "reward": [], "power": [], "power_now": []}
    for a in A:
        obs, r, _, _, info = env.step(a)
        out["obs"].append(obs)
        out["reward"].append(r)
        out["power"].append([info[k] for k in KEYS])
        if checks is not None:
            d = env.iface.wall_data(bid, geometry=True, velocity=True)
            d2 = env.iface.wall_data(bid, geometry=False, velocity=True, traction=False)
            vn = np.einsum("ij,ij->i", d["u_wall"], d["normal"])
            checks.append(
                {
                    "flux_walls": float(np.sum(d["w"] * vn)),
                    "flux_actua": info["net_flux"],
                    "umax_walls": float(np.linalg.norm(d["u_wall"], axis=1).max()),
                    "umax_actua": info["max_wall_velocity"],
                    "p_gathers_max_diff": float(np.abs(d["p"] - d2["p"]).max()),
                    "u_gathers_max_diff": float(np.abs(d["u_wall"] - d2["u_wall"]).max()),
                }
            )
            pw = env._power.evaluate(d["p"], d["u_wall"])
            out["power_now"].append([pw["signed"], pw["positive"], pw["abs"], pw["signed_full_pressure"]])
    return {k: np.array(v) for k, v in out.items()}


res = {}
try:
    checks = []
    P1 = run("mean", checks)
    P2 = run("end")
    res["wall_checks"] = checks
    res["obs_max_abs_diff_mean_vs_end"] = float(np.abs(P1["obs"] - P2["obs"]).max())
    res["reward_max_abs_diff_mean_vs_end"] = float(np.abs(P1["reward"] - P2["reward"]).max())
    res["end_power_vs_instant_max_abs_diff"] = float(np.abs(P2["power"] - P1["power_now"]).max())
    res["power_mean"] = P1["power"].tolist()
    res["power_end"] = P2["power"].tolist()
    res["reward"] = P1["reward"].tolist()
    res["n_actuators"] = int(env.n_actuators)
    res["steps_per_action"] = int(env.steps_per_action)
finally:
    print("[POWER] " + json.dumps(res), flush=True)
    with open(os.path.join(args.run_root, "power_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    env.close()
