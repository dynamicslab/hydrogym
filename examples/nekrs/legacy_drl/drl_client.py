"""Test controller for the NekRS DRL channel: drives the unchanged hydrogym.nek.NekEnv against NekRS.

Policies (all in HydroGym action units, i.e. before rescale_actions multiplies by ctrl_max_amp = 0.063):
  zero   a = 0                        (must reproduce the uncontrolled flow)
  oppo   a = -alpha * v'/ctrl_max_amp (opposition control from the sensed v' at y+ = 15)
  random a ~ U(-1, 1)                 (exercises ZNMF and the protocol)

Run with ./run.sh, which launches  mpirun -n 1 python drl_client.py ... : -n N nekrs-mpmd ...
"""

import argparse
import json
import time

import numpy as np
from omegaconf import OmegaConf

from hydrogym.nek.env import NekEnv

p = argparse.ArgumentParser()
p.add_argument("--conf", default="../../nek/configs/pettingzoo_tcflarge_re207.yml")
p.add_argument("--nproc", type=int, required=True, help="number of NekRS ranks")
p.add_argument("--totctrl", type=int, default=5760, help="must equal casedata totctrl in tcf.par")
p.add_argument("--policy", choices=["zero", "oppo", "random"], default="zero")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--run-root", default="runs/test")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

conf = OmegaConf.load(args.conf)
conf.simulation.nproc = args.nproc
conf.simulation.TOTCTRL = args.totctrl
conf.runner.random_init = 0  # no restart-file shuffling (the Fortran RSETS does not restart anyway)
conf.runner.nb_interactions = args.steps

env = NekEnv(conf=conf, run_root=args.run_root)
rng = np.random.default_rng(args.seed)
amp = float(conf.runner.ctrl_max_amp)
npl = env.obs_per_actuator

obs, _ = env.reset()
log = []
t0 = time.time()
for k in range(args.steps):
    o = obs.reshape(-1, npl)
    if args.policy == "zero":
        a = np.zeros(env.n_actuators)
    elif args.policy == "oppo":
        # obs = v'/u_tau; target wall velocity -alpha v' -> a = -alpha v'/amp
        a = np.clip(-args.alpha * o[:, 1] * env.utau / amp, -1, 1)
    else:
        a = rng.uniform(-1, 1, env.n_actuators)
    obs, r, term, trunc, info = env.step(a.astype(np.float32))
    o = obs.reshape(-1, npl)
    rec = {
        "k": k,
        "time": float(info["time"]),
        "reward": r,
        "dUdy": float(env.baseline_dudy * (1 - r)),
        "obs_u_rms": float(np.sqrt(np.mean(o[:, 0] ** 2))),
        "obs_v_rms": float(np.sqrt(np.mean(o[:, 1] ** 2))),
        "obs_u_mean": float(np.mean(o[:, 0])),
        "obs_v_mean": float(np.mean(o[:, 1])),
        "act_rms": float(np.sqrt(np.mean(a**2))),
        "wall_s": time.time() - t0,
    }
    log.append(rec)
    print("[CLIENT] " + json.dumps(rec), flush=True)
    if term:
        break

with open(f"{args.run_root}/client_log.json", "w") as f:
    json.dump({"args": vars(args), "n_actuators": int(env.n_actuators), "log": log}, f, indent=1)
np.savez(f"{args.run_root}/actuators.npz", **{k: np.asarray(v) for k, v in env.actuator_info.items()}, obs=obs)
env.close()
