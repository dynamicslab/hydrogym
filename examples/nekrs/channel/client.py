"""Test controller for the generic NekRS interface (YAML-defined probes, actuators and reward).

Policies (in action units, the wall velocity is action * amplitude):
  zero   a = 0
  oppo   a = -alpha * v' / amplitude   (opposition control from the probe v')
  random a ~ U(-1, 1)
  td3:<zip>  a = deterministic action of an SB3 TD3 policy (train_td3.py)

--record-every n saves the observations, actions and dUdy of every n-th action to RUN/obs_samples.npz (the channel
side of ../wing/obs_shift.py).

Run with ./run.sh, which launches  mpirun -n 1 python client.py ... : -n N nekrs-mpmd ...
"""

import argparse
import json
import time

import numpy as np

try:  # before mpi4py: HPC-X loads the NCCL of the HPC SDK first, which libtorch_cuda cannot use
    from stable_baselines3 import TD3
except ImportError:
    TD3 = None

p = argparse.ArgumentParser()
p.add_argument("--conf", default="channel_re207.yml")
p.add_argument("--nproc", type=int, required=True, help="number of NekRS ranks")
p.add_argument("--policy", default="zero", help="zero | oppo | random | td3:<path to the SB3 zip>")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--run-root", default="runs/test")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--check", choices=["error", "warn", "off"], help="override actuators.resolution_check.mode")
p.add_argument("--warmup", type=int, default=0, help="solver steps before the first observation")
p.add_argument("--set", action="append", default=[], help="override a YAML entry, e.g. actuators.znmf=none")
p.add_argument("--record-every", type=int, default=0, help="save obs/actions every n actions (0 = off)")
p.add_argument("--index", type=int, help="initial condition of the snapshot library (default: drawn)")
args = p.parse_args()
model = None
if args.policy.startswith("td3:"):
    model = TD3.load(args.policy[4:], device="cpu")
elif args.policy not in ("zero", "oppo", "random"):
    raise SystemExit(f"unknown policy {args.policy}")

import yaml

from hydrogym.nekrs import NekRSFlowEnv, load_config

cfg = load_config(args.conf)
for kv in args.set:
    key, val = kv.split("=", 1)
    *path, last = key.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(val)
if args.check:
    cfg["actuators"].setdefault("resolution_check", {})["mode"] = args.check

env = NekRSFlowEnv(cfg, run_root=args.run_root, nproc=args.nproc)
rng = np.random.default_rng(args.seed)
amp = env.amplitude
npl = len(env.probe_fields)
iv = env.probe_fields.index("v") if "v" in env.probe_fields else None
utau = env.obs_scale

if args.warmup:
    env.iface.runTimeSteps(args.warmup)
obs, _ = env.reset(options=None if args.index is None else {"index": args.index})
log = []
rec_obs, rec_act, rec_k = [], [], []
t0 = time.time()
for k in range(args.steps):
    o = obs.reshape(-1, npl)
    if args.policy == "zero":
        a = np.zeros(env.n_actuators)
    elif args.policy == "oppo":
        a = np.clip(-args.alpha * o[:, iv] * utau[iv] / amp, -1, 1)
    elif model is not None:
        a = np.asarray(model.predict(o, deterministic=True)[0], dtype=np.float64).reshape(-1)
    else:
        a = rng.uniform(-1, 1, env.n_actuators)
    if args.record_every and k % args.record_every == 0:
        rec_obs.append(o.astype(np.float16))  # the observation the action a was taken on
        rec_act.append(a.astype(np.float16))
        rec_k.append(k)
    obs, r, term, trunc, info = env.step(a)
    o = obs.reshape(-1, npl)
    rec = {
        "k": k,
        "time": info["time"],
        "reward": r,
        "dUdy": info["dUdy"],
        "cfl_max": info["cfl_max"],
        "net_flux": info["net_flux"],
        "obs_u_rms": float(np.sqrt(np.mean(o[:, 0] ** 2))),
        "obs_v_rms": float(np.sqrt(np.mean(o[:, 1] ** 2))) if npl > 1 else None,
        "obs_u_mean": float(np.mean(o[:, 0])),
        "obs_v_mean": float(np.mean(o[:, 1])) if npl > 1 else None,
        "act_rms": float(np.sqrt(np.mean(a**2))),
        "wall_s": time.time() - t0,
    }
    log.append(rec)
    print("[CLIENT] " + json.dumps(rec), flush=True)
    if term:
        break

with open(f"{args.run_root}/client_log.json", "w") as f:
    json.dump({"args": vars(args), "n_actuators": int(env.n_actuators), "log": log}, f, indent=1)
np.save(f"{args.run_root}/obs_last.npy", obs)
if rec_k:
    nu = env.iface.info["mu"] / env.iface.info["rho"]
    np.savez(
        f"{args.run_root}/obs_samples.npz",
        obs=np.array(rec_obs),
        action=np.array(rec_act),
        k=np.array(rec_k),
        dUdy=np.array([r["dUdy"] for r in log]),
        time=np.array([r["time"] for r in log]),
        fields=np.array(env.probe_fields),
        obs_div=env._obs_div,
        nu=nu,
        dt=env.iface.info["dt"],
        steps_per_action=env.steps_per_action,
        policy=args.policy,
    )
env.close()
