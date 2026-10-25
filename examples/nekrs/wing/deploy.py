"""
Zero-shot deployment on the NACA0012 wing (paper Fig. 4): run a controller for a number of control actions and
record the wall shear stress and pressure of the whole wing for Cf(x/c), Cd,f and Cd (postprocess.py).

Controllers
  zero          no actuation (uncontrolled flow)
  oc            opposition control: wall velocity -alpha v_n' at the y+ = 15 probe of each node
  ub            uniform blowing of --ub-velocity U_inf over the control window (no zero net flux)
  td3:<zip>     a Stable-Baselines3 TD3 policy trained in the channel, applied to every node of the window
                (obs [u_t'/u_tau, v_n'/u_tau] per node, as the channel agents)

The run is chunked: every --chunk actions the field goes to chk<actions>0.f00000 and state.json records the
progress, so a run killed by the batch time limit continues with run.sh (it links start0.f00001 to the last
chunk; the frozen far field and the trip time origin stay in the run directory).

Usage (from run.sh, which starts NekRS in the run directory and passes --run-root, --nproc and --conf):
  GPU=1,2,3,4 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$PWD/../channel/runs/paper_s0/td3_final.zip
"""

import argparse
import json
import os
import time

import numpy as np

from hydrogym.nekrs import NekRSFlowEnv, NekRSInterface
from hydrogym.nekrs import layout

p = argparse.ArgumentParser()
p.add_argument("--conf", default="wing.yml")
p.add_argument("--nproc", type=int)
p.add_argument("--run-root", default="runs/test")
p.add_argument("--policy", default="zero", help="zero | oc | ub | td3:<path to the SB3 zip>")
p.add_argument("--actions", type=int, default=2500, help="control actions in total (nb_interactions)")
p.add_argument("--chunk", type=int, default=250, help="write a restart field every this many actions")
p.add_argument("--walls-every", type=int, default=5, help="record the wall data every this many actions")
p.add_argument("--alpha", type=float, default=1.0, help="opposition control gain")
p.add_argument("--ub-velocity", type=float, default=1e-3, help="uniform blowing velocity [U_inf]")
p.add_argument("--wall-bids", type=int, nargs="+", default=[1, 2], help="boundaries of the wing surface")
p.add_argument("--max-hours", type=float, default=0, help="write a chunk and stop after this wall time (0: off)")
p.add_argument("--stochastic", action="store_true", help="td3: sample instead of the deterministic action")
args = p.parse_args()



def stop(msg):
    """Exit before the env exists: NekRS is already waiting for the controller, release it (TERMN) first."""
    NekRSInterface(args.nproc).finishRun()
    raise SystemExit(msg)


state_file = os.path.join(args.run_root, "state.json")
state = {"actions": 0, "chunks": [], "policy": args.policy}
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)
    if state["policy"] != args.policy:
        stop(f"{args.run_root} was started with --policy {state['policy']}, not {args.policy}")
if state["actions"] >= args.actions:
    stop(f"{args.run_root}: all {args.actions} actions done")

model = None
if args.policy.startswith("td3:"):
    try:
        from stable_baselines3 import TD3

        model = TD3.load(args.policy[4:], device="cpu")
    except Exception as e:
        stop(f"cannot load the policy {args.policy[4:]}: {e!r}")
elif args.policy not in ("zero", "oc", "ub"):
    stop(f"unknown policy {args.policy}")

# the env logs (steps.jsonl, solver_steps.csv) go to one directory per launch
env = NekRSFlowEnv(args.conf, run_root=os.path.join(args.run_root, f"log{state['actions']:06d}"), nproc=args.nproc)
try:
    npl = len(env.probe_fields)
    iv = env.probe_fields.index("v")
    if args.policy == "ub":
        env._znmf_groups = None  # uniform blowing has a net flux
    obs, info = env.reset()
    print(f"[WING] {args.policy}: {env.n_actuators} agents, {env.steps_per_action} steps per action, "
          f"resuming after {state['actions']} actions at t = {info['time']:.6f}", flush=True)
    o = obs.reshape(-1, npl)
    print("[WING] obs rms " + ", ".join(f"{f} {np.sqrt(np.mean(o[:, i] ** 2)):.3f}" for i, f in enumerate(env.probe_fields))
          + " (channel at y+ = 15: u' ~ 2.8, v' ~ 0.6)", flush=True)

    # wing geometry once per run (same node order on every call), span-averaged per (x, y) station
    geom_file = os.path.join(args.run_root, "walls_geometry.npz")
    geo = {b: env.iface.wall_data(b, geometry=True) for b in args.wall_bids}
    if not os.path.exists(geom_file):
        out = {}
        for b, g in geo.items():
            lab = layout.group_ids(g["xyz"], ["x", "y"])
            ns = lab.max() + 1
            ws = np.bincount(lab, g["w"], ns)
            out[f"label_b{b}"] = lab
            out[f"w_b{b}"] = g["w"]
            out[f"wsum_b{b}"] = ws
            for k, name in enumerate("xyz"):
                out[f"{name}_b{b}"] = np.bincount(lab, g["w"] * g["xyz"][:, k], ns) / ws
                out[f"n{name}_b{b}"] = np.bincount(lab, g["w"] * g["normal"][:, k], ns) / ws
            out[f"span_b{b}"] = float(np.ptp(g["xyz"][:, 2]))
        np.savez(geom_file, **out)
    labels = {b: layout.group_ids(g["xyz"], ["x", "y"]) for b, g in geo.items()}

    def wall_sample():
        """Span-averaged (tx, ty, tz, p) per station and the integrated force per boundary."""
        rec = {}
        for b in args.wall_bids:
            d = env.iface.wall_data(b, geometry=False)
            w, lab = geo[b]["w"], labels[b]
            ns = lab.max() + 1
            ws = np.bincount(lab, w, ns)
            vals = np.column_stack([d["traction"], d["p"]])
            rec[b] = np.stack([np.bincount(lab, w * vals[:, k], ns) / ws for k in range(4)], axis=1)
            # force on the wing: integral of (-p n + t), n into the fluid
            rec[f"F{b}"] = np.concatenate([(w[:, None] * d["traction"]).sum(0),
                                           -(w[:, None] * d["p"][:, None] * geo[b]["normal"]).sum(0)])
        return rec

    def act(obs):
        if args.policy == "zero":
            return np.zeros(env.n_actuators)
        if args.policy == "ub":
            return np.broadcast_to(args.ub_velocity / env.amplitude, (env.n_actuators,))
        o = obs.reshape(-1, npl)
        if args.policy == "oc":
            return -args.alpha * o[:, iv] * env._obs_div[:, iv] / env.amplitude
        a, _ = model.predict(o, deterministic=not args.stochastic)
        return np.asarray(a, dtype=np.float64).reshape(-1)

    samples = {"time": [], "action": []}
    t_wall = time.time()
    k = state["actions"]
    chunk_start = k
    while k < args.actions:
        a = act(obs)
        obs, rew, term, trunc, info = env.step(a)
        k += 1
        if k % args.walls_every == 0 or k == args.actions:
            s = wall_sample()
            samples["time"].append(info["time"])
            samples["action"].append(k)
            for key, v in s.items():
                samples.setdefault(f"b{key}" if isinstance(key, int) else key, []).append(v)
        if k % 10 == 0:
            print(f"[WING] action {k}/{args.actions} t = {info['time']:.6f} reward {rew:.4f} "
                  f"cfl {info['cfl_max']:.3f} |a| {np.abs(a).max():.3f} {time.time() - t_wall:.1f} s", flush=True)
        if term:
            print(f"[WING] CFL limit reached at action {k}, stopping", flush=True)
        out_of_time = args.max_hours > 0 and time.time() - t_wall > 3600 * args.max_hours
        if k % args.chunk == 0 or k == args.actions or term or out_of_time:
            name = f"chk{k:06d}"
            t = env.iface.write_field(os.path.join(args.run_root, name))
            np.savez(os.path.join(args.run_root, f"walls_{chunk_start:06d}_{k:06d}.npz"),
                     **{key: np.asarray(v) for key, v in samples.items()})
            samples = {"time": [], "action": []}
            state["actions"] = k
            state["chunks"].append({"actions": k, "time": t, "file": f"{name}0.f00000"})
            with open(state_file + ".tmp", "w") as f:
                json.dump(state, f, indent=1)
            os.replace(state_file + ".tmp", state_file)
            print(f"[WING] chunk {name} written at t = {t:.6f}", flush=True)
            chunk_start = k
        if term or out_of_time:
            break
finally:
    env.close()
