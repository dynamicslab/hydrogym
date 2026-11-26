"""
Zero-shot deployment on the NACA0012 wing: run a controller for a number of control actions and
record the wall shear stress and pressure of the whole wing for Cf(x/c), Cd,f and Cd (postprocess.py).

Controllers
  zero          no actuation (uncontrolled flow)
  oc            opposition control: wall velocity -alpha v_n' at the y+ = 15 probe of each node
  ub            uniform blowing of --ub-velocity U_inf over the control window (no zero net flux)
  td3:<zip>     a Stable-Baselines3 TD3 policy trained in the channel, applied to every node of the window
                (obs [u_t'/u_tau, v_n'/u_tau] per node, as the channel agents)
  td3:<id>      a released policy of policies/MANIFEST.yaml, e.g. naca0012_wing_re2e5/td3/seed1998: its SB3 zip with
                Stable-Baselines3 if the entry has one and SB3 is installed, else the numpy loader of
                policies/load_policy.py (wall_mlp_npz; wall_blocks: one actor per chordwise block of agents)

Wall samples: with --wall-steps N (default 10) the span-averaged traction and pressure per station and the force on
each wing boundary are recorded after every solver step whose index is a multiple of N, also within an action (the
env then advances one solver step per request, which gives the same flow). Their plain mean is the time mean. A
policy that changes its action in steps makes the friction vary within the action, so samples taken only at the end
of each action (--wall-steps 0: only after the last step of every --walls-every-th action) are biased. The
observations of the controller are recorded at the end of every --walls-every-th action (obs, obs_action, obs_time).

Actuator power: the env reports it with every action (hydrogym.power info keys; NekRSFlowEnv power section of
the YAML): signed, positive (Hasegawa & Kasagi) and absolute (Bewley et al.) C_P with p' about the spanwise mean,
the signed full-pressure value and the whole-wing C_D, normalised with 0.5 rho U^3 c span. --power-sampling mean
(default of wing.yml) averages over every solver step of an action, end samples after its last step. The values of
every action go to the walls_*.npz of each chunk (power_action, power_time, power [signed, positive, abs, full
pressure], power_cd).

The run is chunked: every --chunk actions the field goes to chk<actions>0.f00000 and state.json records the
progress, so a run killed by the batch time limit continues with run.sh (it links start0.f00001 to the last
chunk; the frozen far field and the trip time origin stay in the run directory).

Usage (from run.sh, which starts NekRS in the run directory and passes --run-root, --nproc and --conf):
  GPU=0,1,2,3 NP=4 RUN=runs/td3 ./run.sh --policy td3:naca0012_wing_re2e5/td3/seed1998
  GPU=0,1,2,3 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$PWD/../channel/runs/td3_s0/td3_final.zip
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:  # before mpi4py: HPC-X loads the NCCL of the HPC SDK first, which libtorch_cuda cannot use
    from stable_baselines3 import TD3
except ImportError:
    TD3 = None

from hydrogym.nekrs import NekRSFlowEnv, NekRSInterface, layout

p = argparse.ArgumentParser()
p.add_argument("--conf", default="wing.yml")
p.add_argument("--nproc", type=int)
p.add_argument("--run-root", default="runs/test")
p.add_argument("--policy", default="zero", help="zero | oc | ub | td3:<path to the SB3 zip> | td3:<released policy id>")
p.add_argument("--actions", type=int, default=2500, help="control actions in total (nb_interactions)")
p.add_argument("--chunk", type=int, default=250, help="write a restart field every this many actions")
p.add_argument(
    "--wall-steps",
    type=int,
    default=10,
    help="record the wall data after every this many solver steps (global step index); 0: after the last step of "
    "every --walls-every-th action",
)
p.add_argument(
    "--walls-every",
    type=int,
    default=5,
    help="record the observations every this many actions (with --wall-steps 0 also the wall data)",
)
p.add_argument("--alpha", type=float, default=1.0, help="opposition control gain")
p.add_argument("--ub-velocity", type=float, default=1e-3, help="uniform blowing velocity [U_inf]")
p.add_argument("--wall-bids", type=int, nargs="+", default=[1, 2], help="boundaries of the wing surface")
p.add_argument("--max-hours", type=float, default=0, help="write a chunk and stop after this wall time (0: off)")
p.add_argument("--stochastic", action="store_true", help="td3: sample instead of the deterministic action")
p.add_argument("--no-obs", action="store_true", help="do not record the observations")
p.add_argument(
    "--power-sampling",
    choices=["mean", "end"],
    help="actuator power: every solver step | end of the action (default: --conf, else mean)",
)
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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
model = None
if args.policy.startswith("td3:") and not os.path.exists(args.policy[4:]):
    # a released policy id (policies/MANIFEST.yaml)
    sys.path.insert(0, REPO)
    from policies.load_policy import entry, load_policy

    try:
        _e = entry(args.policy[4:])
    except KeyError as e:
        stop(f"{args.policy[4:]} is neither a file nor a released policy id: {e}")
    _zips = [f["path"] for f in _e["files"] if f["role"] == "weights_sb3"]
    if _zips and TD3 is not None:
        policy_file = os.path.join(REPO, "policies", _zips[0])  # the deployed SB3 file, run with SB3
    else:
        try:
            model = load_policy(_e["id"])
        except Exception as e:
            stop(f"cannot load the policy {_e['id']}: {e!r}")
        if model.format not in ("wall_mlp_npz", "wall_blocks"):
            stop(f"{_e['id']}: format {model.format} is not a wall actuator policy")
        print(f"[WING] {_e['id']}: numpy loader ({model.format})", flush=True)
elif args.policy.startswith("td3:"):
    policy_file = args.policy[4:]
if args.policy.startswith("td3:") and model is None:
    try:
        # policy zips of older SB3 / gym versions load with gymnasium spaces in place of the pickled ones
        import gymnasium

        _co = {
            "observation_space": gymnasium.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "action_space": gymnasium.spaces.Box(-1, 1, (1,), np.float32),
            "lr_schedule": lambda _: 0.0,
            "learning_rate": 0.0,
            "replay_buffer_class": None,
        }
        model = TD3.load(policy_file, device="cpu", custom_objects=_co)
        print(f"[WING] SB3 policy {policy_file}", flush=True)
    except Exception as e:
        stop(f"cannot load the policy {policy_file}: {e!r}")
elif not args.policy.startswith("td3:") and args.policy not in ("zero", "oc", "ub"):
    stop(f"unknown policy {args.policy}")

# the env logs (steps.jsonl, solver_steps.csv) go to one directory per launch
env = NekRSFlowEnv(
    args.conf,
    run_root=os.path.join(args.run_root, f"log{state['actions']:06d}"),
    nproc=args.nproc,
    power_sampling=args.power_sampling,
)
try:
    npl = len(env.probe_fields)
    iv = env.probe_fields.index("v")
    if args.policy == "ub":
        env._znmf_groups = None  # uniform blowing has a net flux
    obs, info = env.reset()
    print(
        f"[WING] {args.policy}: {env.n_actuators} agents, {env.steps_per_action} steps per action, "
        f"resuming after {state['actions']} actions at t = {info['time']:.6f}",
        flush=True,
    )
    o = obs.reshape(-1, npl)
    print(
        "[WING] obs rms "
        + ", ".join(f"{f} {np.sqrt(np.mean(o[:, i] ** 2)):.3f}" for i, f in enumerate(env.probe_fields))
        + " (channel at y+ = 15: u' ~ 2.8, v' ~ 0.6)",
        flush=True,
    )

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
    # the agents of the observations recorded with the wall samples (obs_shift.py)
    agents_file = os.path.join(args.run_root, "agents.npz")
    if not args.no_obs and not os.path.exists(agents_file):
        np.savez(
            agents_file,
            xyz=env.actuators["xyz"],
            probe_xyz=env.probes["xyz"],
            obs_div=env._obs_div,
            amplitude=np.broadcast_to(env.amplitude, (env.n_actuators,)),
            fields=np.array(env.probe_fields),
        )

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
            rec[f"F{b}"] = np.concatenate(
                [(w[:, None] * d["traction"]).sum(0), -(w[:, None] * d["p"][:, None] * geo[b]["normal"]).sum(0)]
            )
        return rec

    def act(obs):
        if args.policy == "zero":
            return np.zeros(env.n_actuators)
        if args.policy == "ub":
            return np.broadcast_to(args.ub_velocity / env.amplitude, (env.n_actuators,))
        o = obs.reshape(-1, npl)
        if args.policy == "oc":
            return -args.alpha * o[:, iv] * env._obs_div[:, iv] / env.amplitude
        if getattr(model, "format", None) == "wall_blocks":  # one actor per chordwise block of agents
            return model(o, x_agents)
        if getattr(model, "format", None) == "wall_mlp_npz":
            return model(o)
        a, _ = model.predict(o, deterministic=not args.stochastic)
        return np.asarray(a, dtype=np.float64).reshape(-1)

    x_agents = np.asarray(env.actuators["xyz"], float)[:, 0]
    samples = {"time": [], "step": [], "action": []}
    POWER_KEYS = ("actuation_power", "actuation_power_positive", "actuation_power_abs", "actuation_power_full_pressure")
    power = {"power_action": [], "power_time": [], "power": [], "power_cd": []}
    t_wall = time.time()
    prof = {} if env.iface.profile is not None else None  # HGNEK_PROFILE: seconds per action of the parts below

    def tick(name, t0):
        if prof is not None:
            prof[name] = prof.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    def on_step(_env, row):
        """step hook (--wall-steps): wall sample after a solver step of the running action k + 1"""
        s = wall_sample()
        samples["time"].append(float(row[1]))
        samples["step"].append(int(row[0]))
        samples["action"].append(k + 1)
        for key, v in s.items():
            samples.setdefault(f"b{key}" if isinstance(key, int) else key, []).append(v)

    if args.wall_steps > 0:
        env.set_step_hook(args.wall_steps, on_step)
        print(f"[WING] wall samples every {args.wall_steps} solver steps", flush=True)

    k = state["actions"]
    chunk_start = k
    while k < args.actions:
        tp = time.perf_counter()
        a = act(obs)
        tp = tick("policy", tp)
        obs, rew, term, trunc, info = env.step(a)
        tp = tick("env.step", tp)
        k += 1
        if "actuation_power" in info:
            power["power_action"].append(k)
            power["power_time"].append(info["time"])
            power["power"].append([info[key] for key in POWER_KEYS])
            power["power_cd"].append(info.get("drag_coefficient", np.nan))
        if k % args.walls_every == 0 or k == args.actions:
            if args.wall_steps == 0:  # wall sample after the last solver step of the action, obs aligned with it
                s = wall_sample()
                tp = tick("wall_sample", tp)
                samples["time"].append(info["time"])
                samples["step"].append(env.tstep)
                samples["action"].append(k)
                for key, v in s.items():
                    samples.setdefault(f"b{key}" if isinstance(key, int) else key, []).append(v)
            elif not args.no_obs:
                samples.setdefault("obs_action", []).append(k)
                samples.setdefault("obs_time", []).append(info["time"])
            if not args.no_obs:
                samples.setdefault("obs", []).append(obs.reshape(-1, npl).astype(np.float16))
        if prof is not None and k % 10 == 0:
            req = env.iface.profile
            print(
                "[WING] profile per action over the last 10: "
                + ", ".join(f"{n} {v / 10:.3f} s" for n, v in prof.items())
                + " | requests: "
                + ", ".join(f"{c} {v[1] / 10:.3f} s ({v[0]})" for c, v in req.items()),
                flush=True,
            )
            prof.clear()
            req.clear()
        if k % 10 == 0:
            pw = (
                f" C_D {info['drag_coefficient']:.5f} C_P {info['actuation_power']:.3e} / "
                f"{info['actuation_power_positive']:.3e} / {info['actuation_power_abs']:.3e} ({info['power_sampling']})"
                if "drag_coefficient" in info
                else ""
            )
            print(
                f"[WING] action {k}/{args.actions} t = {info['time']:.6f} reward {rew:.4f} "
                f"cfl {info['cfl_max']:.3f} |a| {np.abs(a).max():.3f}{pw} {time.time() - t_wall:.1f} s",
                flush=True,
            )
        if term:
            print(f"[WING] CFL limit reached at action {k}, stopping", flush=True)
        out_of_time = args.max_hours > 0 and time.time() - t_wall > 3600 * args.max_hours
        if k % args.chunk == 0 or k == args.actions or term or out_of_time:
            name = f"chk{k:06d}"
            t = env.iface.write_field(os.path.join(args.run_root, name))
            np.savez(
                os.path.join(args.run_root, f"walls_{chunk_start:06d}_{k:06d}.npz"),
                **{key: np.asarray(v) for key, v in samples.items()},
                **({key: np.asarray(v, float) for key, v in power.items()} if power["power"] else {}),
            )
            samples = {"time": [], "step": [], "action": []}
            power = {key: [] for key in power}
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
