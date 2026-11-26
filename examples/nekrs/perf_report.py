"""
Runtime and performance numbers of NekRS runs (channel training, wing deployment) from their logs:

    python perf_report.py wing/runs/zero_trip60            # all *.log of a run (one per job)
    python perf_report.py channel/runs/td3_s1 --json perf.json
    python perf_report.py wing/runs/zero_trip60/deploy-<jobid>.log channel/runs/obs_zero/nekrs_sim0.log

Per log: case size (elements, N, DOFs, ranks, backend), setup (JIT, initialization), NekRS step time, solver
iterations, CFL, the share of the solve spent in the udf (probes, actuation, the exchange with Python), GPU
throughput and memory, the Ax kernel benchmark of the setup (fp64); with the Python lines also the time per action
and its overhead outside NekRS ([WING], [CLIENT]), per training the episodes ([TD3]), and the GPU use, memory, power
and temperature sampled by the sbatch scripts (gpu-JOB.log, rocm-smi). Cost figures are
GPU hours with one GPU (APU, GCD) per NekRS rank; DOFs are E N^3.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("paths", nargs="+", help="log files or run directories (all *.log, also in subdirectories)")
p.add_argument(
    "--skip-steps", type=int, default=20, help="steps at the start of each log left out of the step statistics"
)
p.add_argument("--json", help="write all numbers to this file")
args = p.parse_args()

NUM = r"([-+0-9.eE]+)"


def floats(pattern, text):
    return np.array([float(m) for m in re.findall(pattern, text)])


def stats(v):
    return (
        {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min()),
            "max": float(v.max()),
            "n": int(len(v)),
        }
        if len(v)
        else None
    )


def parse(path):
    text = open(path, errors="replace").read()
    r = {"log": path}
    one = lambda pat, f=float: (lambda m: f(m.group(1)) if m else None)(re.search(pat, text))
    r["ranks"] = one(r"MPI tasks: (\d+)", int)
    r["backend"] = one(r"active occa mode: (\w+)", str)
    r["elements"] = one(r"Nelements: (\d+)", int)
    r["N"] = one(r"polynomial order N: (\d+)", int)
    m = re.search(r"JIT compiling kernels[^\n]*\n\s*done \(" + NUM + r"s\)", text)
    r["jit_s"] = float(m.group(1)) if m else None
    r["init_s"] = one(r"initialization took " + NUM + " s")
    mem = floats(r"occa max memory usage:\s+(\d+) bytes", text)
    r["device_memory_GB_rank0"] = float(mem.max() / 1e9) if len(mem) else None
    m = re.search(r"Ax: N=(\d+) wordSize=64 wordSizeGeo=64 GDOF/s=" + NUM + " GB/s=" + NUM + " GFLOPS/s=" + NUM, text)
    if m:
        r["Ax_kernel_fp64"] = {
            "N": int(m.group(1)),
            "GDOF/s": float(m.group(2)),
            "GB/s": float(m.group(3)),
            "GFLOP/s": float(m.group(4)),
        }
    if r["elements"] and r["N"]:
        r["dofs"] = r["elements"] * r["N"] ** 3

    # per time step
    tdc = re.findall(r"step=\s*\d+\s+t=\s*" + NUM + r"\s+dt=\s*" + NUM + r"\s+CFL=\s*" + NUM, text)
    el = floats(r"elapsedStep=\s*" + NUM + "s", text)
    k = args.skip_steps
    if len(el) > k:
        r["steps"] = int(len(el))
        r["step_s"] = stats(el[k:])
        t, dt, cfl = (np.array([float(x[i]) for x in tdc]) for i in range(3))
        r["dt"] = float(np.median(dt))
        r["t_range"] = [float(t[0]), float(t[-1])]
        r["cfl"] = stats(cfl[k:])
        r["p_iter"] = stats(floats(r"FLUID p\s+: iter (\d+)", text)[k:])
        r["U_iter"] = stats(floats(r"FLUID U\s+: iter (\d+)", text)[k:])
        ts = r["step_s"]["median"]
        r["sim_time_per_wall_hour"] = r["dt"] / ts * 3600
        if r.get("dofs") and r["ranks"]:
            r["MDOF_steps_per_s_per_gpu"] = r["dofs"] / ts / r["ranks"] / 1e6
            r["gpu_h_per_time_unit"] = ts / r["dt"] * r["ranks"] / 3600
    # last runtime statistics block of NekRS (cumulative)
    blocks = text.split(">>> runtime statistics")
    if len(blocks) > 1:
        b = blocks[-1].split("\n\n")[0]
        get = lambda name: (lambda m: float(m.group(1)) if m else None)(re.search(r"\n\s*" + name + r"\s+" + NUM, b))
        solve = get("solve")
        r["runtime_breakdown"] = {
            "solve_s": solve,
            "totalElapsed_s": (lambda m: float(m.group(1)) if m else None)(re.search(r"totalElapsed=\s*" + NUM, b)),
        }
        for name in ("udfExecuteStep", "makef", "velocitySolve", "pressureSolve", "meshSolve", "checkpointing"):
            v = get(name)
            if v is not None and solve:
                r["runtime_breakdown"][name + "_frac"] = v / solve

    # Python side
    wing = re.findall(r"\[WING\] action (\d+)/\d+ .*? " + NUM + " s\n", text)
    spa = re.search(r"\[WING\] .*?(\d+) steps per action, resuming after (\d+) actions", text)
    if spa:
        r["steps_per_action"] = int(spa.group(1))
    if len(wing) > 2:
        a, w = (np.array([float(x[i]) for x in wing]) for i in range(2))
        r["actions_logged"] = [int(a[0]), int(a[-1])]
        r["action_s"] = float(np.polyfit(a, w, 1)[0])
    client = [json.loads(m) for m in re.findall(r"\[CLIENT\] (\{.*\})", text)]
    if len(client) > 2:
        w = np.array([c["wall_s"] for c in client])
        r["action_s"] = float(np.median(np.diff(w)[k // 10 :] if len(w) > 20 else np.diff(w)))
        if "dt" in r and "steps_per_action" not in r:
            r["steps_per_action"] = int(round(np.median(np.diff([c["time"] for c in client])) / r["dt"]))
    if "action_s" in r and "steps_per_action" in r and "step_s" in r:
        nek = r["steps_per_action"] * r["step_s"]["median"]
        r["action_nekrs_s"] = nek
        r["action_overhead_frac"] = 1 - nek / r["action_s"]
        if r["ranks"]:
            r["gpu_h_per_1000_actions"] = r["action_s"] * 1000 * r["ranks"] / 3600
    td3 = [json.loads(m) for m in re.findall(r"\[TD3\] (\{.*\"episode\".*\})", text)]
    if td3:
        r["training"] = training(td3)
    return r


def training(eps):
    """Episodes of train_td3.py (one record per finished episode and simulation)."""
    sims = max(e.get("sim", 0) for e in eps) + 1
    walls = sorted({(e.get("launch", 0), e["wall_s"]) for e in eps})
    dur = [w2 - w1 for (l1, w1), (l2, w2) in zip(walls, walls[1:]) if l1 == l2]
    acts = np.array([e["actions"] for e in eps])
    out = {
        "episodes": len(eps),
        "sims": sims,
        "actions_per_episode": int(np.median(acts)),
        "launches": len({e.get("launch", 0) for e in eps}),
        "episode_wall_s": stats(np.array(dur)) if dur else None,
        "timesteps": max(e["timesteps"] for e in eps),
        "last10_reward": float(np.mean([e["mean_reward"] for e in eps[-10:]])),
        "last10_dUdy": float(np.mean([e["last_third_dUdy"] for e in eps[-10:]])),
    }
    if dur:
        out["action_s_per_sim"] = float(np.median(dur)) / out["actions_per_episode"]
    return out


logs = []
for path in args.paths:
    if os.path.isdir(path):
        logs += sorted(glob.glob(os.path.join(path, "*.log")) + glob.glob(os.path.join(path, "*", "*.log")))
    else:
        logs.append(path)
gpu_logs = [f for f in logs if os.path.basename(f).startswith("gpu-")]
logs = [f for f in logs if not f.endswith("cmake.log") and f not in gpu_logs]
eps_files = sorted(
    {f for path in args.paths if os.path.isdir(path) for f in glob.glob(os.path.join(path, "td3_episodes.jsonl"))}
)

results = []
for f in logs:
    r = parse(f)
    if not any(k in r for k in ("step_s", "training", "action_s")):
        continue
    results.append(r)
for f in eps_files:
    eps = [json.loads(l) for l in open(f) if l.strip()]
    if eps:
        results.append({"log": f, "training": training(eps)})


def gpu_monitor(path):
    """rocm-smi --csv samples of the sbatch GPU monitor: mean and max of every numeric column over cards and time."""
    cols, data = None, {}
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.lower().startswith("device,"):
            cols = line.split(",")
        elif cols and line.startswith("card"):
            for c, v in zip(cols[1:], line.split(",")[1:]):
                try:
                    data.setdefault(c, []).append(float(v))
                except ValueError:
                    pass
    samples = sum(1 for l in open(path, errors="replace") if l.startswith("# time"))
    return {
        "log": path,
        "gpu_samples": samples,
        "gpu": {c: {"mean": float(np.mean(v)), "max": float(np.max(v))} for c, v in data.items() if v},
    }


for f in gpu_logs:
    results.append(gpu_monitor(f))


def fmt(v, spec=".3g"):
    return "-" if v is None else format(v, spec)


for r in results:
    print(f"== {r['log']}")
    if r.get("elements"):
        print(
            f"  case: {r['elements']} elements, N = {r['N']}, {r['dofs'] / 1e6:.3g} M DOFs, {r['ranks']} rank(s) "
            f"({r['backend']}), {r['dofs'] / r['ranks'] / 1e6:.3g} M DOFs per rank"
        )
    if r.get("init_s") is not None or r.get("jit_s") is not None:
        print(
            f"  setup: JIT {fmt(r.get('jit_s'))} s, initialization {fmt(r.get('init_s'))} s, "
            f"device memory rank 0 {fmt(r.get('device_memory_GB_rank0'))} GB"
        )
    if "Ax_kernel_fp64" in r:
        a = r["Ax_kernel_fp64"]
        print(
            f"  Ax kernel (fp64, N = {a['N']}): {a['GDOF/s']:.2f} GDOF/s, {a['GB/s']:.0f} GB/s, "
            f"{a['GFLOP/s']:.0f} GFLOP/s"
        )
    if "step_s" in r:
        s = r["step_s"]
        print(
            f"  steps: {r['steps']} (t = {r['t_range'][0]:.4g}-{r['t_range'][1]:.4g}, dt = {r['dt']:.3g}), step time "
            f"median {s['median'] * 1e3:.1f} ms, mean {s['mean'] * 1e3:.1f} ms, p95 {s['p95'] * 1e3:.1f} ms, "
            f"max {s['max'] * 1e3:.0f} ms"
        )
        print(
            f"  solver: pressure iterations {r['p_iter']['mean']:.1f} (max {r['p_iter']['max']:.0f}), velocity "
            f"{r['U_iter']['mean']:.1f}, CFL mean {r['cfl']['mean']:.3f} max {r['cfl']['max']:.3f}"
        )
        print(
            f"  throughput: {fmt(r.get('MDOF_steps_per_s_per_gpu'))} M DOF-steps/s per GPU, "
            f"{r['sim_time_per_wall_hour']:.3g} time units per "
            f"hour, {fmt(r.get('gpu_h_per_time_unit'))} GPU h per time unit"
        )
    if "runtime_breakdown" in r:
        b = r["runtime_breakdown"]
        print("  solve breakdown: " + ", ".join(f"{k[:-5]} {v:.0%}" for k, v in b.items() if k.endswith("_frac")))
    if "action_s" in r:
        line = f"  actions: {r['action_s']:.2f} s per action"
        if "action_nekrs_s" in r:
            line += (
                f" ({r['steps_per_action']} steps, NekRS {r['action_nekrs_s']:.2f} s, outside NekRS "
                f"{r['action_overhead_frac']:.0%}), {r['gpu_h_per_1000_actions']:.2f} GPU h per 1000 actions"
            )
        print(line)
    if "training" in r:
        t = r["training"]
        ew = t["episode_wall_s"]
        print(
            f"  training: {t['episodes']} episodes ({t['sims']} sim(s), {t['launches']} launch(es)), "
            f"{t['actions_per_episode']} actions per episode, episode wall {fmt(ew and ew['median'] / 60)} min, "
            f"{fmt(t.get('action_s_per_sim'))} s per action and sim, last 10: reward {t['last10_reward']:.4f}, "
            f"dU/dy {t['last10_dUdy']:.2f}"
        )

    if "gpu" in r:
        print(f"  GPU monitor: {r['gpu_samples']} samples")
        for c, v in r["gpu"].items():
            print(f"    {c}: mean {v['mean']:.4g}, max {v['max']:.4g}")

if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"{args.json} written")
