"""Evaluation of trained Kolmogorov PPO/GPPO policies (Re 40 extreme-event preset): uncontrolled statistics,
start-state bank, deterministic policy roll-outs, constant-action baselines, evaluation of the trained runs.

usage (one process per GPU; all processes share --out and split the tasks between them):
  python evaluate.py --out eval --policy-root ../../../policies/kolmogorov_re40 [--with-trained --runs-root runs]
  python evaluate.py --out eval --policy-root x [--with-trained] --status     # 'complete' or the missing outputs
  python evaluate.py --quick ...                                                # tiny CPU smoke test

Task-queue design: every worker process (one per GPU) loops over the task list, claims a runnable task with an
mkdir lock, runs it and writes its output atomically. Outputs are raw per-episode arrays (summaries and thresholds
are computed afterwards by analyse.py), so a task only depends on the start-state bank. A killed job loses at most
one unc chunk / one task; the next job removes ``<out>/locks`` (as slurm/eval_chain.sbatch does) and continues.

Tasks
  unc        8 uncontrolled trajectories x 500 windows (chunks of 50, resumable); bank.pkl as soon as window 250.
  released   bank x {uncontrolled, PPO s30 det, GPPO s30 det} + reset x {unc, PPO det, GPPO det, PPO sto x16,
             GPPO sto x16}, with the released seed-30 policies.
  meanconst  bank + reset x {PPO s30 mean action, GPPO s30 mean action} (needs released).
  grid1_<j>  5^4 constants on bank state j (j in 0,5,10,15) for 25 actions.
  grid2      best 8 by reward + best 4 by dissipation (grid1 mean) on bank + reset for 100 actions.
  trained    every trained run (runs/<variant>/s<seed>/<algo>_params.pkl) det on bank + reset (only with
             --with-trained).
"""

import argparse
import itertools
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ppo.py (ActorCritic) next to this script

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow, KolmogorovFlowState
from hydrogym.jax.utils.utils import compute_energy_dissipation, compute_tke

LEVELS = (-0.4, -0.25, -0.1, 0.0, 0.15)
N_TRAJ, N_WIN, CHUNK = 8, 500, 50
BANK_WINDOWS = (150, 250)
PERT_AMP = 0.02
EP_LEN = 100
GRID_STATES = (0, 5, 10, 15)
GRID_LEN = 25
ESTIMATE_MIN = {"unc": 5, "released": 20, "meanconst": 15, "grid1": 20, "grid2": 25, "trained": 30}


QUICK = False


def nbank():
    return N_TRAJ * len(BANK_WINDOWS)


def make_env():
    if QUICK:  # cheap variant for a CPU smoke test of the code path (as in test_kolmogorov_gppo_resume.py)
        return KolmogorovFlow.extreme_events_re40(
            env_config={"params": {"dt": 0.01}}, flow_config={"grid_size": (16, 16)}
        )
    return KolmogorovFlow.extreme_events_re40()


ENV = None


def env():
    global ENV
    if ENV is None:
        ENV = make_env()
    return ENV


def save_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def step_record(e, p, state, action):
    obs, nxt, reward, done, info = e.step_env(jax.random.PRNGKey(0), state, action, p)
    eps = jax.vmap(lambda w: compute_energy_dissipation(w, e.kx, e.ky, e.flow.nu, e.n))(nxt.trajectory)
    tke = jax.vmap(lambda w: compute_tke(w, e.kx, e.ky, e.n))(nxt.trajectory)
    rec = {
        "reward": reward,
        "dissipation": info["mean_dissipation"],
        "control_cost": info["control_cost"],
        "tke": info["mean_tke"],
        "actuation_power": info["actuation_power"],
        "actuation_power_abs": info["actuation_power_abs"],
        "forcing_power": info["objective"]["forcing_power"],
        "action": jnp.clip(action, p.min_action, p.max_action),
        "eps_snap": eps,
        "tke_snap": tke,
    }
    return obs, nxt, rec


def stack_states(states):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)


# ------------------------------------------------------------------------------------------------ unc
def task_unc(out, deadline=float("inf")):
    e = env()
    p = e.default_params
    ck = out / "unc_ckpt.pkl"
    if ck.exists():
        c = load(ck)
        states = jax.tree_util.tree_map(jnp.asarray, c["states"])
        recs, done_w = c["recs"], c["windows"]
    else:
        _, s0 = e.reset_env(jax.random.PRNGKey(0), p)
        dirs = np.array(jax.random.normal(jax.random.PRNGKey(7), (N_TRAJ, 4)), dtype=np.float64)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        acts = PERT_AMP * dirs
        acts[0] = 0.0
        s8 = stack_states([s0] * N_TRAJ)
        first = jax.jit(jax.vmap(lambda s, a: step_record(e, p, s, a)[1:]))
        states, rec0 = first(s8, jnp.asarray(acts, jnp.float32))
        recs = [jax.tree_util.tree_map(np.asarray, rec0)]
        recs[0] = {k: v[:, None] for k, v in recs[0].items()}
        done_w = 1
        bank = {}
        save_atomic(
            ck,
            {
                "states": jax.tree_util.tree_map(np.asarray, states),
                "recs": recs,
                "windows": done_w,
                "bank": bank,
                "pert_actions": acts,
            },
        )
        c = load(ck)
    bank = c["bank"]

    @jax.jit
    def chunk(s):
        def body(s, _):
            _, nxt, rec = step_record(e, p, s, jnp.zeros(4))
            return nxt, (rec, nxt.omega_hat, nxt.trajectory)

        return jax.lax.scan(body, s, None, length=CHUNK)

    run = jax.vmap(chunk)
    t_chunk = 0.0
    while done_w < N_WIN:
        if time.time() + max(2 * t_chunk, 120) > deadline:
            print(f"unc: stopping at {done_w} windows (deadline)", flush=True)
            return False
        t0 = time.time()
        states, (rec, om, tr) = run(states)
        rec = jax.tree_util.tree_map(np.asarray, rec)
        for w in BANK_WINDOWS:  # window index w = state after w+1 windows (window 0 = the perturbation window)
            j = w - done_w
            if 0 <= j < CHUNK:
                bank[w] = {"omega_hat": np.asarray(om[:, j]), "trajectory": np.asarray(tr[:, j])}
        recs.append(rec)
        done_w += CHUNK
        save_atomic(
            ck,
            {
                "states": jax.tree_util.tree_map(np.asarray, states),
                "recs": recs,
                "windows": done_w,
                "bank": bank,
                "pert_actions": c["pert_actions"],
            },
        )
        t_chunk = time.time() - t0
        print(f"unc: {done_w}/{N_WIN} windows ({t_chunk:.0f} s/chunk)", flush=True)
        if len(bank) == len(BANK_WINDOWS) and not (out / "bank.pkl").exists():
            items = []
            for t in range(N_TRAJ):
                for w in BANK_WINDOWS:
                    items.append(
                        {
                            "traj": t,
                            "window": w,
                            "time": 4 * (w + 2),
                            "omega_hat": bank[w]["omega_hat"][t],
                            "trajectory": bank[w]["trajectory"][t],
                        }
                    )
            save_atomic(out / "bank.pkl", items)
            print("unc: bank written", flush=True)
    allrec = {k: np.concatenate([r[k] for r in recs], axis=1) for k in recs[0]}
    np.savez(out / "unc.npz.tmp.npz", pert_actions=c["pert_actions"], **allrec)
    os.replace(out / "unc.npz.tmp.npz", out / "unc.npz")
    return True


# ------------------------------------------------------------------------------------------------ episodes
def load_bank(out):
    items = load(out / "bank.pkl")
    e = env()
    states = KolmogorovFlowState(
        trajectory=jnp.asarray(np.stack([b["trajectory"] for b in items])),
        omega_hat=jnp.asarray(np.stack([b["omega_hat"] for b in items])),
        time=jnp.zeros(len(items), dtype=jnp.int32),
        terminal=jnp.zeros(len(items), dtype=bool),
    )
    obs = jax.vmap(lambda s: e.get_obs(s, e.default_params))(states)
    return obs, states


def reset_state():
    e = env()
    return e.reset_env(jax.random.PRNGKey(0), e.default_params)


def net_and_template():
    from ppo import ActorCritic

    e = env()
    net = ActorCritic(action_dim=4, hidden_dim=256, initial_log_std=-1.0)
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((e.default_params.obs_dim,)))
    return net, tmpl


def load_params(path, tmpl):
    raw = Path(path).read_bytes()
    if Path(path).suffix == ".pkl":
        raw = pickle.loads(raw)
    return flax.serialization.from_bytes(tmpl, raw)


def run_cases(out, cases, policies, n_steps=EP_LEN):
    """cases: list of dicts {start: 'reset' | bank index, kind: 0 zero / 1 const / 2 det / 3 sto, pid, const, key}.
    policies: list of param paths (stacked; pid indexes it). Returns per-case record arrays (n_cases, n_steps, ...)."""
    e = env()
    p = e.default_params
    net, tmpl = net_and_template()
    plist = [load_params(f, tmpl) for f in policies]
    P = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *plist)
    o_r, s_r = reset_state()
    if any(c["start"] != "reset" for c in cases):
        o_b, s_b = load_bank(out)
    obs, sts = [], []
    for c in cases:
        if c["start"] == "reset":
            obs.append(o_r), sts.append(s_r)
        else:
            obs.append(o_b[c["start"]]), sts.append(jax.tree_util.tree_map(lambda x: x[c["start"]], s_b))
    obs, sts = jnp.stack(obs), stack_states(sts)
    kind = jnp.asarray([c["kind"] for c in cases], jnp.int32)
    pid = jnp.asarray([c.get("pid", 0) for c in cases], jnp.int32)
    const = jnp.asarray([c.get("const", np.zeros(4)) for c in cases], jnp.float32)
    keys = jnp.stack([jax.random.PRNGKey(c.get("key", 0)) for c in cases])
    mid = (p.max_action + p.min_action) / 2
    half = (p.max_action - p.min_action) / 2

    def one(o, s, k, kd, pi_id, a_c):
        prm = jax.tree_util.tree_map(lambda x: x[pi_id], P)

        def body(carry, _):
            o, s, k = carry
            k, sk = jax.random.split(k)
            dist, _ = net.apply(prm, o)
            det = mid + half * jnp.tanh(dist.mean())
            sto = mid + half * jnp.tanh(dist.sample(seed=sk))
            a = jnp.where(kd == 0, jnp.zeros(4), jnp.where(kd == 1, a_c, jnp.where(kd == 2, det, sto)))
            o2, s2, rec = step_record(e, p, s, a)
            return (o2, s2, k), rec

        _, recs = jax.lax.scan(body, (o, s, k), None, length=n_steps)
        return recs

    t0 = time.time()
    recs = jax.jit(jax.vmap(one))(obs, sts, keys, kind, pid, const)
    recs = jax.tree_util.tree_map(np.asarray, recs)
    recs = {k: v for k, v in recs.items() if k != "tke_snap"}
    return recs, time.time() - t0


def released_paths(policy_root):
    pr = Path(policy_root)
    return [pr / "ppo/seed30/params.msgpack", pr / "gppo/seed30/params.msgpack"]


def task_released(out, policy_root):
    cases, names = [], []
    for b in range(nbank()):
        for kd, pid, nm in ((0, 0, "uncontrolled"), (2, 0, "ppo_s30"), (2, 1, "gppo_s30")):
            cases.append({"start": b, "kind": kd, "pid": pid}), names.append((nm, f"bank{b}"))
    for kd, pid, nm in ((0, 0, "uncontrolled"), (2, 0, "ppo_s30"), (2, 1, "gppo_s30")):
        cases.append({"start": "reset", "kind": kd, "pid": pid}), names.append((nm, "reset"))
    for pid, nm in ((0, "ppo_s30_sto"), (1, "gppo_s30_sto")):
        for k in range(nbank()):
            cases.append({"start": "reset", "kind": 3, "pid": pid, "key": 1000 + k}), names.append((nm, f"reset_k{k}"))
    recs, wall = run_cases(out, cases, released_paths(policy_root))
    save_atomic(out / "released.pkl", {"names": names, "recs": recs, "wall_s": wall})
    return True


def task_meanconst(out, policy_root):
    d = load(out / "released.pkl")
    cases, names = [], []
    for nm in ("ppo_s30", "gppo_s30"):
        idx = [i for i, (n, s) in enumerate(d["names"]) if n == nm and s.startswith("bank")]
        ma = d["recs"]["action"][idx].mean((0, 1))
        for b in list(range(nbank())) + ["reset"]:
            (
                cases.append({"start": b, "kind": 1, "const": ma}),
                names.append((f"{nm}_meanconst", f"bank{b}" if b != "reset" else "reset")),
            )
    recs, wall = run_cases(out, cases, released_paths(policy_root)[:1])
    save_atomic(out / "meanconst.pkl", {"names": names, "recs": recs, "wall_s": wall})
    return True


GRID = np.array(list(itertools.product(LEVELS, repeat=4)), dtype=np.float32)


def task_grid1(out, policy_root, j):
    cases = [{"start": j, "kind": 1, "const": a} for a in GRID]
    recs, wall = run_cases(out, cases, released_paths(policy_root)[:1], n_steps=GRID_LEN)
    save_atomic(
        out / f"grid1_{j}.pkl",
        {"points": GRID, "reward": recs["reward"].mean(1), "dissipation": recs["dissipation"].mean(1), "wall_s": wall},
    )
    return True


def task_grid2(out, policy_root):
    g = [load(out / f"grid1_{j}.pkl") for j in GRID_STATES]
    r = np.mean([x["reward"] for x in g], 0)
    d = np.mean([x["dissipation"] for x in g], 0)
    cand = np.unique(np.concatenate([np.argsort(-r)[:8], np.argsort(d)[:4]]))
    cases, names = [], []
    for c in cand:
        for b in list(range(nbank())) + ["reset"]:
            (
                cases.append({"start": b, "kind": 1, "const": GRID[c]}),
                names.append((f"const{int(c)}", f"bank{b}" if b != "reset" else "reset")),
            )
    recs, wall = run_cases(out, cases, released_paths(policy_root)[:1])
    save_atomic(
        out / "grid2.pkl",
        {
            "names": names,
            "recs": recs,
            "candidates": cand,
            "points": GRID,
            "stage1_reward": r,
            "stage1_dissipation": d,
            "wall_s": wall,
        },
    )
    return True


VARIANTS = ("ppo_mb", "ppo_ro", "gppo_b0", "gppo_b2")


def task_trained(out, policy_root, runs_root, variant):
    files = sorted(Path(runs_root).glob(f"{variant}/s*/*_params.pkl"))
    cases, names = [], []
    for i, f in enumerate(files):
        tag = f"{f.parent.parent.name}_{f.parent.name}"
        for b in list(range(nbank())) + ["reset"]:
            (
                cases.append({"start": b, "kind": 2, "pid": i}),
                names.append((tag, f"bank{b}" if b != "reset" else "reset")),
            )
    recs, wall = run_cases(out, cases, files)
    save_atomic(
        out / f"trained_{variant}.pkl",
        {"names": names, "recs": recs, "files": [str(f) for f in files], "wall_s": wall},
    )
    return True


# ------------------------------------------------------------------------------------------------ queue
def task_list(with_trained):
    t = [("unc", "unc.npz", []), ("released", "released.pkl", ["bank.pkl"])]
    t += [(f"grid1_{j}", f"grid1_{j}.pkl", ["bank.pkl"]) for j in GRID_STATES]
    t += [("meanconst", "meanconst.pkl", ["released.pkl"])]
    t += [("grid2", "grid2.pkl", [f"grid1_{j}.pkl" for j in GRID_STATES])]
    if with_trained:
        t += [(f"trained_{v}", f"trained_{v}.pkl", ["bank.pkl"]) for v in VARIANTS]
    return t


def worker(out, a):
    deadline = a.job_start + 60 * a.minutes
    tasks = task_list(a.with_trained)
    failed = set()
    locks = out / "locks"
    locks.mkdir(exist_ok=True)
    while True:
        pending = [t for t in tasks if not (out / t[1]).exists() and t[0] not in failed]
        if not pending:
            print("worker: all tasks done", flush=True)
            return
        started = False
        for name, prod, deps in pending:
            if not all((out / d).exists() for d in deps):
                continue
            est = ESTIMATE_MIN[name.split("_")[0]]
            if time.time() + 60 * est > deadline:
                continue
            try:
                os.mkdir(locks / name)
            except FileExistsError:
                continue
            try:
                if (out / prod).exists():
                    continue
                t0 = time.time()
                print(f"worker: start {name}", flush=True)
                if name == "unc":
                    task_unc(out, deadline)
                elif name == "released":
                    task_released(out, a.policy_root)
                elif name == "meanconst":
                    task_meanconst(out, a.policy_root)
                elif name.startswith("grid1_"):
                    task_grid1(out, a.policy_root, int(name.split("_")[1]))
                elif name == "grid2":
                    task_grid2(out, a.policy_root)
                elif name.startswith("trained_"):
                    task_trained(out, a.policy_root, a.runs_root, name[len("trained_") :])
                print(f"worker: done {name} in {time.time() - t0:.0f} s", flush=True)
            except Exception:
                import traceback

                traceback.print_exc()
                failed.add(name)
                print(f"worker: FAILED {name}", flush=True)
            finally:
                os.rmdir(locks / name)
            started = True
            break
        if started:
            continue
        busy = any((locks / t[0]).exists() for t in pending)
        if not busy and not any(all((out / d).exists() for d in t[2]) for t in pending):
            print("worker: nothing runnable and nothing in progress", flush=True)
            return
        if time.time() + 120 > deadline:
            print("worker: out of time", flush=True)
            return
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-root", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--with-trained", action="store_true")
    ap.add_argument("--job-start", type=float, default=time.time())
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--status", action="store_true", help="print 'complete' if every task output exists")
    ap.add_argument("--quick", action="store_true", help="tiny sizes and a 16x16 grid, CPU smoke test")
    a = ap.parse_args()
    if a.quick:
        global QUICK, N_TRAJ, N_WIN, CHUNK, BANK_WINDOWS, EP_LEN, GRID_LEN, LEVELS, GRID_STATES, GRID
        QUICK, N_TRAJ, N_WIN, CHUNK, BANK_WINDOWS, EP_LEN, GRID_LEN = True, 2, 6, 2, (2, 3), 2, 1
        LEVELS, GRID_STATES = (-0.4, 0.15), (0, 1, 2, 3)
        GRID = np.array(list(itertools.product(LEVELS, repeat=4)), dtype=np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        missing = [t[1] for t in task_list(a.with_trained) if not (out / t[1]).exists()]
        print("complete" if not missing else "missing " + " ".join(missing))
        return
    print("devices", jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
    worker(out, a)


if __name__ == "__main__":
    main()
