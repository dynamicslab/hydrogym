"""Load a policy of this folder by its MANIFEST.yaml id and return a callable obs -> action (numpy only).

    from policies.load_policy import load_policy, list_ids
    pi = load_policy("pinball_2d_re150/ppo/seed2")
    a = pi(obs)                      # normalised action in [-1, 1]; physical command = a * pi.max_control
    pi.reset()                       # stateful formats only (cavity rate limit and frame stack, pinball residual)

    python policies/load_policy.py --list             # id, environment, format of every entry
    python policies/load_policy.py --check            # every file present with its sha256 + a forward pass per entry
    python policies/load_policy.py <id> [<id> ...]    # print the interface of the given entries

Only numpy (and PyYAML for the index) are needed. No solver code is imported. This file restates the forward
pass of each file format from its layout:

  jax_ppo_pkl      pickle (params, norm). params["pi"] = list of {"w": (n_in, n_out), "b": (n_out,)}, tanh hidden
                   layers, linear output = Gaussian mean; params["log_std"]; params["v"] = critic (not used).
                   norm = running observation statistics {"mean", "var", ...}.
                   a = clip(mlp(clip((o - mean) / sqrt(var + 1e-8), -10, 10)), -1, 1)            (deterministic)
  jax_td3_pkl      pickle (params, norm). params["pi"] as above, params["q1"], params["q2"] = critics (not used).
                   a = tanh(mlp(clip((o - mean) / sqrt(var + 1e-8), -10, 10)))
  gust_deploy_npz  NACA0012 gust controller (w<i>, b<i>, norm_mean, norm_var, out = tanh | clip, act_mask). Call
                   pi(obs) with obs = (u_1..u_40, v_1..v_40) in lattice units of the target case; the controller is
                   feed-forward (an optional second argument, the action index, is accepted and not used).
  cavity_spec_npz  2D cavity controller spec (kind ppo). Call pi.observe(u_pts, v_pts) -> (O, s) on the
                   224 interpolation points "pts", then pi(O, s) (s, the wall-shear sensor, is not used by the PPO
                   controllers). Stateful (previous commands, frame stack), rate-limited; lag 0: the command acts
                   on the observation of the state at t_k.
  wall_mlp_npz     wall-turbulence actor applied per actuator (NACA0012 wing): w<i>, b<i>, n_layers, hidden_act
                   (relu | tanh), test_obs / test_act (output of the original framework). obs (n, 2) =
                   [u_t', v_n'] / u_tau of n actuators -> a = tanh(mlp(obs)), (n,), in units of the action amplitude.
  wall_blocks      wall_mlp_npz actors per chordwise block: JSON [{x0, x1, npz}] (npz relative to the JSON). Call
                   pi(obs, x) with x (n,) = chordwise position of each actuator; actuator i uses the block with
                   x0 <= x_i < x1 (the last block also takes x_i = x1).
  pinball_residual_npz  2D pinball feedback controller around a constant base rotation (w<i>, b<i>, n_layers,
                   norm_mean, norm_var, action_base, residual_scale, hold_actions, max_control, substeps).
                   obs (92,) = [u/U (40), v/U (40), previous action (3), C_D (3), C_L (3), C_P (3)] with the drag,
                   lift and shaft-power coefficients of the three cylinders (front, top, bottom) from the previous
                   action interval (zero at the start).
                   a = clip(action_base + residual_scale * clip(mlp(clip((o - mean) / sqrt(var + 1e-8), -10, 10)),
                   -1, 1), -1, 1); the first hold_actions calls after reset() return action_base. Stateful.
  cavity3d_npz     3D cavity slice-set policy. Call pi(P) with P (nz, 100) = per-plane probe values (u, v)/U of
                   all nz spanwise planes; the policy reads the planes "slices", pools over them and returns nA
                   spanwise-uniform jet commands. Stateful (previous command, frame stack), rate-limited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.yaml")


# ------------------------------------------------------------------------------------------------ manifest
_CACHE = {}


def _load_manifest(path=MANIFEST):
    """parsed MANIFEST.yaml, cached per (path, mtime)"""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _CACHE:
        import yaml

        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        with open(path) as f:
            _CACHE[key] = yaml.load(f, Loader=loader)
    return _CACHE[key]


def list_ids(path=MANIFEST):
    return [p["id"] for p in _load_manifest(path)["policies"]]


def entry(pid, path=MANIFEST):
    for p in _load_manifest(path)["policies"]:
        if p["id"] == pid:
            return p
    raise KeyError(f"no policy {pid!r} in {path}")


def _file(e, role):
    for f in e["files"]:
        if f["role"] == role:
            p = os.path.join(HERE, f["path"])
            if not os.path.exists(p):
                alt = os.path.join(HERE, "_large", f["path"])
                if os.path.exists(alt):
                    return alt
                raise FileNotFoundError(f"{p} (large files live in policies/_large/, not in git)")
            return p
    return None


# ------------------------------------------------------------------------------------------------ helpers
def _mlp(layers, x):
    for w, b in layers[:-1]:
        x = np.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def _nobs(o, mean, var):
    return np.clip((np.asarray(o, np.float64) - mean) / np.sqrt(var + 1e-8), -10.0, 10.0)


def _layers_pkl(params):
    return [(np.asarray(p["w"], np.float64), np.asarray(p["b"], np.float64)) for p in params["pi"]]


def _layers_npz(z, wkey, bkey, n):
    return [(np.asarray(z[f"{wkey}{i}"], np.float64), np.asarray(z[f"{bkey}{i}"], np.float64)) for i in range(n)]


class Policy:
    """callable obs -> normalised action; attributes describe the interface"""

    def __init__(self, fn, reset=None, **attrs):
        self._fn, self._reset = fn, reset
        self.__dict__.update(attrs)

    def __call__(self, *args, **kw):
        return self._fn(*args, **kw)

    def reset(self):
        if self._reset is not None:
            self._reset()


# ------------------------------------------------------------------------------------------------ formats
def _load_pkl(path, kind):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    params, norm = blob[0], blob[1]
    L = _layers_pkl(params)
    mean, var = np.asarray(norm["mean"], np.float64), np.asarray(norm["var"], np.float64)
    if kind == "ppo":
        fn = lambda o: np.clip(_mlp(L, _nobs(o, mean, var)), -1.0, 1.0)  # noqa: E731
    else:
        fn = lambda o: np.tanh(_mlp(L, _nobs(o, mean, var)))  # noqa: E731
    std = np.exp(np.asarray(params["log_std"], np.float64)) if "log_std" in params else None
    return Policy(
        fn, obs_dim=L[0][0].shape[0], n_actions=L[-1][0].shape[1], exploration_std=std, format=f"jax_{kind}_pkl"
    )


def _load_gust_npz(path):
    C = np.load(path, allow_pickle=False)
    n = int(C["n_layers"])
    L = _layers_npz(C, "w", "b", n)
    tanh_out = str(C["out"]) == "tanh"
    mean, var = np.asarray(C["norm_mean"], np.float64), np.asarray(C["norm_var"], np.float64)
    mask = np.asarray(C["act_mask"], np.float64) if "act_mask" in C.files else np.ones(3)

    def fn(o, k=None):
        y = _mlp(L, _nobs(np.asarray(o, np.float64), mean, var))
        return (np.tanh(y) if tanh_out else np.clip(y, -1.0, 1.0)) * mask

    meta = json.loads(str(C["meta"])) if "meta" in C.files else {}
    return Policy(
        fn,
        obs_dim=80,
        n_actions=3,
        max_control=float(C["max_control"]),
        substeps=int(C["substeps"]),
        Ma=float(C["Ma"]),
        probes_case=np.asarray(C["probes_case"]),
        probes_chord=np.asarray(C["probes_chord"]),
        meta=meta,
        format="gust_deploy_npz",
    )


def _load_cavity_spec(path, lag=0):
    d = dict(np.load(path, allow_pickle=False))
    kind, mode = str(d["kind"]), str(d["mode"])
    nA, stack, rate = int(d["nA"]), int(d["stack"]), float(d["rate"])
    f = {k: np.asarray(v, np.float64) for k, v in d.items() if v.dtype.kind == "f"}
    if kind != "ppo":
        raise NotImplementedError(f"{path}: cavity controller kind {kind!r}")
    L = _layers_npz(f, "pi_w", "pi_b", int(d["n_layers"]))
    tanh_mu = bool(d["tanh_mu"])
    st = {}

    def limit(p, prev):
        p = np.clip(p, -1.0, 1.0)
        if mode == "abs":
            return np.clip(p, prev - rate, prev + rate) if rate > 0 else p
        return np.clip(prev + rate * p, -1.0, 1.0)  # mode "delta"

    def observe(u_pts, v_pts):
        return np.concatenate([f["Wp"] @ u_pts, f["Wp"] @ v_pts]), float(f["ws"] @ u_pts)

    def reset():
        st.clear()

    def fn(O, s=None):  # noqa: E741 (O = probe observation vector, as in the format docstring)
        O = np.asarray(O, np.float64)  # noqa: E741
        if not st:  # first call after reset: history = first frame, a = 0
            z = np.zeros(nA)
            st.update(a1=z, a2=z, hist=np.tile(np.concatenate([O, z]), stack - 1) if stack > 1 else np.zeros(0))
        a1, a2 = st["a1"], st["a2"]
        amp_fr = a1 + lag * (a2 - a1)
        fr = np.concatenate([O, amp_fr])
        x = _mlp(L, _nobs(np.concatenate([fr, st["hist"]]), f["nmean"], f["nvar"]))
        new = limit(np.tanh(x) if tanh_mu else x, a1)
        if stack > 1:
            st["hist"] = np.concatenate([fr, st["hist"][: st["hist"].shape[0] - fr.shape[0]]])
        st["a2"], st["a1"] = a1, new
        return new

    return Policy(
        fn,
        reset=reset,
        observe=observe,
        kind=kind,
        n_actions=nA,
        stack=stack,
        rate=rate,
        mode=mode,
        pts=d["pts"],
        format="cavity_spec_npz",
    )


def _load_cavity3d(path):
    d = dict(np.load(path, allow_pickle=False))
    n = int(d["n_layers"])
    L = _layers_npz(d, "pi_w", "pi_b", n)
    W_std = np.asarray(d["pi_std"], np.float64)
    mean, var = np.asarray(d["nmean"], np.float64), np.asarray(d["nvar"], np.float64)
    tanh_mu = bool(int(d["tanh_mu"]))
    sl, stack = np.asarray(d["slices"], int), int(d["stack"])
    nA = L[-1][0].shape[1]
    rate = 0.1
    st = {}

    def mu(o):  # o (S, D): per-plane hidden layers, mean/std pooling
        x = _nobs(o, mean, var)
        for w, b in L[:-1]:
            x = np.tanh(x @ w + b)
        hm, hs = x.mean(-2), np.sqrt(x.var(-2) + 1e-8)
        w, b = L[-1]
        m = hm @ w + b + hs @ W_std
        return np.tanh(m) if tanh_mu else m

    def fn(P):
        if not st:
            st.update(amp=np.zeros(nA), hist=None)
        Oz = np.asarray(P, np.float64)[sl]
        fz = np.concatenate([Oz, np.broadcast_to(st["amp"][None], (Oz.shape[0], nA))], -1)
        if st["hist"] is None:
            st["hist"] = np.tile(fz, (1, stack - 1))
        p = np.clip(mu(np.concatenate([fz, st["hist"]], -1)), -1.0, 1.0)
        new = np.clip(p, st["amp"] - rate, st["amp"] + rate)
        if stack > 1:
            st["hist"] = np.concatenate([fz, st["hist"][:, : st["hist"].shape[1] - fz.shape[1]]], -1)
        st["amp"] = new
        return new

    return Policy(
        fn, reset=st.clear, n_actions=nA, slices=sl, stack=stack, nz=int(d["nz"]), rate=rate, format="cavity3d_npz"
    )


def _load_pinball_residual(path):
    z = np.load(path, allow_pickle=False)
    L = _layers_npz(z, "w", "b", int(z["n_layers"]))
    mean, var = np.asarray(z["norm_mean"], np.float64), np.asarray(z["norm_var"], np.float64)
    base, rs, hold = np.asarray(z["action_base"], np.float64), float(z["residual_scale"]), int(z["hold_actions"])
    st = {"k": 0}

    def fn(o):
        k = st["k"]
        st["k"] = k + 1
        if k < hold:
            return base.copy()
        return np.clip(base + rs * np.clip(_mlp(L, _nobs(o, mean, var)), -1.0, 1.0), -1.0, 1.0)

    def reset():
        st["k"] = 0

    return Policy(
        fn,
        reset=reset,
        obs_dim=L[0][0].shape[0],
        n_actions=L[-1][0].shape[1],
        obs_forces=True,
        hold_actions=hold,
        max_control=float(z["max_control"]),
        substeps=int(z["substeps"]),
        format="pinball_residual_npz",
    )


def _load_kolmogorov_npz(path):
    """Kolmogorov PPO / GPPO actor: obs (64 trajectory-mean velocity magnitudes) -> 4 mode amplitudes.

    Deterministic action = mid + half * tanh(MLP(obs)), mid/half from the action bounds; the returned action is in
    physical units (the environment's own bounds), not normalised.
    """
    z = np.load(path, allow_pickle=False)
    L = _layers_npz(z, "w", "b", 3)
    lo, hi = float(z["min_action"]), float(z["max_action"])
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)

    def fn(o):
        return mid + half * np.tanh(_mlp(L, np.asarray(o, np.float64)))

    return Policy(
        fn,
        obs_dim=L[0][0].shape[0],
        n_actions=L[-1][0].shape[1],
        min_action=lo,
        max_action=hi,
        exploration_std=np.exp(z["log_std"]),
        meta=json.loads(str(z["meta"])),
        format="kolmogorov_npz",
    )


def _wall_mlp(path):
    """wall_mlp_npz: layers, hidden activation, checked against the stored reference output"""
    z = np.load(path, allow_pickle=False)
    L = _layers_npz(z, "w", "b", int(z["n_layers"]))
    relu = str(z["hidden_act"]) == "relu"

    def fn(o):
        x = np.asarray(o, np.float64)
        for w, b in L[:-1]:
            x = x @ w + b
            x = np.maximum(x, 0.0) if relu else np.tanh(x)
        w, b = L[-1]
        return np.tanh(x @ w + b)[:, 0]

    err = np.abs(fn(z["test_obs"]) - z["test_act"]).max()
    if err > 1e-6:
        raise ValueError(f"{path}: forward pass differs from its reference output by {err:.3g}")
    return fn, L[0][0].shape[0]  # forward pass, input size


def _load_wall_mlp(path):
    fn, n_in = _wall_mlp(path)
    return Policy(fn, obs_dim=n_in, n_actions=1, per_actuator=True, format="wall_mlp_npz")


def _load_wall_blocks(path):
    with open(path) as f:
        blocks = json.load(f)
    nets = [_wall_mlp(os.path.join(os.path.dirname(path), b["npz"]))[0] for b in blocks]

    def fn(o, x):
        o, x = np.asarray(o, np.float64), np.asarray(x, np.float64)
        a = np.full(len(x), np.nan)
        for k, (b, net) in enumerate(zip(blocks, nets)):
            s = (x >= b["x0"]) & (x < b["x1"])
            if k == len(blocks) - 1:
                s |= np.isclose(x, b["x1"])
            if not np.all(np.isnan(a[s])):
                raise ValueError(f"block {b} overlaps another block")
            a[s] = net(o[s]) if s.any() else a[s]
        if np.isnan(a).any():
            raise ValueError(f"{int(np.isnan(a).sum())} actuators in no block")
        return a

    return Policy(fn, obs_dim=2, n_actions=1, per_actuator=True, blocks=blocks, format="wall_blocks")


LOADERS = {
    "kolmogorov_npz": _load_kolmogorov_npz,
    "jax_ppo_pkl": lambda p: _load_pkl(p, "ppo"),
    "jax_td3_pkl": lambda p: _load_pkl(p, "td3"),
    "gust_deploy_npz": _load_gust_npz,
    "cavity_spec_npz": _load_cavity_spec,
    "cavity3d_npz": _load_cavity3d,
    "wall_mlp_npz": _load_wall_mlp,
    "wall_blocks": _load_wall_blocks,
    "pinball_residual_npz": _load_pinball_residual,
}


def load_policy(pid, manifest=MANIFEST):
    """policy id (MANIFEST.yaml 'id') -> Policy, from the file and format of the entry's 'loader' block."""
    e = entry(pid, manifest)
    ld = e.get("loader") or {}
    fmt, role = ld.get("format"), ld.get("file_role")
    if fmt not in LOADERS:
        raise NotImplementedError(f"{pid}: no numpy loader for format {fmt!r}")
    pi = LOADERS[fmt](_file(e, role))
    pi.id, pi.manifest_entry = pid, e
    if getattr(pi, "max_control", None) is None and "max_control" in ld:
        pi.max_control = ld["max_control"]
    return pi


def load_probes(path):
    """probe coordinates (n, 2) or (n, 3) of a file in policies/probes ('probes/<name>' or an absolute path):
    JSON with 'probes_xy' / 'probes_xyz', or text with one point per line"""
    f = path if os.path.isabs(path) else os.path.join(HERE, path)
    if f.endswith(".json"):
        d = json.load(open(f))
        key = "probes_xyz" if "probes_xyz" in d else "probes_xy"
        a = np.asarray(d[key], np.float64)
        return a.reshape(-1, 3 if key == "probes_xyz" else 2)
    return np.loadtxt(f, dtype=np.float64, ndmin=2)


# ------------------------------------------------------------------------------------------------ CLI
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _smoke(pi, rng):
    fmt = pi.format
    if fmt in ("jax_ppo_pkl", "jax_td3_pkl"):
        return pi(rng.normal(size=(4, pi.obs_dim)))
    if fmt == "gust_deploy_npz":
        pi.reset()
        return np.stack([pi(rng.normal(size=80), k) for k in range(3)])
    if fmt == "cavity_spec_npz":
        pi.reset()
        n = pi.pts.shape[0]
        return np.stack([pi(*pi.observe(rng.normal(size=n) * 0.01 + 1.0, rng.normal(size=n) * 0.01)) for _ in range(3)])
    if fmt == "kolmogorov_npz":
        return pi(rng.uniform(0.0, 2.0, size=(3, pi.obs_dim)))
    if fmt == "wall_mlp_npz":
        return pi(rng.normal(size=(8, 2)) * [2.8, 0.6])
    if fmt == "wall_blocks":
        x = np.linspace(pi.blocks[0]["x0"], pi.blocks[-1]["x1"], 8)
        return pi(rng.normal(size=(8, 2)) * [2.8, 0.6], x)
    if fmt == "pinball_residual_npz":
        pi.reset()
        return np.stack([pi(rng.normal(size=pi.obs_dim)) for _ in range(pi.hold_actions + 3)])
    if fmt == "cavity3d_npz":
        pi.reset()
        return np.stack([pi(rng.normal(size=(pi.nz, 100)) * 0.1) for _ in range(3)])
    raise ValueError(fmt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--list", action="store_true", help="print id, environment and format of every entry")
    ap.add_argument("--check", action="store_true", help="verify all files (sha256, size) and run every loader once")
    a = ap.parse_args(argv)
    man = _load_manifest()
    if a.list:
        for p in man["policies"]:
            env = (p.get("hydrogym_env") or {}).get("id", "-")
            print(f"{p['id']:60s} {env:40s} {(p.get('loader') or {}).get('format')}")
        return 0
    if a.check:
        bad = 0
        rng = np.random.default_rng(0)
        for p in man["policies"]:
            for f in p["files"]:
                path = os.path.join(HERE, f["path"])
                if not os.path.exists(path):
                    path = os.path.join(HERE, "_large", f["path"])
                if not os.path.exists(path):
                    print("MISSING", f["path"])
                    bad += 1
                    continue
                if os.path.getsize(path) != f["size_bytes"] or _sha(path) != f["sha256"]:
                    print("SIZE/SHA256 MISMATCH", f["path"])
                    bad += 1
            fmt = (p.get("loader") or {}).get("format")
            if fmt in LOADERS:
                y = np.asarray(_smoke(load_policy(p["id"]), rng))
                if not np.all(np.isfinite(y)) or np.abs(y).max() > 1.0 + 1e-12:
                    print("BAD ACTION", p["id"], y)
                    bad += 1
        print("check:", "OK" if not bad else f"{bad} problem(s)", f"({len(man['policies'])} entries)")
        return 1 if bad else 0
    for pid in a.ids:
        pi = load_policy(pid)
        print(
            pid,
            "| format",
            pi.format,
            "| actions",
            pi.n_actions,
            "| obs_dim",
            getattr(pi, "obs_dim", "-"),
            "| max_control",
            getattr(pi, "max_control", "-"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
