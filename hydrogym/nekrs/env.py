"""
YAML-driven Gymnasium environment for NekRS cases with the HydroGym control interface.

Everything the agent sees and does is defined in the YAML (see examples/nekrs/channel/channel_re207.yml):

  environment:  nproc, steps_per_action, episode_actions, target_cfl
  actuators:    boundaries, positions (or an explicit `jets` list), footprint, direction, amplitude, znmf,
                resolution_check {mode: error | warn | off, min_nodes_per_width}
  probes:       fields, positions, subtract_plane_mean, normalization
  reward:       type wall_shear {boundary, quantity: gradient | force, baseline_dudy, average: legacy | mean}
  initial_conditions (episode reset):
                source: continue (default, the flow runs on) | snapshots | files
                snapshots {count, interval, warmup, save, load}: library of exact solver states, built with zero
                    actuation (warmup steps, then one snapshot every interval steps) or read from disk
                files [globs]: NekRS field files (u, p; the time-stepper history is rebuilt from the loaded state)
                interpolate, selection: random | cycle | first, settle_steps (zero-action steps after the reset)

The solver only marks the boundaries that may be actuated ([CASEDATA] hydrogymBoundaries in the par file).
Every solver step is logged with its CFL number (solver_steps.csv), every action with the reward, the CFL range
and the actuation diagnostics (steps.jsonl).
"""

import copy
import glob
import json
import os
import time
import warnings
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import yaml

from . import layout
from .interface import NekRSInterface


def _merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(config: Union[str, Dict]) -> Dict:
    """YAML file or dict; `base: other.yml` (next to the file, else relative to the cwd) is merged underneath."""
    if isinstance(config, dict):
        cfg, here = copy.deepcopy(config), "."
    else:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        here = os.path.dirname(config) or "."
    base = cfg.pop("base", None)
    if base:
        path = os.path.join(here, base)
        cfg = _merge(load_config(path if os.path.exists(path) else base), cfg)
    return cfg


class NekRSFlowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Union[str, Dict], run_root: str = "runs/nekrs", nproc: Optional[int] = None):
        self.cfg = load_config(config)
        env = self.cfg["environment"]
        self.nproc = nproc if nproc is not None else env.get("nproc")
        self.steps_per_action = int(env["steps_per_action"])
        self.episode_actions = int(env.get("episode_actions", 0))
        self.target_cfl = float(env.get("target_cfl", np.inf))
        self.run_root = run_root
        os.makedirs(run_root, exist_ok=True)

        self.iface = NekRSInterface(self.nproc)
        print(f"[HGNEK] solver: {self.iface.info}", flush=True)
        try:
            self._setup_actuators()
            self._setup_probes()
            self._setup_reward()
            self._setup_initial_conditions()
        except BaseException:
            # a rejected layout (e.g. under-resolved actuators) must also stop the solver, else NekRS waits forever
            self.iface.finishRun()
            self.iface = None
            raise

        self.n_actuators = len(self.actuators["xyz"])
        self.obs_per_actuator = len(self.probe_fields) * len(self.probes["xyz"]) // max(self.n_actuators, 1)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n_actuators,), dtype=np.float32)
        n_obs = len(self.probes["xyz"]) * len(self.probe_fields)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        self.act_index = 0
        self.episode = -1
        self.tstep = int(self.iface.info["tstep"])
        self._step_log = open(os.path.join(run_root, "steps.jsonl"), "w")
        self._solver_log = open(os.path.join(run_root, "solver_steps.csv"), "w")
        cols = ["tstep", "time", "dt", "cfl"]
        for bid, kind in self.monitors:
            names = ["Fvx", "Fvy", "Fvz", "Fpx", "Fpy", "Fpz"] if kind == "force" else ["dudn", "dvdn", "dwdn", "-", "-", "-"]
            cols += [f"{n}_b{bid}" for n in names]
        self._solver_log.write(",".join(cols) + "\n")
        self._episode_log = open(os.path.join(run_root, "episodes.jsonl"), "w")
        with open(os.path.join(run_root, "config.yml"), "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

    # ------------------------------------------------------------------ setup

    def _setup_actuators(self):
        a = self.cfg["actuators"]
        if "jets" in a:
            jets = a["jets"]
            xyz = np.array([j["position"] for j in jets], dtype=float)
            extra = {}
            parts = [layout.footprint_arrays(j.get("footprint", a.get("footprint")), 1) for j in jets]
            ftype = np.concatenate([p[0] for p in parts])
            widths = np.concatenate([p[1] for p in parts])
            parts = [layout.direction_arrays(j.get("direction", a.get("direction")), 1) for j in jets]
            dmode = np.concatenate([p[0] for p in parts])
            vec = np.concatenate([p[1] for p in parts])
            ang = np.concatenate([p[2] for p in parts])
            amp = np.array([float(j.get("amplitude", a.get("amplitude", 1.0))) for j in jets])
        else:
            xyz, extra = layout.positions(a["positions"], self.iface)
            n = len(xyz)
            ftype, widths = layout.footprint_arrays(a.get("footprint"), n)
            dmode, vec, ang = layout.direction_arrays(a.get("direction"), n)
            amp = np.full(n, float(a.get("amplitude", 1.0)))

        check = a.get("resolution_check", {})
        if isinstance(check, str):
            check = {"mode": check}
        res = self.iface.define_actuators(
            boundaries=[int(b) for b in a["boundaries"]],
            centres=xyz,
            footprint=ftype,
            widths=widths,
            direction=dmode,
            vectors=vec,
            angles=ang,
            znmf=a.get("znmf", "none"),
            check=check.get("mode", "error"),
            min_nodes_per_width=float(check.get("min_nodes_per_width", 2.0)),
        )
        self.amplitude = amp
        self.actuators = {"xyz": xyz, **extra, **res}
        # grid identifiers of the actuators, as hydrogym.nek (agent names of NekParallelEnv); zeros for free jets
        zeros = np.zeros(len(xyz), dtype=int)
        self.actuator_info = {
            "NID": np.asarray(extra.get("rank", zeros), dtype=int),
            "GLLID": np.asarray(extra.get("element", zeros), dtype=int) if "element" in extra else np.arange(len(xyz)),
            "FACEID": np.asarray(extra.get("face", zeros), dtype=int),
            "ix": np.asarray(extra.get("ix", zeros), dtype=int),
            "iy": np.asarray(extra.get("iy", zeros), dtype=int),
            "iz": np.asarray(extra.get("iz", zeros), dtype=int),
        }
        self._write_table("actuators.csv", self.actuators)
        nr = res["nodes_per_width"]
        print(
            f"[HGNEK] {len(xyz)} actuators, wall nodes per actuator {res['nodes'].min()}..{res['nodes'].max()}, "
            f"under-resolved {int(res['under_resolved'].sum())}"
            + (f", nodes per width >= {nr[nr >= 0].min():.2f}" if (nr >= 0).any() else ""),
            flush=True,
        )

    def _setup_probes(self):
        p = self.cfg["probes"]
        xyz, extra = layout.positions(p["positions"], self.iface, self.actuators)
        self.probe_fields = list(p["fields"])
        res = self.iface.define_probes(xyz, self.probe_fields, p.get("subtract_plane_mean"))
        norm = p.get("normalization", 1.0)
        self.obs_scale = np.broadcast_to(np.asarray(norm, dtype=float), (len(self.probe_fields),)).copy()
        self.probes = {"xyz": xyz, **extra}
        self._write_table("probes.csv", self.probes)
        print(f"[HGNEK] {len(xyz)} probes x {self.probe_fields}, {res['n_border']} on element borders", flush=True)

    def _setup_reward(self):
        r = self.cfg["reward"]
        if r["type"] != "wall_shear":
            raise ValueError(f"unknown reward type {r['type']} (override compute_reward for custom rewards)")
        self.monitors = [(int(r["boundary"]), r.get("quantity", "gradient"))]
        self.monitor_area = self.iface.define_monitors(self.monitors)

    def _setup_initial_conditions(self):
        ic = self.cfg.get("initial_conditions") or {}
        self.ic_source = ic.get("source", "continue")
        if self.ic_source not in ("continue", "snapshots", "files"):
            raise ValueError(f"initial_conditions.source {self.ic_source}: continue | snapshots | files")
        self.ic_selection = ic.get("selection", "random")
        if self.ic_selection not in ("random", "cycle", "first"):
            raise ValueError(f"initial_conditions.selection {self.ic_selection}: random | cycle | first")
        self.ic_settle = int(ic.get("settle_steps", 0))
        self.ic_snap = dict(ic.get("snapshots") or {})
        self.ic_interpolate = bool(ic.get("interpolate", False))
        self.ic_files = []
        if self.ic_source == "files":
            for pat in ic.get("files") or []:
                found = sorted(glob.glob(pat))
                if not found:
                    raise FileNotFoundError(f"initial_conditions.files: no match for {pat} (relative to {os.getcwd()})")
                self.ic_files += found
        self.ic_count = len(self.ic_files) if self.ic_source == "files" else int(self.ic_snap.get("count", 1))
        self._library_ready = self.ic_source != "snapshots"

    def _advance(self, n: int) -> np.ndarray:
        rows = self.iface.runTimeSteps(n)
        for row in rows:
            self._solver_log.write(",".join(repr(float(v)) for v in row) + "\n")
        self._solver_log.flush()
        self.tstep = int(rows[-1, 0])
        return rows

    def _build_library(self):
        """Snapshot library: read from disk, or run with zero actuation and store a state every interval steps."""
        sn, count = self.ic_snap, self.ic_count
        t0 = time.time()
        if sn.get("load"):
            for i in range(count):
                r = self.iface.read_snapshot(i, f"{sn['load']}_{i:03d}")
                print(f"[HGNEK] snapshot {i} read (t = {r['time']:.4f}, step {r['tstep']})", flush=True)
        else:
            interval = int(sn.get("interval", 100))
            self.iface.setControlProperties(np.zeros(self.n_actuators))
            hist = max(0, 3 - self.tstep)  # the time stepper needs three steps of history
            warm = max(int(sn.get("warmup", 0)), hist)
            if warm:
                self._advance(warm)
            for i in range(count):
                if i:
                    self._advance(interval)
                r = self.iface.save_snapshot(i)
                print(f"[HGNEK] snapshot {i} at t = {r['time']:.4f} ({r['bytes'] / 1e6:.1f} MB)", flush=True)
                if sn.get("save"):
                    os.makedirs(os.path.dirname(sn["save"]) or ".", exist_ok=True)
                    self.iface.write_snapshot(i, f"{sn['save']}_{i:03d}")
        self._library_ready = True
        print(f"[HGNEK] snapshot library of {count} states ready in {time.time() - t0:.1f} s", flush=True)

    def _write_table(self, name, cols):
        n = len(cols["xyz"])
        out = {"x": cols["xyz"][:, 0], "y": cols["xyz"][:, 1], "z": cols["xyz"][:, 2]}
        for k, v in cols.items():
            v = np.asarray(v)
            if k == "xyz" or v.shape[:1] != (n,):
                continue
            if v.ndim == 2:
                for d in range(v.shape[1]):
                    out[f"{k}{d}"] = v[:, d]
            else:
                out[k] = v
        keys = list(out)
        with open(os.path.join(self.run_root, name), "w") as f:
            f.write(",".join(keys) + "\n")
            for i in range(n):
                f.write(",".join(repr(out[k][i].item()) for k in keys) + "\n")

    # ------------------------------------------------------------------ gym API

    def _observe(self) -> Tuple[np.ndarray, float, float]:
        t, cfl, vals = self.iface.get_probes()
        obs = (vals / self.obs_scale).astype(np.float32).ravel()  # probe-major: [fields of probe 0, probe 1, ...]
        return obs, t, cfl

    def reset(self, seed=None, options=None):
        """
        Start an episode from an initial condition of the library (see initial_conditions); with source
        continue the flow runs on from its current state (as RSETS in the Nek5000 coupling).
        options: {"index": i} selects a library entry explicitly.
        """
        super().reset(seed=seed)
        self.act_index = 0
        self.episode += 1
        rec = {"episode": self.episode, "source": self.ic_source, "solver_step": self.tstep}
        if self.ic_source != "continue":
            if not self._library_ready:
                self._build_library()
            if options and "index" in options:
                idx = int(options["index"])
            elif self.ic_selection == "random":
                idx = int(self.np_random.integers(self.ic_count))
            elif self.ic_selection == "cycle":
                idx = self.episode % self.ic_count
            else:
                idx = 0
            if self.ic_source == "snapshots":
                r = self.iface.reset_to_snapshot(idx)
            else:
                r = self.iface.reset_from_file(self.ic_files[idx], self.ic_interpolate)
                rec["file"] = self.ic_files[idx]
            rec.update(index=idx, state_time=r["state_time"])
            if self.ic_settle:
                self._advance(self.ic_settle)
        obs, t, cfl = self._observe()
        rec.update(time=t, cfl=cfl)
        self._episode_log.write(json.dumps(rec) + "\n")
        self._episode_log.flush()
        return obs, {"time": t, "cfl": cfl, **{k: v for k, v in rec.items() if k in ("index", "state_time", "file")}}

    def compute_reward(self, rows: np.ndarray) -> Tuple[float, Dict]:
        r = self.cfg["reward"]
        bid, kind = self.monitors[0]
        if kind == "gradient":
            dudy = rows[:, 4]
        else:
            dudy = np.abs(rows[:, 4]) / (self.iface.info["mu"] * self.monitor_area[0])
        if r.get("average", "mean") == "legacy":
            # running mean of drl_reward.f: rwd_1 = d_1, rwd_i = (i rwd_{i-1} + d_i) / (i + 1)
            m = dudy[0]
            for i in range(2, len(dudy) + 1):
                m = (m * i + dudy[i - 1]) / (i + 1)
        else:
            m = dudy.mean()
        return float(1 - m / float(r["baseline_dudy"])), {"dUdy": float(m), "dUdy_last": float(dudy[-1])}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        if action.size != self.n_actuators:
            raise ValueError(f"action size {action.size}, expected {self.n_actuators}")
        t0 = time.time()
        diag = self.iface.setControlProperties(action * self.amplitude)
        rows = self._advance(self.steps_per_action)

        reward, rinfo = self.compute_reward(rows)
        obs, t, cfl_obs = self._observe()
        self.act_index += 1

        cfl = rows[:, 3]
        cfl_exceeded = bool(cfl.max() >= self.target_cfl)
        if cfl_exceeded:
            warnings.warn(f"CFL {cfl.max():.3f} >= target {self.target_cfl} at t = {t:.4f}")
        # a CFL violation ends the episode (terminal), the episode length is a time limit (truncation)
        terminated = cfl_exceeded
        truncated = self.episode_actions > 0 and self.act_index >= self.episode_actions
        info = {
            "time": t,
            "cfl_max": float(cfl.max()),
            "cfl_min": float(cfl.min()),
            "cfl_last": float(cfl[-1]),
            "cfl_exceeded": cfl_exceeded,
            **rinfo,
            **{k: float(v) for k, v in diag.items()},
            "wall_s": time.time() - t0,
            "reward_per_actuator": np.full(self.n_actuators, reward, dtype=np.float32),
        }
        rec = {"episode": self.episode, "k": self.act_index, "reward": reward, **{k: v for k, v in info.items() if k != "reward_per_actuator"}}
        self._step_log.write(json.dumps(rec) + "\n")
        self._step_log.flush()
        return obs, reward, terminated, truncated, info

    def close(self):
        if getattr(self, "iface", None) is not None:
            self._step_log.close()
            self._solver_log.close()
            self._episode_log.close()
            self.iface.finishRun()
            self.iface = None
