"""
YAML-driven Gymnasium environment for NekRS cases with the HydroGym control interface.

Everything the agent sees and does is defined in the YAML (see examples/nekrs/channel/channel_re207.yml and,
for the wall-unit options of a non-uniform wall, examples/nekrs/wing/wing.yml):

```text
  environment:  nproc, steps_per_action (n or {dt_plus: value}), episode_actions, target_cfl, dt (optional:
                the solver time step of the par file the configuration is made for; checked at start-up)
  wall_units:   nu (default mu / rho of the solver), u_tau: value | {polynomial: [c0, c1, ...], axis: x}
                (needed by the u_tau / plus options below)
  actuators:    boundaries, positions (or an explicit `jets` list), footprint, direction,
                amplitude (value or {u_tau: factor}), node_box [xmin, xmax, ymin, ymax, zmin, zmax] (only these
                boundary nodes are actuated), period [px, py, pz] (periodic footprints, 0 = none),
                znmf (none | uniform | actions, solver side, or
                {group_by: [x, y], tol}: zero net flux on every group of actuators, e.g. per spanwise line),
                resolution_check {mode: error | warn | off, min_nodes_per_width}
  probes:       fields, positions (actuator_offset normal_distance may be {plus: y+}), subtract_plane_mean
                (solver side) or subtract_group_mean: [x, y] (mean over probes sharing x, y, weighted by the
                actuator footprint), frame: global | wall (u, v -> wall tangent u_t (t_x > 0) and inward normal
                v_n) | hg_fortran (projection of drl_state.f in the HydroGym Nek5000 wing case, kept for
                compatibility with policies trained there: u_t = u n_y + v n_x; v_n as for wall), normalization
                (value per field, or u_tau)
  reward:       type wall_shear {boundary, quantity: gradient | force, baseline_dudy, average: legacy | mean}
  monitors:     further [{boundary, quantity: force | gradient}] recorded at every solver step (solver_steps.csv)
  power:        actuator power (hydrogym.power interface, see hydrogym/nekrs/power.py); all keys optional:
                enabled (default true; false skips the wall requests), boundaries (default actuators.boundaries),
                pressure_mean (default [x, y]: p' about the mean over the nodes sharing x and y; [] = full p),
                reference {velocity, length, span} or {velocity, area} (C_P = P / (0.5 rho U^3 A); defaults
                U = 1 and A = area of the power boundaries), drag {boundaries, axis} (drag_coefficient from the
                force monitors, same reference), power_weight (0), power_cost (abs), power_sampling (mean: every
                solver step of the action, one step per request; end: after the last step)
  initial_conditions (episode reset):
                source: continue (default, the flow runs on) | snapshots | files
                snapshots {count, interval, warmup, save, load}: library of exact solver states, built with zero
                    actuation (warmup steps, then one snapshot every interval steps) or read from disk
                files [globs]: NekRS field files (u, p; the time-stepper history is rebuilt from the loaded state)
                interpolate, selection: random | cycle | first, settle_steps (zero-action steps after the reset)
```

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

from ..power import POWER_COSTS, PowerConfig, power_info
from . import layout
from .interface import NekRSInterface
from .power import PowerAverage, WallPower, reference_power


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

    def __init__(
        self,
        config: Union[str, Dict],
        run_root: str = "runs/nekrs",
        nproc: Optional[int] = None,
        iface: Optional[NekRSInterface] = None,
        power_weight: Optional[float] = None,
        power_cost: Optional[str] = None,
        power_sampling: Optional[str] = None,
    ):
        """
        iface: an interface to one instance of a multi-instance launch (see multisim); default: the only one.
        power_weight / power_cost / power_sampling override the keys of the YAML power section (hydrogym.power).
        """
        self.cfg = load_config(config)
        env = self.cfg["environment"]
        self.nproc = nproc if nproc is not None else env.get("nproc")
        self._steps_spec = env["steps_per_action"]
        self.episode_actions = int(env.get("episode_actions", 0))
        self.target_cfl = float(env.get("target_cfl", np.inf))
        self.run_root = run_root
        os.makedirs(run_root, exist_ok=True)

        self.iface = iface if iface is not None else NekRSInterface(self.nproc)
        print(f"[HGNEK] solver: {self.iface.info}", flush=True)
        try:
            self._check_dt(env.get("dt"))
            wu = self.cfg.get("wall_units") or {}
            self.nu = float(wu.get("nu", self.iface.info["mu"] / self.iface.info["rho"]))
            self._setup_actuators()
            self.steps_per_action = self._resolve_steps(self._steps_spec)
            self._setup_probes()
            self._setup_reward()
            self._setup_initial_conditions()
            self._setup_power(power_weight, power_cost, power_sampling)
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
        self._single = False  # the running action advances one solver step per request
        self._step_hook, self._step_hook_every = None, 0
        self._step_log = open(os.path.join(run_root, "steps.jsonl"), "w")
        self._solver_log = open(os.path.join(run_root, "solver_steps.csv"), "w")
        cols = ["tstep", "time", "dt", "cfl"]
        for bid, kind in self.monitors:
            names = (
                ["Fvx", "Fvy", "Fvz", "Fpx", "Fpy", "Fpz"]
                if kind == "force"
                else ["dudn", "dvdn", "dwdn", "-", "-", "-"]
            )
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
            amp = a.get("amplitude", 1.0)
            amp = float(amp["u_tau"]) * self.u_tau(xyz) if isinstance(amp, dict) else np.full(n, float(amp))

        znmf = a.get("znmf", "none")
        self._znmf_groups = None
        if isinstance(znmf, dict):
            self._znmf_groups = layout.group_ids(xyz, znmf["group_by"], znmf.get("tol"))
            znmf = "none"
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
            znmf=znmf,
            check=check.get("mode", "error"),
            min_nodes_per_width=float(check.get("min_nodes_per_width", 2.0)),
            node_box=a.get("node_box"),
            period=a.get("period"),
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
        if self._znmf_groups is not None:
            self.actuators["znmf_group"] = self._znmf_groups
        self.actuators["amplitude"] = amp
        self._write_table("actuators.csv", self.actuators)
        nr = res["nodes_per_width"]
        print(
            f"[HGNEK] {len(xyz)} actuators, wall nodes per actuator {res['nodes'].min()}..{res['nodes'].max()}, "
            f"under-resolved {int(res['under_resolved'].sum())}"
            + (f", {self._znmf_groups.max() + 1} zero-net-flux groups" if self._znmf_groups is not None else "")
            + (f", nodes per width >= {nr[nr >= 0].min():.2f}" if (nr >= 0).any() else ""),
            flush=True,
        )

    def u_tau(self, xyz: np.ndarray) -> np.ndarray:
        """Friction velocity at the points (wall_units.u_tau: a value or a polynomial in one coordinate)."""
        spec = (self.cfg.get("wall_units") or {}).get("u_tau")
        if spec is None:
            raise ValueError("wall_units.u_tau is required by the u_tau / plus options of the YAML")
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if isinstance(spec, dict):
            s = xyz[:, "xyz".index(spec.get("axis", "x"))]
            return np.polynomial.polynomial.polyval(s, np.asarray(spec["polynomial"], dtype=float))
        return np.full(len(xyz), float(spec))

    def _check_dt(self, dt_cfg):
        """environment.dt (optional): the solver time step the configuration is made for (set in the par file)."""
        if dt_cfg is None:
            return
        dt = float(self.iface.info["dt"])
        if not np.isclose(dt, float(dt_cfg), rtol=1e-6, atol=0.0):
            raise ValueError(f"environment.dt = {float(dt_cfg):g}, but the solver runs at dt = {dt:g} (par file)")

    def _resolve_steps(self, spec) -> int:
        """steps_per_action: n, or {dt_plus: value} with the footprint-weighted mean u_tau of the actuators."""
        if not isinstance(spec, dict):
            return int(spec)
        w = np.asarray(self.actuators["flux_per_unit"], dtype=float)
        ut = float(np.sum(w * self.u_tau(self.actuators["xyz"])) / np.sum(w))
        dt = float(self.iface.info["dt"])
        n = max(1, int(round(float(spec["dt_plus"]) * self.nu / ut**2 / dt)))
        print(f"[HGNEK] steps_per_action {n} (dt+ {spec['dt_plus']} at mean u_tau {ut:.4f}, dt {dt:g})", flush=True)
        return n

    def _setup_probes(self):
        p = self.cfg["probes"]
        pos = dict(p["positions"])
        nd = pos.get("normal_distance")
        if isinstance(nd, dict):
            pos["normal_distance"] = float(nd["plus"]) * self.nu / self.u_tau(self.actuators["xyz"])
        xyz, extra = layout.positions(pos, self.iface, self.actuators)
        self.probe_fields = list(p["fields"])
        res = self.iface.define_probes(xyz, self.probe_fields, p.get("subtract_plane_mean"))
        # the actuator a probe belongs to (wall position, normal, footprint), if placed by actuator_offset
        act = extra.get("actuator")
        at = (lambda k: self.actuators[k][act]) if act is not None else None

        norm = p.get("normalization", 1.0)
        nf = len(self.probe_fields)
        if norm == "u_tau":
            ut = self.u_tau(at("xyz") if at else xyz)
            self._obs_div = np.repeat(ut[:, None], nf, axis=1)
        else:
            self._obs_div = np.tile(np.broadcast_to(np.asarray(norm, dtype=float), (nf,)), (len(xyz), 1))
        self.obs_scale = self._obs_div.mean(axis=0)  # per field (exact unless normalization is u_tau)

        self._frame = None
        self._frame_kind = p.get("frame", "global")
        if self._frame_kind not in ("global", "wall", "hg_fortran"):
            raise ValueError(f"probes.frame must be global, wall or hg_fortran, got {self._frame_kind!r}")
        if self._frame_kind in ("wall", "hg_fortran"):
            if at is None or "normal" not in self.actuators:
                raise ValueError(
                    f"probes.frame {self._frame_kind} needs actuator_offset probes on boundary_nodes actuators"
                )
            if "u" not in self.probe_fields or "v" not in self.probe_fields:
                raise ValueError(f"probes.frame {self._frame_kind} needs the fields u and v")
            nrm = np.array(at("normal")[:, :2], dtype=float)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            tan = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
            tan[tan[:, 0] < 0] *= -1  # downstream tangent
            self._frame = (self.probe_fields.index("u"), self.probe_fields.index("v"), tan, nrm)
            extra.update(tangent=tan, normal=nrm)

        self._probe_groups = None
        if p.get("subtract_group_mean"):
            self._probe_groups = layout.group_ids(xyz, p["subtract_group_mean"], p.get("group_tol"))
            self._probe_weights = at("flux_per_unit") if at else None
            extra["group"] = self._probe_groups

        self.probes = {"xyz": xyz, **extra, "scale": self._obs_div[:, 0]}
        self._write_table("probes.csv", self.probes)
        print(
            f"[HGNEK] {len(xyz)} probes x {self.probe_fields}, {res['n_border']} on element borders"
            + (f", {self._probe_groups.max() + 1} mean groups" if self._probe_groups is not None else "")
            + (
                (", hg_fortran projection" if self._frame_kind == "hg_fortran" else ", wall frame")
                if self._frame
                else ""
            ),
            flush=True,
        )

    def _setup_reward(self):
        r = self.cfg["reward"]
        if r["type"] != "wall_shear":
            raise ValueError(f"unknown reward type {r['type']} (override compute_reward for custom rewards)")
        self.monitors = [(int(r["boundary"]), r.get("quantity", "gradient"))]
        self.monitors += [(int(m["boundary"]), m.get("quantity", "force")) for m in self.cfg.get("monitors") or []]
        self.monitor_area = self.iface.define_monitors(self.monitors)

    def _setup_power(self, weight, cost, sampling):
        spec = self.cfg.get("power")
        spec = {"enabled": False} if spec is False else dict(spec or {})
        self.power_config = PowerConfig.from_config(spec, power_weight=weight, power_cost=cost, power_sampling=sampling)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self._power = None
        self._drag = None
        self._pavg = None
        if not spec.get("enabled", True):
            if self.power_config.weight != 0.0:
                raise ValueError("power_weight > 0 needs the actuator power (power.enabled)")
            return
        self._power_bids = [int(b) for b in spec.get("boundaries", self.cfg["actuators"]["boundaries"])]
        geo = [self.iface.wall_data(b, geometry=True, traction=False) for b in self._power_bids]
        w = np.concatenate([g["w"] for g in geo])
        ref = reference_power(self.iface.info["rho"], spec.get("reference"), float(w.sum()))
        self._power = WallPower(
            np.concatenate([g["xyz"] for g in geo]),
            w,
            np.concatenate([g["normal"] for g in geo]),
            self.actuators["xyz"],
            ref,
            spec.get("pressure_mean", ["x", "y"]),
            spec.get("group_tol"),
        )
        drag = spec.get("drag")
        if drag:
            axis = "xyz".index(drag.get("axis", "x"))
            cols = []
            for b in drag["boundaries"]:
                mon = [i for i, (bid, kind) in enumerate(self.monitors) if bid == int(b) and kind == "force"]
                if not mon:
                    raise ValueError(
                        f"power.drag: boundary {b} needs a force monitor (monitors, or the reward with quantity force)"
                    )
                cols += [4 + 6 * mon[0] + axis, 4 + 6 * mon[0] + 3 + axis]  # viscous and pressure force
            u = float((spec.get("reference") or {}).get("velocity", 1.0))
            self._drag = (cols, ref / u)  # 0.5 rho U^2 A
        print(
            f"[HGNEK] actuator power on boundaries {self._power_bids}: {len(w)} nodes, area {w.sum():.6g}, "
            f"reference power {ref:.6g}, sampling {self.power_sampling}, weight {self.power_config.weight} "
            f"({self.power_config.cost})" + (f", drag from columns {self._drag[0]}" if self._drag else ""),
            flush=True,
        )

    def actuation_power_now(self) -> Dict:
        """C_P forms of the current instant (pressure and imposed wall velocity of the power boundaries)."""
        d = [self.iface.wall_data(b, geometry=False, velocity=True, traction=False) for b in self._power_bids]
        return self._power.evaluate(np.concatenate([x["p"] for x in d]), np.concatenate([x["u_wall"] for x in d]))

    def _power_info(self, pw: Dict, sampling: str, rows: Optional[np.ndarray] = None, objective=None) -> Dict:
        cd = None
        if self._drag is not None and rows is not None:
            cols, fref = self._drag
            f = rows[:, cols].sum(axis=1)
            cd = float((f.mean() if sampling == "mean" else f[-1]) / fref)
        info = power_info(
            pw["per_actuator"],
            sampling=sampling,
            drag_coefficient=cd,
            objective=objective,
            forms={k: pw[k] for k in POWER_COSTS},
        )
        info["actuation_power_full_pressure"] = float(pw["signed_full_pressure"])
        return info

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
        self.iface.start_time_steps(n)
        return self._advance_wait()

    def _advance_wait(self) -> np.ndarray:
        rows = self.iface.wait_time_steps()
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
        obs = self.transform_probes(vals)
        return obs, t, cfl

    def transform_probes(self, vals: np.ndarray) -> np.ndarray:
        """Raw probe values [n_probes, n_fields] -> observation (frame, group mean, normalization), probe-major."""
        vals = np.array(vals, dtype=float)
        if self._frame is not None:
            iu, iv, tan, nrm = self._frame
            u, v = vals[:, iu].copy(), vals[:, iv].copy()
            if self._frame_kind == "hg_fortran":
                # projection of drl_state.f in the HydroGym Nek5000 wing case, kept for compatibility with policies
                # trained there: Ut = vx*n_y + vy*n_x; Vn = vx*n_x + vy*n_y as in the wall frame
                vals[:, iu] = nrm[:, 1] * u + nrm[:, 0] * v
            else:
                vals[:, iu] = tan[:, 0] * u + tan[:, 1] * v
            vals[:, iv] = nrm[:, 0] * u + nrm[:, 1] * v
        if self._probe_groups is not None:
            vals -= layout.group_mean(vals, self._probe_groups, self._probe_weights)
        return (vals / self._obs_div).astype(np.float32).ravel()  # [fields of probe 0, probe 1, ...]

    def wall_velocity(self, action: np.ndarray) -> np.ndarray:
        """Actuator wall velocities for actions in [-1, 1] (amplitude, then the per-group zero net flux)."""
        vel = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1) * self.amplitude
        if self._znmf_groups is not None:
            vel = vel - layout.group_mean(vel, self._znmf_groups, self.actuators["flux_per_unit"])
        return vel

    def reset(self, seed=None, options=None):
        """
        Start an episode from an initial condition of the library (see initial_conditions); with source
        continue the flow runs on from its current state (as RSETS in the Nek5000 coupling).
        ``options={"index": i}`` selects a library entry explicitly.
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
        info = {"time": t, "cfl": cfl, **{k: v for k, v in rec.items() if k in ("index", "state_time", "file")}}
        if self._power is not None:  # instantaneous power of the wall velocity now imposed
            info.update(self._power_info(self.actuation_power_now(), "end"))
        return obs, info

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
        self.step_async(action)
        return self.step_wait()

    def step_async(self, action) -> None:
        """First half of step(): set the actuation and start the solver steps without waiting for them."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        if action.size != self.n_actuators:
            raise ValueError(f"action size {action.size}, expected {self.n_actuators}")
        self._t0 = time.time()
        self._diag = self.iface.setControlProperties(self.wall_velocity(action))
        power_mean = self._power is not None and self.power_sampling == "mean"
        self._single = power_mean or self._step_hook is not None
        if self._single:
            # one solver step per request (the flow is the same as with n steps at once): the power after each
            # step (power_sampling mean) and the step hook (set_step_hook) at its solver steps
            self._pavg = PowerAverage() if power_mean else None
            self._pparts, self._pdone = [], 0
            self.iface.start_time_steps(1)
        else:
            self.iface.start_time_steps(self.steps_per_action)

    def set_step_hook(self, every: int, fn=None) -> None:
        """
        Call fn(env, row) after every solver step of an action whose global step index (solver_steps.csv tstep)
        is a multiple of `every`; row = [tstep, time, dt, cfl, monitors...] of that step. The solver is idle
        during the call, so fn may make requests such as iface.wall_data. With a hook the actions advance one
        solver step per request, which gives the same flow. every = 0 (or fn None) removes the hook. Only the
        steps of step() / step_async() call it, not those of reset().
        """
        every = int(every)
        if every < 0:
            raise ValueError(f"set_step_hook: every = {every} < 0")
        self._step_hook, self._step_hook_every = (fn, every) if (fn is not None and every > 0) else (None, 0)

    def power_step_progress(self) -> bool:
        """
        One-step mode (power_sampling mean or a step hook): wait for the running solver step, sample the power
        and call the step hook, then start the next step of the action; True once all steps of the action are
        done (then step_wait returns at once). Lets several instances advance in lockstep (multisim).
        """
        if not self._single or self._pdone >= self.steps_per_action:
            return True
        r = self._advance_wait()
        self._pparts.append(r)
        self._pdone += len(r)
        if self._pavg is not None:
            self._pavg.add(self.actuation_power_now(), weight=float(r[-1, 2]))  # weighted by dt
        if self._step_hook is not None and int(r[-1, 0]) % self._step_hook_every == 0:
            self._step_hook(self, r[-1])  # the solver state is that after this step
        if self._pdone < self.steps_per_action:
            self.iface.start_time_steps(1)
            return False
        return True

    def step_wait(self):
        t0, diag = self._t0, self._diag
        pw = None
        if self._single:
            while not self.power_step_progress():
                pass
            rows = np.concatenate(self._pparts)
            self._pparts, self._single = [], False
            if self._pavg is not None:
                pw, self._pavg = self._pavg.mean(), None
            elif self._power is not None:
                pw = self.actuation_power_now()
        else:
            rows = self._advance_wait()
            if self._power is not None:
                pw = self.actuation_power_now()

        reward, rinfo = self.compute_reward(rows)
        pinfo = {}
        if pw is not None:
            pinfo = self._power_info(pw, self.power_sampling, rows, objective=rinfo)
            reward = self.power_config.apply(reward, {k: pw[k] for k in POWER_COSTS})
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
            **pinfo,
        }
        skip = ("reward_per_actuator", "actuation_power_per_actuator", "objective")
        rec = {
            "episode": self.episode,
            "k": self.act_index,
            "reward": reward,
            **{k: v for k, v in info.items() if k not in skip},
        }
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
