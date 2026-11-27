"""Per-case settings of the released policies: environment, configuration, observation wiring and plotted quantity.

This module needs only numpy and PyYAML (no MPI, no m-AIA). It is shared by ``prepare_run.py`` (workspace),
``run_policy.py`` (roll-outs, MPMD application) and ``plot_policy.py`` (figures).

A policy id of ``policies/MANIFEST.yaml`` has the form ``<case>/<variant>/seed<k>``. ``CASES[<case>]`` gives:

- ``env``: the HydroGym m-AIA environment (``None``: the id from the MANIFEST entry, ``hydrogym_env.id``);
- ``config``: the ``maia`` settings of ``environment_config.yaml`` the policy runs with (merged into the
  environment's own file; ``"policy"`` = take the value from the policy file / MANIFEST loader block);
- ``properties``: m-AIA property lines changed in the run directory's ``properties_run.toml``;
- ``observation``: how the probe data of the environment become the policy input (``OBSERVATIONS``);
- ``quantity``: the plotted quantity (``drag``, ``lift`` or ``tke``), with ``ylabel`` and ``ylim`` of the figure;
- ``n_actions`` / ``start`` / ``window``: the default roll-out (controlled actions, uncontrolled actions before
  control, actions averaged in the summary);
- ``excerpt``: the part of the run shown in the default figure, (start, end) in convective time after the reset
  (``excerpt_indices``; gust), mapped onto a normalised 0-1 time axis; without it the whole run;
- ``tke_skip``: TKE figures (cavity) start this many TKE samples (= actions) after the first one, i.e. the 0-1 time
  axis begins there (cavity Re 4200: 37 samples, 0.7 D/U, where the controlled and uncontrolled TKE still coincide);
  the TKE is normalised by the mean uncontrolled TKE over the plotted samples, in the figure and in the summary;
- ``episode_end``: TKE cases, the action (counted from control onset) at whose end the summary's
  ``episode_end_ratio`` is taken (default 200; cavity Re 4200: 237, the end of the default roll-out and the last
  point of its figure);
- ``variants``: settings that differ for one variant of the case (``<case>/<variant>/seed<k>``).

``PLOT_ONLY_CASES`` holds the figure settings of the Kolmogorov policies (HydroGym JAX solver). The figure style
(quantity, axis labels and limits, summary window, policy label and colour) comes from here and from the policy id,
not from the run file.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
POLICIES = os.path.join(REPO, "policies")
if not os.path.isfile(os.path.join(POLICIES, "load_policy.py")):
    # The examples were copied out of the source tree (e.g. `cp -r /opt/hydrogym/examples ~/hydrogym-examples`, as
    # the container notes suggest for read-only images): use the policies/ of the installed (editable) HydroGym.
    import hydrogym

    POLICIES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(hydrogym.__file__))), "policies")
if POLICIES not in sys.path:
    sys.path.insert(0, POLICIES)

from load_policy import LOADERS, _file, entry, load_policy, load_probes  # noqa: E402

# ------------------------------------------------------------------------------------------------ colours
# Paper colours. The uncontrolled run is always #2E2E2E. Any other policy family uses the neutral brown #8C564B.
COLORS = {
    "Uncontrolled": "#2E2E2E",
    "PPO": "#1f77b4",
    "GPPO": "#2ca02c",
    "DDPG": "#d62728",
    "TD3": "#9467bd",
    "other": "#8C564B",
}

# ------------------------------------------------------------------------------------------------ cases
_DRAG = dict(quantity="drag", ylabel="Drag coefficient $C_D$")
# 2D pinball controllers with force observation: m-AIA writes the wall forces of every lattice step
# (WakeForces reads them from <forceFileName><segment>.csv in the run directory)
_FORCE_LOG = {"calculateWallForces": "true", "calculateWallForcesInterval": 1}
# n_actions and window count from the end of the controller's hold_actions (constant base rotation), which the
# roll-out adds in front (after_hold)
_POWER = dict(observation="wake_forces", properties=_FORCE_LOG, n_actions=600, window=(300, 600), after_hold=True)
CASES = {
    "pinball_2d_re100": dict(
        _DRAG,
        env="Pinball_2D_Re100",
        config={"max_control": "policy", "omega": 0.0},
        observation="wake",
        probes="probes/pinball_2d_wake_probes.json",
        n_actions=200,
        window=(100, 200),
        ylim=(-0.1, 3.7),
        # ppo_power: max_control 0.25 from the controller file
        variants={"ppo_power": dict(_POWER)},
    ),
    "pinball_2d_re150": dict(
        _DRAG,
        env="Pinball_2D_Re150",
        config={"max_control": 0.4, "num_sim_substeps_per_actuation": 152, "omega": 0.0},
        observation="wake",
        probes="probes/pinball_2d_wake_probes.json",
        n_actions=600,
        window=(300, 600),
        ylim=(-0.1, 3.7),
        variants={"ppo_power": dict(_POWER)},
    ),
    "pinball_3d_re150": dict(
        # the default configuration of Pinball_3D_Re150 (Ma 0.1, max_control 0.35, 352 steps per action)
        _DRAG,
        env="Pinball_3D_Re150",
        config={"omega": 0.0},
        observation="wake3d",
        probes="probes/pinball_3d_wake_probes.json",
        scale_2d_to_3d=1.6,
        n_actions=100,
        window=(50, 100),
        ylim=(-0.1, 3.7),
    ),
    "cylinder_2d_re200": dict(
        _DRAG,
        config={"max_control": 0.017320508, "num_sim_substeps_per_actuation": 88},
        observation="wake",
        n_actions=100,
        window=(50, 100),
        ylim=(0.6, 1.6),
    ),
    "cylinder_2d_re1000": dict(
        _DRAG,
        config={"max_control": 0.017320508, "num_sim_substeps_per_actuation": 224},
        observation="wake",
        n_actions=100,
        window=(50, 100),
        ylim=(0.6, 2.2),
    ),
    "cylinder_2d_re3900": dict(
        _DRAG,
        config={"max_control": 0.017320508, "num_sim_substeps_per_actuation": 496},
        observation="wake",
        n_actions=100,
        window=(50, 100),
        ylim=(0.3, 3.6),
    ),
    "square_cylinder_2d_re200": dict(
        _DRAG,
        config={"max_control": 0.021},
        observation="wake",
        n_actions=100,
        start=67,
        window=(50, 100),
        ylim=(0.8, 1.8),
    ),
    "naca0012_gust_re1000": dict(
        quantity="lift",
        ylabel="Lift coefficient $C_L$",
        env="NACA0012Gust_3D_Re1000_AOA20",
        config={"max_control": "policy", "num_sim_substeps_per_actuation": "policy"},
        env_options={"gust": "fig2c"},  # the gust of Fig. 2c (gust_fig2c.bin of the case), at every reset
        observation="gust",
        gust_planes=dict(n=8, z=(-2.0, 2.0)),
        n_actions=200,
        window=(0, 200),
        # The default figure shows an excerpt of the gust passage, in convective time c/U after the reset (the gust
        # is superposed on the start state at every reset): from 2.53 c/U, during the rise of the lift, to 5.0 c/U,
        # when the tail of the gust reaches the leading edge (gust length 4 c, front 1 c upstream of the leading edge
        # at the reset, convected at U). The lift sample i of a run (counted from 0) is taken at the end of action
        # i + 1, at (i + 1) dt_action: the excerpt is actions 70-139 in 3D (0.0361 c/U per action) and 93-185 in 2D
        # (0.0271 c/U per action), counted from 1. The roll-out and its summary are the whole run.
        excerpt=(2.53, 5.0),
        ylim=(-1.5, 4.9),
    ),
    "cavity_2d_re4200": dict(
        quantity="tke",
        ylabel="TKE / mean uncontrolled TKE",
        observation="cavity2d",
        tke_window=1.0,
        # 200 actions + the 37 that the figure skips (tke_skip): the plotted part is as long as a 200-action run
        n_actions=237,
        window=(0, 237),
        # the policies' first actions are tiny (|a| about 0.002): for 37 TKE samples (0.7 D/U, one sample per action)
        # the controlled and the uncontrolled TKE differ by less than 0.01 and coincide in the figure. The figure and
        # the summary use the samples from 37 on: TKE / mean uncontrolled TKE over them, episode_end_ratio at the end
        # of action 237 (the last plotted point)
        tke_skip=37,
        episode_end=237,
        ylim=(0.0, 2.0),
    ),
    "cavity_2d_re7500": dict(
        quantity="tke",
        ylabel="TKE / mean uncontrolled TKE",
        observation="cavity2d",
        tke_window=1.0,
        n_actions=200,
        window=(0, 200),
        ylim=(0.0, 2.0),
    ),
    "cavity_3d_re7500": dict(
        quantity="tke",
        ylabel="TKE / mean uncontrolled TKE",
        observation="cavity3d",
        tke_window=2.0,
        n_actions=3156,
        window=(0, 3156),
        ylim=(0.0, 1.6),
    ),
    "cavity_3d_re4200": dict(
        quantity="tke",
        ylabel="TKE / mean uncontrolled TKE",
        observation="cavity3d",
        tke_window=2.0,
        n_actions=1052,
        window=(0, 1052),
        ylim=(0.0, 1.6),
    ),
}


# Plot settings of released policies that run in another solver (examples/jax/released_policies).
PLOT_ONLY_CASES = {
    "kolmogorov_re40": dict(
        quantity="dissipation",
        ylabel=r"Dissipation $\varepsilon$",
        xlabel="Time $t$",
        ylim=(0.0, 0.45),
        action_ylim=(-0.45, 0.2),
    ),
}


def split_id(pid):
    parts = pid.split("/")
    if len(parts) != 3:
        raise ValueError(f"policy id {pid!r}: expected <case>/<variant>/seed<k> (see policies/MANIFEST.yaml)")
    return parts


def case_of(pid):
    case, variant, _ = split_id(pid)
    if case not in CASES:
        raise KeyError(f"{pid}: no m-AIA example settings for case {case!r} (cases: {', '.join(sorted(CASES))})")
    c = {k: v for k, v in CASES[case].items() if k != "variants"}
    c.update(CASES[case].get("variants", {}).get(variant, {}))
    if c.pop("after_hold", False):  # roll-out = the controller's hold_actions + n_actions; window after the hold
        h = int(getattr(load_policy(pid), "hold_actions", 0))
        c["n_actions"] = h + c["n_actions"]
        c["window"] = (h + c["window"][0], h + c["window"][1])
    return case, c


def plot_case(pid):
    """(case, settings) for the figures of a policy: the m-AIA cases and the plot-only cases"""
    case = split_id(pid)[0]
    if case in PLOT_ONLY_CASES:
        return case, dict(PLOT_ONLY_CASES[case])
    return case_of(pid)


def algorithm_family(pid):
    """(label, colour) of a policy from the variant of its id: PPO / GPPO / TD3 / DDPG in the paper colours, others
    in the neutral colour."""
    variant = split_id(pid)[1].lower()
    text = variant.upper()
    if variant.startswith("gppo"):
        return "GPPO", COLORS["GPPO"]
    for fam in ("TD3", "DDPG", "PPO"):
        if fam in text:
            return fam, COLORS[fam]
    return "policy", COLORS["other"]


def env_name(pid, dim=2):
    case, c = case_of(pid)
    env = c.get("env")
    if isinstance(env, dict):
        return env[dim]
    if env:
        return env
    e = entry(pid)["hydrogym_env"]
    return str(e.get("id_3d") if dim == 3 and e.get("id_3d") else e["id"])


def env_dim(name):
    return 3 if "_3D_" in name else 2


def load_for_run(pid, dim=2, file_role=None):
    """The policy as used in the roll-out. file_role: another file of the entry in the loader's format (e.g.
    'weights_best_eval')."""
    case, c = case_of(pid)
    if file_role:
        e = entry(pid)
        pi = LOADERS[e["loader"]["format"]](_file(e, file_role))
        pi.id, pi.manifest_entry = pid, e
        if getattr(pi, "max_control", None) is None and "max_control" in e["loader"]:
            pi.max_control = e["loader"]["max_control"]
        return pi
    return load_policy(pid)


def maia_config(pid, pi, dim=2):
    """maia section overrides of environment_config.yaml for this policy"""
    case, c = case_of(pid)
    e = entry(pid)
    out = {}
    for k, v in c.get("config", {}).items():
        if v == "policy":
            if k == "max_control":
                v = getattr(pi, "max_control", None) or (e.get("loader") or {}).get("max_control")
            elif k == "num_sim_substeps_per_actuation":
                v = getattr(pi, "substeps", None)
            if v is None:
                continue
        out[k] = float(v) if k == "max_control" else v
    return out


# ------------------------------------------------------------------------------------------------ observations
class ObsMap:
    """probe data of the environment -> policy action.

    probes(): (n, dim) probe coordinates the env is asked for; reset(): start of a controlled episode;
    act(P, k, U): action in [-1, 1]^n from the probe data P = env.probeData (2D: u, v, rho, p; 3D: u, v, w, rho, p,
    lattice units) after the last step, k = action index from control onset, U = Ma / sqrt(3);
    features(P, U): per-step record for the paper quantity (cavity: the shear-layer observation obs), else None.
    """

    def __init__(self, pi, case_cfg, env_dir):
        self.pi, self.cfg, self.env_dir = pi, case_cfg, env_dir

    def reset(self):
        if hasattr(self.pi, "reset"):
            self.pi.reset()

    def features(self, P, U):
        return None

    def after_step(self, env, info):
        """called after every env.step (info: the step's info dict)"""


class Wake(ObsMap):
    """[u/U, v/U] at the wake probes (+ previous action when the network expects it): pinball 2D, cylinders"""

    def probes(self):
        e = self.pi.manifest_entry
        f = (e.get("observation") or {}).get("probes_file") or self.cfg.get("probes")
        xy = load_probes(f)
        self.n = len(xy)
        self.prev_action = self.pi.obs_dim == 2 * self.n + self.pi.n_actions
        return xy

    def reset(self):
        super().reset()
        self.prev = np.zeros(self.pi.n_actions)

    def act(self, P, k, U):
        o = [P[: self.n, 0] / U, P[: self.n, 1] / U] + ([self.prev] if self.prev_action else [])
        a = np.asarray(self.pi(np.concatenate(o)), float).reshape(-1)
        self.prev = a
        return a


class WakeForces(Wake):
    """2D pinball controllers with force observation: [u/U, v/U] at the wake probes, the previous action and the drag,
    lift and shaft-power coefficients of the three cylinders from the previous action interval. C_D and C_L are the
    means of the per-step wall forces that m-AIA has appended to <forceFileName><segment>.csv in the run directory
    since the previous action (calculateWallForcesInterval 1, set by the case's properties), normalised as in the
    environment (after the first action of an episode: the force at its end); C_P is the environment's actuation
    power per cylinder (info "actuation_power_per_actuator")."""

    def probes(self):
        xy = super().probes()
        txt = open(os.path.join(self.env_dir, "properties_run.toml")).read()
        m = re.search(r'^\s*forceFileName\s*=\s*"([^"]*)"', txt, re.M)
        self.stem = os.path.join(self.env_dir, m.group(1) if m else "forces.log")
        return xy

    def reset(self):
        super().reset()
        self.f = np.zeros(3 * self.pi.n_actions)
        # skip what was written before the reset (an earlier roll-out)
        self.pos = {f: os.path.getsize(f) for f in glob.glob(self.stem + "*.csv")}
        self.first = True

    def _new_rows(self, path):
        rows = []
        if os.path.exists(path):
            with open(path, "rb") as fh:
                fh.seek(self.pos.get(path, 0))
                while True:
                    line = fh.readline().decode()
                    if not line.endswith("\n"):  # end of file (or a line still being written)
                        break
                    self.pos[path] = fh.tell()
                    if line.strip() and not line.startswith("#"):
                        rows.append([float(x) for x in line.replace(",", " ").split()])
        if not rows:
            raise RuntimeError(f"{path}: no new wall-force rows (calculateWallForces in properties_run.toml?)")
        return np.asarray(rows)

    def after_step(self, env, info):
        U = env.Ma / np.sqrt(3.0)
        norm = 2.0 / (U**2 * env.referenceLength / env.dX * env.zLength / env.dX)
        rows = [self._new_rows(f"{self.stem}{b}.csv")[:, 2:4] for b in env.bcId]
        if self.first:
            # m-AIA appends the rows of an action interval during the next one: after the first action of an
            # episode the new rows are those of the steps of env.reset(), which do not belong to the episode, and the
            # force at the end of the first action is used instead
            self.first = False
            F = np.stack([np.asarray(env.maiaInterface.getForce(b), float)[:2] for b in env.bcId]) * norm
        else:
            F = np.stack([r.mean(0) for r in rows]) * norm
        cp = np.asarray(info["actuation_power_per_actuator"], float).reshape(-1)
        self.f = np.concatenate([F[:, 0], F[:, 1], cp])

    def act(self, P, k, U):
        o = np.concatenate([P[: self.n, 0] / U, P[: self.n, 1] / U, self.prev, self.f])
        a = np.asarray(self.pi(o), float).reshape(-1)
        self.prev = a
        return a


class Wake3D(ObsMap):
    """3D pinball: the 40 wake probes on 4 spanwise planes (z-major), averaged over the planes; previous action and
    policy output in 2D units, 3D command a3 = clip(scale a2, -1, 1)"""

    def probes(self):
        e = self.pi.manifest_entry
        f = (e.get("observation") or {}).get("probes_file") or self.cfg["probes"]
        xyz = load_probes(f)
        self.n = (self.pi.obs_dim - self.pi.n_actions) // 2
        self.nz = len(xyz) // self.n
        self.scale = float((e.get("action") or {}).get("scale_2d_to_3d", self.cfg["scale_2d_to_3d"]))
        return xyz

    def reset(self):
        super().reset()
        self.prev = np.zeros(self.pi.n_actions)

    def act(self, P, k, U):
        m = self.n * self.nz
        u = P[:m, 0].reshape(self.nz, self.n).mean(0) / U
        v = P[:m, 1].reshape(self.nz, self.n).mean(0) / U
        a2 = np.asarray(self.pi(np.concatenate([u, v, self.prev])), float).reshape(-1)
        a3 = np.clip(self.scale * a2, -1.0, 1.0)
        self.prev = a3 / self.scale
        return a3


class Gust(ObsMap):
    """NACA0012 gust: u/U, v/U at the 40 probes of the controller file (2D), in 3D averaged over spanwise planes"""

    def probes(self):
        xy = np.asarray(self.pi.probes_case, float)
        self.n = len(xy)
        if self.dim == 2:
            self.nz = 1
            return xy
        g = self.cfg["gust_planes"]
        z0, z1 = g["z"]
        zs = z0 + (np.arange(g["n"]) + 0.5) * (z1 - z0) / g["n"]
        self.nz = len(zs)
        return np.array([(x, y, z) for z in zs for (x, y) in xy])

    def act(self, P, k, U):
        m = self.n * self.nz
        u = P[:m, 0].reshape(self.nz, self.n).mean(0) / U
        v = P[:m, 1].reshape(self.nz, self.n).mean(0) / U
        return np.asarray(self.pi(np.concatenate([u, v]), k), float).reshape(-1)


class Cavity2D(ObsMap):
    """2D cavity controller spec: the 224 points 'pts' of the spec; obs, s = observe(u/U, v/U); a = pi(obs, s)"""

    def probes(self):
        self.n = len(self.pi.pts)
        return np.asarray(self.pi.pts, float)

    def _obs(self, P, U):
        return self.pi.observe(P[: self.n, 0] / U, P[: self.n, 1] / U)

    def act(self, P, k, U):
        obs, s = self._obs(P, U)
        return np.asarray(self.pi(obs, s), float).reshape(-1)

    def features(self, P, U):
        return self._obs(P, U)[0]


class Cavity3D(ObsMap):
    """3D cavity: the 220 cells of each of the 8 policy planes (probes_8planes.txt of the environment), per plane
    obs = [Wp u/U, Wp v/U] (observation_weights.json); a = pi(P_z) with P_z (nz, 100)"""

    def probes(self):
        w = json.load(open(os.path.join(self.env_dir, "observation_weights.json")))
        self.Wp = np.asarray(w["Wp"], float)
        self.nc, self.nz = int(w["n_cells2d"]), int(w["nz"])
        self.planes = np.asarray(w["policy_planes"], int)
        if not np.array_equal(self.planes, np.asarray(self.pi.slices, int)):
            raise ValueError(f"policy planes {self.pi.slices} != environment planes {self.planes}")
        return np.loadtxt(os.path.join(self.env_dir, "probes_8planes.txt"), ndmin=2)

    def _planes(self, P, U):
        q = P[: len(self.planes) * self.nc, :2].reshape(len(self.planes), self.nc, 2) / U
        return np.concatenate([q[..., 0] @ self.Wp.T, q[..., 1] @ self.Wp.T], -1)

    def act(self, P, k, U):
        Pz = np.zeros((self.nz, 2 * self.Wp.shape[0]))
        Pz[self.planes] = self._planes(P, U)
        return np.asarray(self.pi(Pz), float).reshape(-1)

    def features(self, P, U):
        return self._planes(P, U).mean(0)  # spanwise mean over the policy planes


OBSERVATIONS = {
    "wake": Wake,
    "wake_forces": WakeForces,
    "wake3d": Wake3D,
    "gust": Gust,
    "cavity2d": Cavity2D,
    "cavity3d": Cavity3D,
}


def make_observation(pid, pi, env_dir, dim):
    case, c = case_of(pid)
    ad = OBSERVATIONS[c["observation"]](pi, c, env_dir)
    ad.dim = dim
    return ad


# ------------------------------------------------------------------------------------------------ paper quantity
def tke(obs, dt, window):
    """fluctuation energy of the shear-layer observation: mean over the components of their variance over a
    trailing window (in D/U). obs (n, m) at t_j = j dt. Returns (t of the window end, E)."""
    w = max(2, int(round(window / dt)))
    c1 = np.cumsum(np.vstack([np.zeros(obs.shape[1]), obs]), 0)
    c2 = np.cumsum(np.vstack([np.zeros(obs.shape[1]), obs**2]), 0)
    k = np.arange(w, len(obs) + 1)
    E = (((c2[k] - c2[k - w]) / w) - ((c1[k] - c1[k - w]) / w) ** 2).mean(1)
    return (k - 1) * dt, E


def excerpt_indices(case_cfg, R, n):
    """(i0, i1): first and last sample (inclusive, counted from control onset) of the excerpt of the default figure
    (``excerpt`` of the case, in time after the reset), or None (no excerpt, or the run does not reach its end).
    Sample i of a run is the end of action k0 + i + 1 after the reset, at time (k0 + i + 1) dt_action."""
    ex = case_cfg.get("excerpt")
    if not ex:
        return None
    dt, k0 = float(R["dt_action"]), int(R["start"])
    i0 = max(0, int(round(ex[0] / dt)) - 1 - k0)
    i1 = int(round(ex[1] / dt)) - 1 - k0
    return (i0, i1) if i0 < i1 < n else None


SERIES = {"drag": "C_D", "lift": "C_L", "dissipation": "dissipation"}  # quantity -> per-action series of a run file


def tke_skip(case_cfg, n):
    """the number of leading TKE samples that the figure and the normalisation leave out (``tke_skip`` of the case),
    for a series of n samples; 0 if the series is not longer than tke_skip + 1"""
    skip = int(case_cfg.get("tke_skip", 0))
    return skip if skip and n > skip + 1 else 0


def paper_series(R, run, case_cfg):
    """(time, value) of the paper quantity of one roll-out ('unc' or 'ctl') of a run file R (dict of arrays),
    from control onset. TKE: all samples, normalised by the mean uncontrolled TKE over the samples that the figure
    shows (from ``tke_skip`` on, up to the end of the run's n_actions)"""
    k0 = int(R["start"])
    q = case_cfg["quantity"]
    dt = float(R["dt_action"])
    if q in SERIES:
        y = np.asarray(R[f"{run}_{SERIES[q]}"], float)[k0:]
        n = len(y)
        return np.linspace(0, n * dt, n), y
    obs = np.asarray(R[f"{run}_O"], float)[k0:]  # row 0 = state at control onset, row j = after j actions
    t, E = tke(obs, dt, case_cfg["tke_window"])
    obs_u = np.asarray(R["unc_O"], float)[k0:]
    tu, Eu = tke(obs_u, dt, case_cfg["tke_window"])
    span = tu <= min(tu[-1], float(R["n_actions"]) * dt) + 1e-9
    span[: tke_skip(case_cfg, int(span.sum()))] = False  # the samples before the plotted part
    return t, E / float(np.mean(Eu[span]))


def summary(R, case_cfg, n_episode=None):
    """summary numbers of a run file (stored in its meta and printed by run_policy.py / plot_policy.py). TKE:
    ``episode_end_ratio`` = controlled TKE at the end of action n_episode (default: ``episode_end`` of the case, else
    200; counted from control onset) / mean uncontrolled TKE over the samples up to then, from ``tke_skip`` on (as in
    the figure); absent if the run is shorter"""
    k0 = int(R["start"])
    q = case_cfg["quantity"]
    out = {}
    if q == "drag":
        cd = np.asarray(R["ctl_C_D"], float)[k0:]
        cd0 = np.asarray(R["unc_C_D"], float)[k0:]
        w0, w1 = case_cfg["window"]
        w1 = min(w1, len(cd))
        if w1 > w0:
            out.update(C_D=float(cd[w0:w1].mean()), C_D0=float(cd0[w0:w1].mean()), window=[w0, w1])
            out["DR"] = 1.0 - out["C_D"] / out["C_D0"]
            cp = np.asarray(R["ctl_C_P"], float)[k0:][w0:w1]
            if np.isfinite(cp).any():
                out["C_P"] = float(np.nanmean(cp))
        out["max_C_D"] = float(cd.max())
    elif q == "lift":
        cl = np.asarray(R["ctl_C_L"], float)[k0:]
        clu = np.asarray(R["unc_C_L"], float)[k0:]
        cl0 = float(R["C_L0"])
        out.update(
            dCL_ratio=float(np.mean(np.abs(cl - cl0)) / np.mean(np.abs(clu - cl0))),
            peak_change_pct=float(100.0 * (cl.max() / clu.max() - 1.0)),
            C_L_max=float(cl.max()),
            C_L_max_uncontrolled=float(clu.max()),
        )
        a = np.asarray(R["ctl_action"], float)[k0:]
        out["CJ"] = float(np.mean(np.sum(np.abs(a) ** 3, 1) / a.shape[1]))
    elif q == "dissipation":
        d = np.asarray(R["ctl_dissipation"], float)[k0:]
        du = np.asarray(R["unc_dissipation"], float)[k0:]
        out.update(dissipation=float(d.mean()), dissipation_uncontrolled=float(du.mean()))
        out["dissipation_ratio"] = out["dissipation"] / out["dissipation_uncontrolled"]
    else:
        dt = float(R["dt_action"])
        obs = np.asarray(R["ctl_O"], float)[k0:]
        obs_u = np.asarray(R["unc_O"], float)[k0:]
        t, E = tke(obs, dt, case_cfg["tke_window"])
        tu, Eu = tke(obs_u, dt, case_cfg["tke_window"])
        tep = (n_episode or int(case_cfg.get("episode_end", 200))) * dt
        if len(E) and len(Eu) and tu[-1] >= tep - 1e-9 and t[-1] >= tep - 1e-9:
            span = tu <= tep + 1e-9
            span[: tke_skip(case_cfg, int(span.sum()))] = False  # as in the figure
            E0 = float(np.mean(Eu[span]))
            out["episode_end_ratio"] = float(E[int(np.argmin(np.abs(t - tep)))] / E0)
        tt = np.arange(len(obs)) * dt
        for a, b in ((20.0, 40.0), (40.0, 60.0)):
            m = (tt >= a) & (tt < b)
            if m.sum() > 20 and tt[-1] >= b - 2 * dt:
                out[f"fluct_ratio_t{a:g}_{b:g}"] = float(np.var(obs[m], 0).mean() / np.var(obs_u[m], 0).mean())
    a = np.asarray(R["ctl_action"], float)[k0:]
    out["mean_abs_action"] = float(np.abs(a).mean())
    return out


# ------------------------------------------------------------------------------------------------ run files
def drop_empty(d):
    """d without None and empty values (recursively for dicts)"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = drop_empty(v)
        if v is None or (isinstance(v, (dict, list, tuple, str)) and len(v) == 0):
            continue
        out[k] = v
    return out


def run_files(out):
    """(figure prefix, npz, json) of the output name of a run: run.npz -> run, run.npz, policy_run.json;
    ppo.npz -> ppo, ppo.npz, ppo_policy_run.json (in the directory of out)"""
    prefix = out[:-4] if out.endswith(".npz") else out
    d, stem = os.path.split(prefix)
    return prefix, prefix + ".npz", os.path.join(d, "policy_run.json" if stem == "run" else f"{stem}_policy_run.json")


def write_run_files(out, R, meta, record):
    """the npz (meta = JSON string) and the policy_run.json of a run; atomic"""
    _, npz, js = run_files(out)
    if os.path.dirname(npz):
        os.makedirs(os.path.dirname(npz), exist_ok=True)
    tmp = npz + ".tmp.npz"
    np.savez(tmp, meta=json.dumps(meta), **R)
    os.replace(tmp, npz)
    with open(js, "w") as fh:
        json.dump(record, fh, indent=1)
    return npz, js
