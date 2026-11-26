#!/usr/bin/env python3
"""Roll out one released policy and the uncontrolled flow in m-AIA, from the same start state.

MPMD application 0 (the controller). Prepare the run directory first (``prepare_run.py``), then, from it:

    mpirun -np 1 python run_policy.py --policy <id> [--actions N] [--start K] [--readable] [--store_data] \
        : -np 1 maia properties_run.toml

(GPU image: one m-AIA rank per GPU, i.e. 1 for the 2D cases and 4 on 4 GPUs for the 3D cases.)

One launch runs both roll-outs: the uncontrolled one (zero action), then ``env.reset()`` (m-AIA reloads the
restart) and the controlled one. ``--start K`` runs K uncontrolled actions before the policy is switched on (other
start states of the same flow); both roll-outs then contain the same K zero-action steps.

At the end, the figure is made from memory in the paper style (``plot_policy.py``): ``run.png`` (paper quantity;
gust: an excerpt of the gust passage, cavity: on a normalised time axis); ``--readable`` adds ``run_readable.png``
and ``run_actions_readable.png`` (the actions of the policy).

``--store_data`` also writes ``run.npz`` and ``policy_run.json`` (policy, environment and the settings of the run).
Per action, the npz holds for each roll-out (prefix ``unc_`` / ``ctl_``): the normalised action, the solver
command, the force coefficients of every body and their sums (``C_D``, ``C_L``; end of the action interval, the
normalisation of the environment's reward), the actuation power ``C_P`` reported by the environment (``info``;
NaN where the environment has no power model) and, for the cavity, the shear-layer observation ``O`` from which
the fluctuation energy (TKE) is computed. ``meta`` (JSON) holds the policy, environment, dimension, action count,
action interval and the physical settings; ``plot_policy.py`` plots the file. A run of one roll-out only
(``--runs unc`` or ``--runs ctl``, one part of a split run) always stores its data, for
``plot_policy.py ctl/run.npz unc/run.npz``.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_cases as PC  # noqa: E402


def make_env(spec, work, probes, nproc=None):
    """The HydroGym m-AIA environment of the run directory (its environment data = the run directory)."""
    import hydrogym.maia as maia

    name = spec["environment"]
    cls = maia.list_registered_types()[name.split("_", 1)[0]]

    class RunDirectoryEnv(cls):
        def _setup_environment_data(self):
            return work

    cfg = dict(
        environment_name=name,
        configuration_file=os.path.join(work, "environment_config.yaml"),
        probe_locations=[float(v) for v in np.asarray(probes, float).ravel()],
        obs_normalization_strategy="none",
        use_clean_cache=False,
        **spec.get("env_options", {}),
    )
    if nproc:
        cfg["nproc"] = nproc
    return RunDirectoryEnv(cfg)


def coefficients(env):
    """force coefficients (n_body, nDim) at the end of the last action, normalised as in the reward"""
    U = env.Ma / np.sqrt(3.0)
    proj = env.referenceLength / env.dX * env.zLength / env.dX
    F = np.stack([np.asarray(env.maiaInterface.getForce(b), float) for b in env.bcId])
    return 2.0 * F / (U**2 * proj)


def rollout(env, ad, controlled, n_total, k0, log, tag, save):
    U = env.Ma / np.sqrt(3.0)
    rec = {k: [] for k in ("action", "command", "C_body", "C_P", "O")}
    env.reset()
    ad.reset()
    P = np.asarray(env.probeData, float)
    f = ad.features(P, U)
    if f is not None:
        rec["O"].append(f)
    t0 = time.time()
    for k in range(n_total):
        if controlled and k >= k0:
            if k == k0 and k0 > 0:
                ad.reset()
            a = np.asarray(ad.act(P, k - k0, U), float).reshape(-1)
        else:
            a = np.zeros(env.num_inputs)
        _, _, _, _, info = env.step(a)
        ad.after_step(env, info)
        P = np.asarray(env.probeData, float)
        if not np.isfinite(P).all():
            raise FloatingPointError(f"{tag}: non-finite probe data after action {k}")
        rec["action"].append(a)
        rec["command"].append(np.asarray(env.convert_action([x * env.MAX_CONTROL for x in a]), float))
        rec["C_body"].append(coefficients(env))
        rec["C_P"].append(float(info.get("actuation_power", np.nan)))
        f = ad.features(P, U)
        if f is not None:
            rec["O"].append(f)
        if (k + 1) % 25 == 0 or k + 1 == n_total:
            cb = rec["C_body"][-1]
            log(
                f"[{tag} {time.time() - t0:7.1f} s] action {k + 1}/{n_total}  a {np.round(a, 3).tolist()}  "
                f"C_D {cb[:, 0].sum():.4f}  C_L {cb[:, 1].sum():.4f}"
            )
            save(tag, rec)
    return rec


def to_arrays(tag, rec):
    out = {}
    if rec["action"]:
        cb = np.asarray(rec["C_body"], float)
        out.update(
            {
                f"{tag}_action": np.asarray(rec["action"], float),
                f"{tag}_command": np.asarray(rec["command"], float),
                f"{tag}_C_D_body": cb[..., 0],
                f"{tag}_C_L_body": cb[..., 1],
                f"{tag}_C_D": cb[..., 0].sum(1),
                f"{tag}_C_L": cb[..., 1].sum(1),
                f"{tag}_C_P": np.asarray(rec["C_P"], float),
            }
        )
    if rec["O"]:
        out[f"{tag}_O"] = np.asarray(rec["O"], float)
    return out


def run_meta(pid, spec, n, dt_action, env=None, file_role=None):
    """meta of the npz: policy, environment, dimension, protocol and the physical settings (no plot styling)"""
    meta = dict(policy=pid, environment=spec["environment"], dim=int(spec["dim"]), n_actions=int(n))
    meta["dt_action"] = float(dt_action)
    if env is not None:
        meta.update(
            max_control=float(env.MAX_CONTROL),
            substeps=int(env.num_substeps_per_iteration),
            Ma=float(env.Ma),
            Re=float(env.Re),
        )
    return PC.drop_empty(dict(meta, file_role=file_role))


def run_record(pid, spec, n, k0, runs, file_role=None):
    """policy_run.json: policy, environment, dimension and the non-empty settings of the run"""
    settings = dict(spec.get("maia_settings") or {})
    settings.update(spec.get("env_options") or {})
    settings.update(properties=spec.get("properties"), file_role=file_role, actions=int(n), start=int(k0))
    settings["runs"] = list(runs)
    return PC.drop_empty(dict(policy=pid, environment=spec["environment"], dim=int(spec["dim"]), settings=settings))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="policy id of policies/MANIFEST.yaml")
    ap.add_argument("--actions", type=int, default=None, help="controlled actions (default: the case protocol)")
    ap.add_argument("--start", type=int, default=None, help="uncontrolled actions before control (default: case)")
    ap.add_argument(
        "--out", default="run.npz", help="name of the npz (relative to the run directory); prefix of the figures"
    )
    ap.add_argument("--runs", default="unc,ctl", help="roll-outs to run: unc, ctl or both (default)")
    ap.add_argument("--readable", action="store_true", help="also the readable figures and the figure of the actions")
    ap.add_argument("--store_data", action="store_true", help="also write the npz and policy_run.json")
    ap.add_argument("--no-plot", action="store_true", help="no figures (needs --store_data; plot the npz elsewhere)")
    ap.add_argument("--work-dir", default=".", help="run directory made by prepare_run.py (default: cwd)")
    ap.add_argument("--nproc", type=int, default=None, help="m-AIA ranks (optional launch check)")
    ap.add_argument("--file-role", default=None, help="another weights file of the entry (MANIFEST files 'role')")
    a = ap.parse_args()

    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if not runs or set(runs) - {"unc", "ctl"}:
        raise SystemExit(f"--runs {a.runs}: expected unc, ctl or unc,ctl")
    if a.no_plot and not a.store_data:
        raise SystemExit("--no-plot without --store_data would write nothing")
    work = os.path.abspath(a.work_dir)
    spec = json.load(open(os.path.join(work, "prepare_run.json")))
    if spec["policy"] != a.policy or spec.get("file_role") != a.file_role:
        raise SystemExit(
            f"{work} was prepared for {spec['policy']} ({spec.get('file_role')}), not {a.policy} "
            f"({a.file_role}): run prepare_run.py again"
        )
    c = PC.case_of(a.policy)[1]
    n = int(a.actions if a.actions is not None else c["n_actions"])
    k0 = int(a.start if a.start is not None else c.get("start", 0))
    dim = int(spec["dim"])
    pi = PC.load_for_run(a.policy, dim, a.file_role)
    ad = PC.make_observation(a.policy, pi, work, dim)
    probes = ad.probes()
    env = make_env(spec, work, probes, a.nproc)
    out_path = os.path.join(work, a.out)
    prefix, npz_path, json_path = PC.run_files(out_path)
    store = a.store_data
    if len(runs) == 1 and not store:  # one part of a split run: its data is needed to plot both parts together
        store = True
        print(
            f"--runs {runs[0]}: one part of a split run, storing {os.path.basename(npz_path)} and "
            f"{os.path.basename(json_path)} (plot both parts with plot_policy.py ctl/run.npz unc/run.npz)",
            flush=True,
        )
    plot = len(runs) == 2 and not a.no_plot
    if plot:
        import plot_policy  # matplotlib: fail before the roll-outs, not after them
    U = env.Ma / np.sqrt(3.0)
    base = dict(
        start=k0,
        n_actions=n,
        dt_action=float(env.num_substeps_per_iteration * U / (env.referenceLength / env.dX)),
        C_L0=float(getattr(env, "unperturbed_avg_lift", np.nan)),
    )
    meta = run_meta(a.policy, spec, n, base["dt_action"], env, a.file_role)
    if plot and c["quantity"] not in PC.SERIES:  # TKE figure: needs at least one trailing window of actions
        w = max(2, int(round(c["tke_window"] / base["dt_action"])))
        if n + 1 < w:
            print(
                f"note: {n} actions are shorter than the TKE window of the figure ({w} actions): no figures", flush=True
            )
            plot = False
    data = {}

    def log(msg):
        print(msg, flush=True)

    def save(tag, rec):
        data.update(to_arrays(tag, rec))
        if store:  # written during the roll-out, so that a long run that stops keeps what it has
            PC.write_run_files(out_path, dict(base, **data), meta, run_record(a.policy, spec, n, k0, runs, a.file_role))

    try:
        log(f"{a.policy}: {spec['environment']}, {n} actions after {k0} uncontrolled, dt {base['dt_action']:.4g}")
        for tag in runs:
            rollout(env, ad, tag == "ctl", k0 + n, k0, log, tag, save)
    finally:
        env.close()
    R = dict(base, **data)
    if store:
        log(f"wrote {npz_path} and {json_path}")
    if len(runs) == 2:
        if plot:
            written, s = plot_policy.plot_run(R, meta, prefix, readable=a.readable)
            for w in written:
                log(f"wrote {w}")
        else:
            s = PC.summary(R, c)
        log("summary: " + json.dumps(s))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # never leave m-AIA waiting in its MPMD loop
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        from mpi4py import MPI

        MPI.COMM_WORLD.Abort(1)
