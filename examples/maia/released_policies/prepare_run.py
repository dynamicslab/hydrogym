#!/usr/bin/env python3
"""Prepare the run directory of a released policy (no MPI, run before the MPMD launch).

    python prepare_run.py --policy cylinder_2d_re200/ppo/seed1 --work-dir runs/cyl200
    python prepare_run.py --policy naca0012_gust_re1000/td3_3jet_jet1.5U/seed1 --work-dir runs/gust3d
    python prepare_run.py --policy <id> --work-dir <dir> --env-dir /path/to/<Environment>   (offline copy)

The run directory is the m-AIA working directory and, at the same time, the environment data folder that
``run_policy.py`` points the HydroGym environment to:

- ``environment_config.yaml``: the environment's file with the settings the policy runs with (e.g. ``max_control``
  0.3125 for the 2D pinball Re 100 policies, the jet bound of the gust controller file);
- ``properties_run.toml`` (with the case's property changes, if any) and ``out_lb/restart_.Netcdf``: copies, so
  that the cached environment data is never modified;
- everything else (grid, geometry, STL, probe files): symbolic links to the environment data;
- ``prepare_run.json``: policy id, environment and settings of the run directory, read by ``run_policy.py``.

The environment data comes from the Hugging Face Hub (``dynamicslab/HydroGym-environments``, cached on first use)
or from a local copy of the environment folder (``--env-dir``; HF layout, or with grid and restart in ``out_lb/``).
"""

import argparse
import json
import os
import re
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_cases as PC  # noqa: E402

COPY_LIMIT = 2 << 30  # restarts larger than this are linked, not copied (3D cases)


def fetch_environment(name, revision=None):
    from hydrogym.data_manager import HFDataManager  # no MPI import

    dm = HFDataManager(repo_id="dynamicslab/HydroGym-environments", use_clean_cache=False, revision=revision)
    return dm.get_environment_path(name)


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def set_properties(path, values):
    """replace 'key = value' lines of an m-AIA property file"""
    txt = open(path).read()
    for key, val in values.items():
        pat = re.compile(r"^(\s*" + re.escape(key) + r"\s*=)[^#\n]*?([ \t]*(#.*)?)$", re.M)
        if not pat.search(txt):
            raise KeyError(f"{path}: no property {key} (environment data of an older revision?)")
        txt = pat.sub(lambda m: f"{m.group(1)} {val}{m.group(2)}", txt, count=1)
    with open(path, "w") as fh:
        fh.write(txt)


def prepare(pid, work_dir, dim=2, env_dir=None, revision=None, copy_restart=None, file_role=None):
    name = PC.env_name(pid, dim)
    dim = PC.env_dim(name)
    pi = PC.load_for_run(pid, dim, file_role)
    src = os.path.abspath(env_dir) if env_dir else fetch_environment(name, revision)
    work = os.path.abspath(work_dir)
    os.makedirs(os.path.join(work, "out_lb"), exist_ok=True)
    case, c = PC.case_of(pid)
    for f in sorted(os.listdir(src)):
        s, d = os.path.join(src, f), os.path.join(work, f)
        if f in ("out_lb", "environment_config.yaml", "restart_.Netcdf"):
            continue
        if f == "properties_run.toml":
            shutil.copyfile(s, d)
        elif f == "grid.Netcdf":
            link(s, d)
            link(s, os.path.join(work, "out_lb", f))
        else:
            link(s, d)
    for f in ("grid.Netcdf", "restart_.Netcdf"):  # environment folders with the solver files in out_lb/
        for cand in (os.path.join(src, f), os.path.join(src, "out_lb", f)):
            if os.path.exists(cand):
                d = os.path.join(work, "out_lb", f)
                big = os.path.getsize(cand) > COPY_LIMIT
                if f == "restart_.Netcdf" and (copy_restart or (copy_restart is None and not big)):
                    if os.path.lexists(d):
                        os.remove(d)
                    shutil.copyfile(cand, d)
                else:
                    link(cand, d)
                if f == "grid.Netcdf" and not os.path.exists(os.path.join(work, f)):
                    link(cand, os.path.join(work, f))
                break
        else:
            raise FileNotFoundError(f"{f} not found in {src} (nor in its out_lb/)")
    set_properties(os.path.join(work, "properties_run.toml"), c.get("properties") or {})
    cfg = yaml.safe_load(open(os.path.join(src, "environment_config.yaml")))
    over = PC.maia_config(pid, pi, dim)
    cfg["maia"].update(over)
    with open(os.path.join(work, "environment_config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    spec = PC.drop_empty(
        dict(
            policy=pid,
            environment=name,
            dim=dim,
            maia_settings=over,
            file_role=file_role,
            properties=c.get("properties"),
            env_options=c.get("env_options"),
        )
    )
    with open(os.path.join(work, "prepare_run.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    return work, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="policy id of policies/MANIFEST.yaml")
    ap.add_argument("--work-dir", required=True, help="run directory to create")
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3), help="2D or 3D target case (gust policies)")
    ap.add_argument("--env-dir", default=None, help="local copy of the environment folder (instead of the HF Hub)")
    ap.add_argument("--hf-revision", default=None, help="HF dataset revision to use")
    ap.add_argument("--copy-restart", action="store_true", default=None, help="copy also restarts larger than 2 GB")
    ap.add_argument("--file-role", default=None, help="another policy file of the entry (as run_policy.py)")
    a = ap.parse_args()
    work, spec = prepare(a.policy, a.work_dir, a.dim, a.env_dir, a.hf_revision, a.copy_restart, a.file_role)
    print(f"run directory: {work}")
    print(f"environment:   {spec['environment']}  settings {spec.get('maia_settings', {})}")
    if spec.get("env_options"):
        print(f"options:       {spec['env_options']}")
    here = os.path.dirname(os.path.abspath(__file__))
    print("launch (1 Python rank + N m-AIA ranks), from the run directory:")
    print(f"  cd {work}")
    # GPU image: one m-AIA rank per GPU (2D: 1, 3D: 4 GPUs); with the CPU build use one rank per core
    n = 1 if a.dim == 2 and "_3d_" not in a.policy else 4
    print(f"  mpirun -np 1 python {here}/run_policy.py --policy {a.policy} --out run.npz : -np {n} maia properties_run.toml --silent")
    print("  (with fewer cores than ranks: mpirun --bind-to none --oversubscribe ...; run_example.sh does all of this)")


if __name__ == "__main__":
    main()
