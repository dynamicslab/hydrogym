# Task 4.1 — Example scripts audit report

**Audit item 4.1**: "Verify every example script in `examples/` actually runs
as documented; fix or flag each one."

**Method**: every file under `examples/` (39 files across the four backends'
example trees, plus the JAX-Fluids example) was read and checked against the
current source API on branch `hydrogym-audit-v2`: imports resolved, config
keys matched against the env/config classes, CLI flags matched against the
scripts' own `argparse`/config schemas, and shell scripts checked for
correctness (`set -e` interactions, paths, container names).

**Verdict legend**: OK · INCONSISTENT (runs but misleads) · STALE
(documents removed behavior) · BROKEN (fails as written).

---

## Firedrake backend — all OK

All 10 files (`examples/firedrake/getting_started/*.py`, `*.ipynb`,
`docs/notebooks` index) use APIs that exist and match current signatures
(`gym.make("HydroGym-*")` IDs registered by Task 6.1, `hf_env` kwargs via
`HFEnvConfigMixin`). No fixes needed.

## JAX backend — 4 OK / 1 STALE / 1 BROKEN / 4 INCONSISTENT

- **OK**: `run_env.py`, `run_ppo.py`, `configs/jax_pc.yaml`, `README.md` main body.
- **BROKEN — `docs/notebooks/kolmogorov.ipynb`**: the control path is dead.
  The notebook sets `flow.control_function`, but nothing consumes it — the
  real actuation path is `control_field` threaded through `solve(...)`
  (`hydrogym/jax/solvers/base.py`). Following the notebook as written
  produces an unforced simulation, silently.
- **INCONSISTENT**:
  - `run_env_docker.sh`, `run_ppo_docker.sh`, `run_detached_docker.sh` are
    cluster job scripts (SLURM-style, no Docker anywhere) despite the name.
  - `README.md` claims wall-shear-stress ≈ 0.0019; the recorded notebook run
    shows ~0.34 at steady state. Magnitude is configuration-dependent, but
    the README value does not correspond to the shipped default config.
  - `run_ppo_docker.sh` has the same `set -e` + EXIT_CODE trap dead-code
    pattern as the MAIA script (see MAIA section): with `set -e`, the script
    exits at the first failing command and the exit-code reporting block
    (which the trap/branch logic around it suggests is meant to run) never
    executes.

## MAIA backend — 3 OK / 2 INCONSISTENT / 2 STALE

- **OK**: `test_maia_env.py`, `prepare_workspace.py`, `train_sb3_maia.py`
  (API calls verified against `hydrogym/maia/env_core.py`: `from_hf`,
  `list_available_environments`, `max_episode_steps`; and
  `hydrogym/maia/workspace.py` kwargs `local_fallback_dir`,
  `use_clean_cache`, `hf_token`, `hf_revision`).
- **INCONSISTENT — `run_example_docker.sh`**: despite "docker" in the name
  it loads HPC environment modules and submits to a cluster; and with
  `set -e` active, the `EXIT_CODE=$?` reporting block (lines ~83–91) can
  never run after a failure — the script dies at the failing command first.
- **STALE — `examples/maia/README.md` and `getting_started/README.md`**:
  both instruct creating **`properties.toml`**; the code and workspace
  loader actually require **`properties_run.toml`** (confirmed:
  `hydrogym/data_manager.py:85`, MAIA_LB `workspace_files` maps
  `"properties_run.toml"` → itself). A user following the README gets a
  missing-config error.
  - Additionally `examples/maia/README.md` line ~173 has an unterminated
    string literal in the embedded Python snippet (`"maia-lb-cylinder2d`
    with no closing quote) — the snippet won't parse if copy-pasted.

## Nek5000 backend — 19 OK / 3 BROKEN / 12 STALE / 5 INCONSISTENT

- **BROKEN**:
  1. `examples/nek/3_pettingzoo/train_sb3_pettingzoo.py` — sets
     `ctrl_min_amp`/`ctrl_max_amp` config keys that `NekEnv` validates
     against an allow-list (`hydrogym/nek/env.py:496-500`), raising
     `ConfigError` on every run.
  2. `examples/nek/5_hydrogym_control/test_nek_env_controller.py:117` —
     passes `nb_interactions=num_steps` where `num_steps` defaults to
     `None` → `ConfigError`/`TypeError` unless the user overrides it.
  3. `examples/nek/5_hydrogym_control/train_sb3_with_integrate.py:243` —
     calls `integrate(..., num_steps=...)`; `hydrogym/nek/integrate.py`'s
     signature has **no `num_steps` parameter** (it takes
     `t_span, dt, callbacks, controller, max_steps`) → `TypeError` on run.
- **STALE**: the two backend READMEs above document phantom APIs
  (`NekDataManager`, `from hydrogym import integrate`, a `'nek_path'`
  obs-info key, and the nonexistent `num_steps=` kwarg); the ten
  `ci_test_run/` output directories checked into the examples tree are run
  artifacts, not examples.
- **INCONSISTENT**: `run_nekenv_docker.sh`'s `LOCAL_DIR` points at
  `/workspace/hydrogym_NEK/packaged_envs` while sibling scripts use
  `/workspace/hydrogym/packaged_envs`; the 3_pettingzoo README's config
  relative path `../configs/` should be `../../configs/`;
  `tcflarge.yml` carries a `TCFmini` header comment.

## JAX-Fluids backend — 1 INCONSISTENT / 1 OK

- **INCONSISTENT — `test_jaxfluids_env.py:23`** (and the mirrored comment in
  `environment_config.yaml`): docstring says "Custom path to **MAIA**
  config.yaml" — copy-paste from the MAIA backend; should read JAX-Fluids.
- **OK**: `environment_config.yaml` (keys match `hydrogym/jaxfluids` config
  consumption).

## Repo hygiene found during this audit (out-of-scope fix, tracked separately)

- `.gitignore` only excludes
  `examples/firedrake/getting_started/output/`; the following run-output
  directories are currently **untracked but unignored**, and will keep
  leaking into `git status` / risk accidental commits:
  `examples/jaxfluids/outputs/`,
  `examples/maia/getting_started/ci_test_run_cpu/`,
  `examples/maia/getting_started/ci_test_run_gpu/`,
  `examples/nek/getting_started/1_nekenv_single/ci_test_run/`,
  `examples/nek/getting_started/1_nekenv_single/ci_test_run_task62_base/`.

## Source bug discovered by this audit (new tracked task, not an inline fix)

`hydrogym/nek/integrate.py` (verified against current tree):

1. Line ~131: `obs, reward, done, info = result[:4]` mis-unpacks the
   Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)` — `info`
   receives the `truncated` flag and the real info dict is dropped.
2. Line ~43: `obs = env.reset()` keeps the `(obs, info)` tuple as `obs`
   instead of indexing `[0]`.
3. The signature lacks the `num_steps` kwarg that
   `examples/nek/5_hydrogym_control/train_sb3_with_integrate.py` passes.

Per Safety Protocol rule 5 this becomes its own task (one commit, dev-container
Nek MPMD gate) rather than an edit bundled into this docs task.

---

## Summary counts

| Backend | OK | BROKEN | STALE | INCONSISTENT |
|---|---|---|---|---|
| Firedrake | 10 | 0 | 0 | 0 |
| JAX | 4 | 1 | 1 | 4 |
| MAIA | 3 | 0 | 2 | 2 |
| Nek5000 | 19 | 3 | 12 | 5 |
| JAX-Fluids | 1 | 0 | 0 | 1 |
| **Total** | **37** | **4** | **15** | **12** |

Every BROKEN/STALE/INCONSISTENT item above maps to a fix group in the Task
4.5 follow-up (per-backend, one commit per group, path-limited).
