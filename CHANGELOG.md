# Changelog

Notable user-facing changes to HydroGym. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); new entries go under
"Unreleased" until release.

## Unreleased

### Breaking — `terminated`/`truncated` semantics unified (MAIA, Nek)

The three external-process/in-process backends previously disagreed on
the gymnasium 5-tuple, which silently broke RL code (including
Stable-Baselines3's episode-boundary handling) ported between backends.
All gymnasium-API environments now follow the same semantics:

- `terminated = True` only when the **physics are invalid** (CFL blowup /
  divergence). For Nek this surfaces as a raised
  `hydrogym.nek.NekDivergenceError` from `step()` (the former bare
  `exit()`), not as a returned flag; the solver side is shut down cleanly
  (TERMN) before the raise. For MAIA there is currently no in-band
  divergence signal, so `terminated` is always `False` and a solver-side
  crash surfaces as an MPI communication error. For Firedrake (core.py)
  `terminated` was already always `False` and is unchanged.
- `truncated = True` when the **episode budget is reached**:
  `max_episode_steps` (MAIA, Firedrake), `nb_interactions` or the
  simulation end time `tmax` (Nek).

Migration: RL training code that previously checked
`terminated or truncated` keeps working unchanged. Code that relied on
Nek/MAIA reporting the step budget via `terminated` (e.g. logging or
bootstrap-on-termination logic) must now check `truncated` for the
budget and catch `NekDivergenceError` for Nek physics failure.

### Added

- `hydrogym.registration` — canonical gymnasium environment IDs
  (`hydrogym/Cylinder-v0`, `hydrogym-maia/Cylinder_2D_Re200-v0`,
  `hydrogym-nek/TCFmini_3D_Re180-v0`, `hydrogym-jaxfluids/Nozzle2D-v0`,
  ...); lazy per-backend entry points; JAX deliberately unregistered
  (functional/gymnax contract).
- `hydrogym.nek.NekDivergenceError` — catchable replacement for the
  former bare `exit()` on CFL blowup.
- `hydrogym.core_external` — shared `ExternalProcessEnvMixin` +
  `mpi_split`/`split_comm_by_appnum` helpers for external-process
  (MPMD) solver backends.
- `hydrogym.hf_env_mixin.HFEnvConfigMixin` — shared HuggingFace-Hub
  environment-data staging, used by the maia/jax/jaxfluids/nek backends.
- Developer guides: `docs/docs/developers/adding-a-solver.md`,
  `docs/docs/developers/adding-an-environment.md`; reference solver
  skeletons under `examples/developer_templates/` with a plain-CI job.
- `MaiaFlowEnv`'s `env_config` accepts an optional `nproc` key: the
  expected number of m-AIA solver ranks in the MPMD launch, validated at
  construction with a clear error naming the fix on a mismatch (mirrors
  the equivalent, pre-existing check for Nek). Omit to skip validation,
  unchanged from prior behavior.

### Fixed

- `FlowConfig`'s checkpoint auto-resolution (`hydrogym/firedrake/flow.py`)
  now picks deterministically from a directory containing multiple
  candidate checkpoint files (sorted by filename, last wins), instead of
  an unsorted directory listing whose result could silently vary by
  filesystem/download order. Also fixes 7 call sites that referenced the
  nonexistent `firedrake.logging.WARN` (Firedrake's logging shim only
  defines `WARNING`) — any checkpoint-resolution error that should have
  logged a warning and fallen back gracefully instead crashed with an
  unrelated `AttributeError`.
- `gym.make(...)` was broken for 2 of the 3 non-Firedrake gymnasium-API
  backends, and for Firedrake's own zero-argument case:
  - `SemiImplicitBDF` required `dt` positionally, contradicting its own
    parent classes' documented "defaults to `flow.DEFAULT_DT`" contract
    — broke every registered Firedrake ID's default construction.
  - MAIA's `gym.make()` path imported `hydrogym.maia.env_core` directly,
    bypassing the lazy loader that registers environment classes —
    `from_hf()` always saw an empty registry. Fixed, plus MAIA now raises
    a clear `ConfigError` (not a crash) when `probe_locations` is missing,
    and `hydrogym-maia/Cylinder_2D_Re200-v0` ships a verified default
    probe grid so it works with zero extra arguments.
  - `hydrogym.jaxfluids.envs` never re-exported its environment classes
    (unlike every other backend's `envs` package), and `registration.py`
    never defaulted `environment_name` — both fixed.

### Removed

- `hydrogym.distributed` — an empty placeholder package with no
  functionality; multi-agent RL support lives under `hydrogym.nek`
  (`NekParallelEnv`, `NekPettingZooEnv`). Re-add as a real package when
  distributed-training support actually lands.
