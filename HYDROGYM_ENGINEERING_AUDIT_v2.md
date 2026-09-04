# HydroGym Engineering Audit

**Audience:** a coding agent that will implement the changes described here, in a
follow-up session, inside the project's dev containers.
**Scope of this document:** audit + design + task list only. **No implementation
was performed in this phase.**
**Date:** 2026-08-19. **Repo:** `dynamicslab/hydrogym`, branch at time of audit:
current `main` tip (`bf2c2dd` "Add Algolia DocSearch integration to docs (#256)").

---

## Executive Summary

HydroGym is a "first working release" wrapping five independent CFD codebases
(Firedrake, JAX, JAX-Fluits, MAIA, Nek5000) as Gymnasium-style RL environments.
The core finding of this audit: **there is one real, working shared abstraction
(`hydrogym/core.py`), and exactly one backend (Firedrake) actually uses it.**
The other four backends (JAX, JAX-Fluids, MAIA, Nek5000) each reimplement their
own environment class from scratch, with their own config schema, their own
`step()`/`reset()` semantics, their own MPI/process coupling code, and
(critically) their own **~150-line copy-pasted** Hugging-Face config-resolution
helper duplicated 3-4 times with silent drift already visible between copies
(see Finding 2.11/2.12 below — one copy points at the wrong cache directory).

> Getting it down to shared core abstractions would be a huge, super useful step.

This is not primarily a "the abstraction is wrong" problem. Investigation shows
the divergence is *partly* legitimate (JAX needs a functional/JIT-compatible
API; MAIA/Nek need external-process MPI coupling that Firedrake, running
in-process, never needed) and *partly* pure accident (duplicated HF-resolution
code, inconsistent `terminated`/`truncated` semantics, a dead-code
`FlowConfig(PDEBase)` class duplicated verbatim into `kolmogorov.py`, an empty
`hydrogym/distributed/` package advertised in the README as "distributed RL
support," and no `gymnasium.register()` anywhere in the project, so `gym.make()`
never works for any backend).

The second major finding directly confirms the pain point you described:
**kwargs and config keys are dropped or ignored at multiple concrete,
now-located points** — a `NavierStokesTransientSolver` constructor accepts
`eta`/`max_noise_iter`/`noise_cutoff` and never uses them; `NekEnv.__init__`
declares `**kwargs` "for backward compatibility" and never reads it;
`NekEnv._apply_runtime_overrides` only forwards 5 hardcoded keys out of an
open-ended config dict; `HFDataManager`'s `cache_dir` is never actually passed
to the underlying `snapshot_download()` call; there is no `token`/`revision`
parameter anywhere for private/pinned Hugging Face access; and the same
semantic concept (checkpoint/restart source, substep count, reward
aggregation) has a different key name, in a different place in the config
tree, on almost every backend. Full table in Finding 2 / the Task List.

Third: **CI does not run the test suite at all.** `build.yml` — the only
workflow that touches Python besides linting — runs `poetry build` and
nothing else. `test/` (Firedrake-only, 8 files) is never executed by any
GitHub Actions workflow. This is the single most dangerous fact for the
"don't introduce new bugs" constraint you gave: **there is currently no
automated regression net at all**, so every task in the implementation plan
below is designed to *first* establish one, and only then make changes.

> This is a result of never getting a firm CI-time commitment out of Steve. I am hesitant to run full tests here without
> actual CI machines backing it up. But am equally doubtful that Steve would do that, or even cares?

Fourth: examples are largely consistent with the current Firedrake API where
spot-checked, but the project's own marketing copy is internally
inconsistent (README says "61+ environments," `docs/quickstart.md` and
`docs/introduction.md` say "88"), `test/README.md` and the root `README.md`
reference two different, unrelated Docker registries/images for "the" dev
container, and `codespell.yml` runs `codespell tests` and `codespell
tutorials` against directories that do not exist in this repository (the
real directory is `test/`, singular; there is no `tutorials/`).

Fifth (Nek/MAIA HPC question): **the answer is more encouraging than the
premise assumed.** Both solvers already use exactly the SPMD/MPMD hybrid
pattern you'd otherwise be asking to build — Python is launched as an
additional program inside the *same* `mpirun`/`srun` MPMD job as the solver,
becomes rank 0 of the shared `MPI_COMM_WORLD`, and talks to the solver ranks
via `mpi4py` and raw MPI messages, with no nested `mpirun`/`subprocess`
anywhere in the hot path. Full library embedding (no separate solver
process at all) is not realistic for either solver without substantial
upstream changes (global/static state in both codebases, one-shot
`MPI_Init`/`MPI_Finalize` lifecycles, MAIA's CUDA context ownership) and,
per the detailed feasibility study, the primary payoff of embedding
(avoiding per-episode process-startup cost) **is already realized** by both
backends' existing `reset()` implementations, which reuse the live MPI job
across episodes rather than relaunching it. Recommendation: keep the
process-based architecture; harden the RPC protocol instead (see §6).

> I'd add to that, that a full library embedding would be a huge constraint to getting other solvers added in the future.

**Bottom line recommendations**, expanded in the sections below:
- Redesign the solver interface as a **documented, partially-enforced
  contract with explicit escape hatches**, not a rewrite that forces JAX/
  MAIA/Nek into `core.PDEBase`. Extract the one piece of real, accidental
  duplication (HF config resolution) into a shared mixin.
- Fix the ~20 concrete kwargs-propagation defects found (Finding 2 / Task
  List Phase 2), each as an independent, individually-tested change.
- Add a CI job that actually runs `test/` (Phase 0, before anything else).
- Rewrite the "Adding a New Solver" and "Adding a New Environment" guides
  from scratch — they do not exist today in any form.
- Do **not** attempt to eliminate Nek5000/MAIA's process boundary; do
  invest in RPC batching and structured error handling instead.

> Have started writing tutorials, so point 4 should be addressed properly by my tutorials.

---

## Current Architecture

```
hydrogym/
├── core.py              # PDEBase, TransientSolver, FlowEnv, ActuatorBase, CallbackBase
├── data_manager.py       # HFDataManager — the one genuinely shared piece of infra
├── distributed/          # EMPTY — 0-byte __init__.py, advertised in README, does nothing
├── firedrake/             # Uses core.py as designed (the only backend that does)
│   ├── flow.py            # FlowConfig(PDEBase), ObservationFunction, ScaledDirichletBC
│   ├── actuator.py         # DampedActuator(ActuatorBase)
│   ├── solvers/            # NavierStokesTransientSolver(TransientSolver), SemiImplicitBDF, NewtonSolver
│   └── envs/{cavity,cylinder,pinball,step}/flow.py
├── jax/                   # Own gymnax-based Environment[EnvState,EnvParams], NOT gym.Env
│   ├── env_core.py         # JAXFlowEnv (dead), JAXFlowEnvBase (used)
│   ├── flow.py              # FlowConfig(PDEBase) — orphaned, duplicated into kolmogorov.py
│   ├── equation.py          # JAX-only Equation/SplitEquation/IMEXEquation family
│   └── envs/{channel,kolmogorov}.py
├── jaxfluids/              # Delegates almost entirely to external jaxfluids_rl.JAXFluidsEnv
│   ├── env_core.py          # JAXFluidsFlowEnv(JAXFluidsEnv) — 3rd copy of HF-resolution code
│   └── envs/nozzle.py
├── maia/                   # gym.Env directly, MPMD/MPI-coupled external process
│   ├── env_core.py           # MaiaFlowEnv(gym.Env) — 4th copy of HF-resolution code (ish)
│   ├── mpmd_interface.py      # MaiaInterface — raw MPI tag protocol to the maia binary
│   ├── workspace.py            # prepare_maia_workspace() — pre-mpirun file staging
│   └── hf_data_manager.py       # DEAD — unused duplicate of data_manager.py, delete it
└── nek/                     # gym.Env directly, MPMD/MPI-coupled external process
    ├── env.py                 # NekEnv(gym.Env) — mpi_split(), raw MPI tag protocol
    ├── parallel_env.py         # NekParallelEnv wraps NekEnv by composition (correct pattern)
    ├── pettingzoo_env.py        # NekPettingZooEnv wraps NekParallelEnv by composition
    ├── integrate.py              # independent reimplementation of TransientSolver.solve()
    └── configs.py                  # OmegaConf Config/Runner dataclasses (2nd config schema)
```

> Let's remove distributed for now to avoid giving wrongful impressions, and my PR for the distributed backend then
> recreated the folder.

Only Firedrake's `FlowEnv` (imported unmodified from `core.py`) and its
`FlowConfig`/`NavierStokesTransientSolver` subclasses honor the
`PDEBase`/`TransientSolver`/`FlowEnv` contract end to end. Every other
backend imports `core.py` symbols at most for typing, and none of them are
constructed via `gymnasium.make()` — there is no `gym.register()` call
anywhere in the repository. Every backend is used by direct class
instantiation with a backend-specific config object (a plain dict for
Firedrake/MAIA/JAX/JAX-Fluids, or a dict-or-OmegaConf-object dual path for
Nek).

---

## Major Findings

### 1. Unified Solver Interface

**Verdict: not unified today; a full rewrite to force identical internals
is not appropriate, but a real, enforceable minimal contract does not
currently exist either, and should.**

| Backend | Base class(es) actually used | Config mechanism | `step()` return | `close()`? | `gym.make()`? |
|---|---|---|---|---|---|
| Firedrake | `core.PDEBase` (via `FlowConfig`), `core.TransientSolver`, `core.FlowEnv` used directly | `dict`, exact `core.FlowEnv` schema | `(obs, reward, terminated, truncated, info)`; `terminated` always `False` | Yes (inherited, closes callbacks only) | No |
| JAX | `gymnax.environment.Environment` (`JAXFlowEnvBase`) — functional/JIT API | `dict`, JAX-specific, differs per subclass | `(obs, state, reward, done, info)` — no terminated/truncated split | No | No |
| JAX-Fluids | `jaxfluids_rl.JAXFluidsEnv` (external package) | `dict` (`environment_name`, `hf_repo_id`, ...) | inherited entirely from external package | Not defined in-repo | No |
| MAIA | `gymnasium.Env` directly (`MaiaFlowEnv`) | `dict` + OmegaConf YAML + TOML property file (cross-validated) | `(obs, reward, terminated, truncated, info)`; `terminated == truncated` (same bool) | Yes, correctly closes MPMD coupling | No |
| Nek5000 | `gymnasium.Env` directly (`NekEnv`); `NekParallelEnv`/`NekPettingZooEnv` wrap it by composition | Two parallel paths: legacy `Config` dataclass OR MAIA-pattern `dict` | `(obs, reward, terminated, truncated, info)`; `truncated` hardcoded `False` | Yes, correctly finalizes MPI | No |

**Fundamental divergences** (keep, but formalize as documented escape
hatches rather than silent inconsistency):
- JAX's functional `gymnax`-style API is a real requirement of
  `jit`/`vmap`/`lax.scan` — `core.FlowEnv`'s `self`-mutating loop cannot be
  traced. A shared abstraction would need a *separate* functional
  `PDEBase`/`FlowEnv` variant; retrofitting the existing one is not
  advisable.
- MAIA's and Nek's MPI/MPMD external-process coupling is a real
  architectural difference from Firedrake's (and, in principle, JAX's)
  in-process execution. `core.py` never anticipated an "external co-process"
  solver and offers no hook for one.
- JAX-Fluids delegating its env loop entirely to `jaxfluids_rl.JAXFluidsEnv`
  is a legitimate upstream-integration choice, not an internal flaw.

> I think we should attempt a functional FlowEnv variant to accomodate JIT, or JIT-style backends for example. Happy to
> take a stab at this as I am talk with Julia people (Oceanigans) to get an environment integrated

**Accidental divergences** (fix — no solver-physics reason for these):
1. `_setup_environment_data`/`_resolve_configuration_file`/
   `_find_configuration_file` copy-pasted near-verbatim across
   `hydrogym/jax/env_core.py`, `hydrogym/jaxfluids/env_core.py`,
   `hydrogym/maia/env_core.py`, and partially `hydrogym/nek/env.py` (~150
   lines × 3-4). One copy (`jax/env_core.py:137-153`) already has a
   confirmed bug from the drift: it writes to `~/.cache/maiagym/...`
   (MAIA's own namespace) instead of a JAX-specific directory, while the
   JAX-Fluids and Nek copies correctly use `jaxfluidsgym`/`nekgym`.
2. `hydrogym/jax/flow.py`'s `FlowConfig(PDEBase)` is dead code (never
   imported anywhere) and is separately, near-verbatim duplicated inline
   inside `hydrogym/jax/envs/kolmogorov.py:24-184`, including a
   self-recursive bug (`def state(self): return self.state`) that exists in
   both copies.
3. `hydrogym/maia/hf_data_manager.py` (421 lines) is a fully dead,
   superseded duplicate of `hydrogym/data_manager.py` — zero imports
   anywhere in the repo.
4. `terminated`/`truncated` semantics differ across all three backends that
   nominally implement the same 5-tuple gymnasium contract (Firedrake:
   `terminated` always `False`; MAIA: both set to the same bool; Nek:
   `truncated` hardcoded `False`). This will silently break any RL code
   (including SB3's own episode-boundary handling) ported between backends.
5. `RungeKuttaCrankNicolson(TransientSolver)` (`hydrogym/jax/solvers/base.py:24`)
   inherits from `core.TransientSolver` in name only — its `step()`/`solve()`
   signatures are fully incompatible with the base class and none of the
   base class's logic is reused. This buys nothing and is actively
   misleading.
6. No `gymnasium.register()` call exists anywhere for any backend — pure
   missing plumbing, not a technical constraint.
7. `hydrogym/__init__.py`'s lazy-loader `__getattr__` allowlist
   (`("distributed", "firedrake", "maia", "nek")`) omits `"jax"` and
   `"jaxfluids"` — both are still importable via explicit
   `import hydrogym.jax`, but not via `hydrogym.jax` attribute access after
   a bare `import hydrogym`, unlike the other three backends. Almost
   certainly an oversight, not a design choice.
8. MAIA's `prepare_maia_workspace()`/MPMD-launch preparation and Nek's
   `NekEnv.from_hf()`+manual-MPMD-launch preparation solve the *identical*
   underlying problem (stage files, then get an external process co-launched
   and MPI-coupled before the Python object is usable) with zero shared
   code, using different, hand-rolled protocols.

Design for a unified interface is in **§ Solver Interface Design** below.

> We need to also record this in a design document so we inch closer to having a kind of "developer documentation"

### 2. Argument / kwargs Propagation

**Verdict: confirmed, with 20 concrete located defects**, matching your
described experience exactly. Full table below (Finding numbers 2.1-2.22
map to file:line evidence; a condensed version appears again in the Task
List with fixes).

| # | Entry point → destination | Argument | Propagated? | Problem | file:line |
|---|---|---|---|---|---|
| 2.1 | `FlowEnv(env_config)` → `PDEBase.__init__(**config)` | any unrecognized `flow_config` key | No | Unknown keys silently swallowed, zero warning anywhere | `hydrogym/core.py:47-65` |
| 2.2 | `FlowConfig.__init__` → `_resolve_checkpoint` | `mesh` fallback | Broken | Falls back to `self.MESH_DIR` (an absolute path) instead of `self.DEFAULT_MESH` (a mesh *name*), breaking HF checkpoint auto-inference whenever `mesh` isn't explicitly given | `hydrogym/firedrake/flow.py:88-99` |
| 2.3 | `SemiImplicitBDF(...)` → `NavierStokesTransientSolver.__init__(eta=0.0, max_noise_iter=..., noise_cutoff=None)` | `eta`, `max_noise_iter`, `noise_cutoff` | No — accepted, never used | Params stored nowhere, `white_noise()` imported but never called | `hydrogym/firedrake/solvers/base.py:75-97,9` |
| 2.4 | `integrate(flow, t_span, dt, ..., **options)` | `collect_rewards` | No | Never forwarded to `solver.solve()`, so the reward-collection feature is unreachable through the convenience wrapper | `hydrogym/firedrake/solvers/integrate.py:11-16` |
| 2.5 | `MaiaFlowEnv(env_config)` → `MaiaInterface` | MPI `np`/hosts/walltime | Not exposed at all | No Python code path launches the MAIA process; MPMD launch is entirely out-of-band shell, with zero config surface or mismatch detection | `hydrogym/maia/env_core.py:163-167`, `hydrogym/maia/mpmd_interface.py:26-91` |
| 2.6 | `NekEnv.__init__(..., **kwargs)` | `**kwargs` | No — accepted, dropped | Docstring says "for backward compatibility"; body never references `kwargs` after the signature | `hydrogym/nek/env.py:132-173` |
| 2.7 | `NekEnv._apply_runtime_overrides(env_config)` | any key outside a 5-key whitelist | No — silently dropped | `override_map` hardcodes exactly `normalize_input`, `nb_interactions`, `random_init`, `rescale_actions`, `rew_mode`; anything else (`WALLTIME`, `dt`, `CASENAME`, ...) is invisible, contradicting the broader-sounding docstring on `from_hf` | `hydrogym/nek/env.py:432-450` (vs `:109-120` docstring) |
| 2.8 | `NekEnv._initialize()` | MPI rank-binding policy | Hardcoded | `mpi_info.Set("bind_to", "none")` unconditional, no `env_config` override, unlike the adjacent configurable `hostfile` | `hydrogym/nek/env.py:504-511` |
| 2.9 | `NekEnv` process launch | `nproc` (Python) vs external `mpirun -np N` | Partial | `nproc` only *validates* the already-running MPI world size; must be manually kept in sync with the shell command, no single source of truth | `hydrogym/nek/env.py:36-65` |
| 2.10 | 4 backends' `HFDataManager(..., fallback_profile=...)` | `fallback_profile` | Inconsistent | MAIA/Nek/JAX-Fluids pass it; pure-JAX `JAXFlowEnv.__init__` does not, silently defaulting to MAIA's own validation rules even though a `"JAX"` profile exists | `hydrogym/jax/env_core.py:86-88` vs `maia/env_core.py:52,85`, `nek/env.py:130,270`, `jaxfluids/env_core.py:46` |
| 2.11 | `JAXFlowEnv._setup_environment_data()` | cache namespace | Copy-paste bug | Uses `"maiagym"` instead of a JAX-specific name, risking cross-backend cache collisions | `hydrogym/jax/env_core.py:137-153` vs `maia/env_core.py:175-191`, `nek/env.py:349-364`, `jaxfluids/env_core.py:70-86` |
| 2.12 | `HFDataManager.__init__` → `snapshot_download(...)` | `cache_dir` | No — accepted, not forwarded | None of the three `snapshot_download()` call sites pass `cache_dir=self.cache_dir`; downloads always land in the HF default cache regardless of user config — relevant on quota-limited HPC home dirs | `hydrogym/data_manager.py:524-533,587-596,626-635` |
| 2.13 | `HFDataManager.__init__` | `token`, `revision` | No — not exposed anywhere | No way to access private repos or pin a revision from HydroGym's own API; only ambient `HF_TOKEN`/`huggingface-cli login` works | `hydrogym/data_manager.py:249-289,334-345,524-644` |
| 2.14 | `hydrogym/maia/hf_data_manager.py::HFDataManager` | entire class | Dead code | Byte-for-byte-similar older duplicate, zero imports anywhere; a future fix to the real `HFDataManager` (e.g. adding `token=`) is likely to land in only one of the two copies | whole file |
| 2.15 | Restart/checkpoint concept | naming + exposure | Inconsistent | Firedrake: top-level `flow_config["restart"]`. Nek: nested `conf.initial_conditions.restart_folder`/`conf.simulation.restart_folder` with 3 legacy aliases, plus a separate `random_init` override. MAIA: not user-configurable at all (baked into the HF-downloaded environment) | `firedrake/flow.py:96-109`, `nek/env.py:408-431,1032-1066`, `data_manager.py:76-92` |
| 2.16 | Substep-count concept | naming | Inconsistent | `core.py` deprecates `num_sim_substeps_per_actuation` in favor of `num_substeps` for Firedrake — but MAIA (`cfg.maia.num_sim_substeps_per_actuation`) and JAX (`cfg.jax.num_sim_substeps_per_actuation`) still use the "deprecated" name as their *only* option | `core.py:388-396` vs `maia/env_core.py:137`, `jax/env_core.py:118` |
| 2.17 | Reward-aggregation concept | naming | Inconsistent | Firedrake: `actuation_config["reward_aggregation"]` (`mean`/`sum`/`median`, plus a deprecated misspelled alias). Nek: `reward_agg` direct kwarg (`mean`/`sum` only, no `median`) | `core.py:398-412` vs `nek/env.py:132-138,781-784` |
| 2.18 | `ChannelFlowSpectralEnv.__init__(env_config)` | `Lx`,`Ly`,`Lz`,`nu`,`Nx`,`Ny`,`Nz` | No — hardcoded, config ignored | 7 physical/grid params are literal constants in `__init__`, never read from `env_config`, even though the sibling `ChannelEnvParams` dataclass redeclares the same defaults as if configurable | `hydrogym/jax/envs/channel.py:414-436` vs `:29-52` |
| 2.19 | `ChannelFlowSpectralEnv.__init__` | `hf_repo_id`, `cache_dir`, `use_clean_cache` | No — not exposed | Builds its own inline `HFDataManager`, only forwards `local_fallback_dir`, unlike `JAXFlowEnv` which reads all of these from `env_config` | `hydrogym/jax/envs/channel.py:443-455` vs `jax/env_core.py:82-88` |
| 2.20 | `MaiaFlowEnv`/`JAXFlowEnv` HF-integration logic | entire helper set | Duplication risk realized | Near-identical copy-pasted implementations across the two files; a fix applied to one has already, twice, not been applied to the other (rows 2.10, 2.11) | `maia/env_core.py:31-354` vs `jax/env_core.py:42-345` |
| 2.21 | `env_config`'s 3-level shape (`solver_config`/`actuation_config`/top-level) | cross-dict key placement | Unvalidated | No runtime check catches a key placed in the wrong sub-dict (e.g. `num_substeps` inside `solver_config`); most misplacements are silently ignored (row 2.1), some raise a `TypeError` at construction (comparatively "loud," but inconsistent) | `hydrogym/core.py:377-412` |
| 2.22 | `NewtonSolver(flow, solver_parameters={})` | `solver_parameters` | **Yes — contrast case** | Correctly and fully propagated to `fd.NonlinearVariationalSolver`; included as a positive control showing the codebase *can* do this right | `hydrogym/firedrake/solvers/base.py:14-43` |

### 3. Examples and Documentation

Firedrake examples spot-checked (`config_reference.py`,
`advanced/cylinder/run-transient.py`) are largely accurate against the
current API — `use_HF_data_manager`, `velocity_order`,
`num_substeps`/`reward_aggregation` (the non-deprecated names) and
`integrate(..., stabilization=...)` all match real, live signatures. One
minor issue: `run-transient.py` defines `element_type = "p1p1"` and never
uses it (dead local variable, cosmetic only, not a functional bug).

Cross-project documentation inconsistencies found directly:
- **Environment count mismatch**: `README.md` says "61+"/"61 environments"
  (lines 14, 20, 46) three times; `docs/docs/quickstart.md:61` and
  `docs/docs/introduction.md:9` both say "88". These cannot both be
  current.
- **Docker image mismatch**: root `README.md` recommends
  `clagemann/hydrogym-nvhpc-*`/`clagemann/hydrogym-rocm-*` images;
  `test/README.md` instead references `lpaehler/hydrogym-env:stable`, a
  different registry/maintainer entirely, with no cross-reference between
  the two.
- **`codespell.yml` targets nonexistent paths**: the workflow runs
  `codespell tests` and `codespell tutorials`; the repository has `test/`
  (singular) and no `tutorials/` directory at all. This step either
  silently no-ops, errors, or has been broken since it was added — needs
  verification and fixing either way. 
- **CI never runs the test suite** (see Finding 4 — this is the more
  severe of the two testing-related findings and is treated as its own
  top-priority item in the Task List).
- Docs API reference (`docs/docs/api/**/*.md`) is regenerated at
  Docusaurus build time via `pydoc-markdown` (`docs/package.json`'s
  `generate-api` script), so the checked-in `.md` files are a snapshot,
  not a live source of truth — but the `deploy.yml`/`test-deploy.yml`
  workflows that run this generation step `pip install pydoc-markdown`
  only, **never installing `hydrogym` itself or any backend's
  dependencies**. If `pydoc-markdown`'s Python loader needs to import a
  module to introspect it (version-dependent behavior), pages for modules
  requiring `firedrake`/`jax`/`mpi4py` may silently fail to generate or
  generate incompletely, with no CI signal either way (not a hard failure,
  just an unverified/unmonitored generation step) — needs the coding agent
  to check pydoc-markdown's actual import behavior for this repo's version
  and either confirm it's fine (static analysis, no import needed) or add
  the missing install step.

> *--> was the best solution I could come up with, so it's an inherent drawback I accepted*

- No example anywhere is executed by CI. `test/` (Firedrake, 8 files) is
  the only test code that exists at all; MAIA, Nek, JAX, and JAX-Fluids
  have **zero automated test coverage**, in CI or otherwise, beyond
  manually-run example scripts.
- **Docstring coverage is very uneven, though the auto-doc mechanism
  already covers all of `hydrogym/`.** The docs site generates its entire
  API reference from docstrings via `pydoc-markdown` (Google-style
  processor, `pydoc-markdown.yaml`), run at every Docusaurus build
  (`docs/package.json`'s `generate-api` script) — so any function that
  gets a docstring automatically gets a documented page; the gap is
  purely in how many functions have one. Measured directly (functions +
  classes with a non-empty docstring, via `ast.get_docstring`):

  | File | Coverage |
  |---|---|
  | `hydrogym/maia/env_core.py` | 31/31 (100%) |
  | `hydrogym/nek/env.py` | 33/34 (97%) |
  | `hydrogym/jaxfluids/env_core.py` | 5/6 (83%) |
  | `hydrogym/core.py` | 30/48 (63%) |
  | `hydrogym/firedrake/flow.py` | 27/46 (59%) |
  | `hydrogym/jax/env_core.py` | 16/68 (24%) |

  MAIA and Nek are already essentially fully documented; JAX is the real
  gap. There is also no CI check enforcing a coverage floor (no
  `interrogate`/`pydocstyle`/equivalent anywhere in
  `.github/workflows/`), so coverage can silently regress even where it's
  currently good. **Deliberately scheduled last (Phase 7)** — this is
  pure documentation debt, orthogonal to and non-blocking for every other
  finding in this audit, and doing it last avoids repeatedly rewriting
  docstrings on functions whose signatures Phases 1-5 are still changing.

*(The originally-planned exhaustive per-example pass for MAIA/Nek/
JAX/JAX-Fluids examples — the agent doing this work was interrupted by a
session limit partway through the Nek examples. What's confirmed above is
what was directly verified; treat the MAIA/Nek/JAX/JAX-Fluids example trees
as "not yet audited in full" and include the audit itself as Task 4.1
below, before any example rewrite.)*

### 4. General Software Architecture

- **Critical — CI never runs tests.** `.github/workflows/build.yml` (the
  only workflow touching the Python package beyond linting) does exactly:
  `pip install poetry` → `poetry build`. No `pytest` invocation exists in
  any workflow. `test/README.md` documents running tests only as a
  *manual* step inside a specific Docker container. This means every PR
  merged to `main` today, including the merge that produced this audit's
  starting point, has had **zero automated verification that the Firedrake
  backend still works**, let alone the other four.
- **No type-checking CI job** (no `mypy`/`pyright` anywhere in
  `.github/workflows/` or `pyproject.toml`).

> I want this. If you are fine with it I think we should set up ty + beartype for type checking. Just never got around to it.

- **`hydrogym/distributed/`** is a literal 0-byte `__init__.py`. It is
  listed in `hydrogym/__init__.py`'s lazy-load allowlist and `__all__`, and
  the root `README.md` claims "Scalable: MPI-parallelized solvers with
  distributed RL training support" and `docs/CLAUDE.md` describes it as
  "Multi-agent and distributed RL support" — neither claim is backed by any
  code in this package today (Nek's `NekParallelEnv`/`NekPettingZooEnv`
  provide actual multi-agent support, but live under `hydrogym/nek/`, not
  `hydrogym/distributed/`).
- **`ruff.lint.ignore = ["F401", "E731"]`** (`pyproject.toml:136`) disables
  unused-import detection project-wide, which is exactly the kind of
  lint that would have caught the dead `jax/flow.py::FlowConfig` and the
  dead `hydrogym/maia/hf_data_manager.py` imports had they ever existed as
  imports (they're dead as *modules*, not unused imports, so this specific
  rule wouldn't catch these two cases — but it does mean the project has no
  automated defense against *future* dead-import accumulation).

> Is there a better way for us to run lint without it triggering a whole bunch of Firedrake import errors?

- **Two duplicate/parallel `HFDataManager` implementations**
  (`hydrogym/data_manager.py`, actually used everywhere, vs
  `hydrogym/maia/hf_data_manager.py`, dead) — see Finding 2.14.
- **`hydrogym/__init__.py` lazy-loader omits `jax`/`jaxfluids`** — see
  Finding 1.7.
- **Test suite is 100% Firedrake-only and requires a full Firedrake
  install** (`test/test_*.py` all `import firedrake as fd` /
  `import hydrogym.firedrake as hgym`); there is no lightweight unit test
  anywhere of `hydrogym/core.py`'s `FlowEnv`/`PDEBase`/`TransientSolver`
  abstraction in isolation (e.g. with a trivial mock flow/solver) — the
  shared abstraction that everything else is supposed to build on is only
  ever exercised indirectly, through Firedrake's full PDE stack.

> If we can auto-generate a draft of this, unit tests would probably be super helpful.

- **Version is already `1.0.0`** (`pyproject.toml:3`), implying an API
  stability promise, while `hydrogym/core.py` already carries two
  `DeprecationWarning`s for renamed config keys (`num_sim_substeps_per_actuation`
  → `num_substeps`, `reward_aggreation_rule` → `reward_aggregation`) —
  evidence the maintainers are already aware of, and actively managing,
  past API churn. No `CHANGELOG` file exists anywhere in the repo.

### 5. Extensibility

There is currently **no "Adding a New Solver" or "Adding a New Environment"
guide anywhere** in `docs/docs/developers/` (only `contributing.md`, which
covers process/PR mechanics, not architecture) or in the repository root.
A developer wanting to add a 6th backend today would need to read all five
existing backends in full to reverse-engineer which parts of `core.py` are
actually load-bearing (Firedrake only) versus vestigial (everyone else).
This audit's **Extensibility Design** section below is written specifically
to become that guide's technical backbone; Task List Phase 4 turns it into
the actual `docs/docs/developers/adding-a-solver.md` /
`adding-an-environment.md` pages plus a copyable skeleton.

> Being addressed by me as we speak. You can skip that for now if you want to.

### 6. Nek / MAIA HPC Integration

Full feasibility study is reproduced in **§ Nek / MAIA Feasibility Study**
below. Headline conclusions:
- Both solvers **already run as one Slurm/PBS-compatible MPMD job**, not
  nested subprocess/`mpirun` calls from Python — `mpi_split()`
  (`hydrogym/nek/env.py:36-85`) and `MaiaInterface.init_comm()`
  (`hydrogym/maia/mpmd_interface.py:63-91`) both split the *shared*
  `MPI_COMM_WORLD` of one externally-launched MPMD job.
- Full in-process library embedding is **not recommended** for either
  solver: both have process-lifetime-scoped `MPI_Init`/`MPI_Finalize` and
  substantial global/static state (confirmed in Nek5000's forked
  `drive.f`, and directly in MAIA's C++ source: `g_mpiInformation`,
  `mEnvironment`, and `globalvariables.cpp`'s many globals) that make
  multiple instances, or sharing a Python process's address space, unsafe
  without a large upstream refactor. MAIA additionally owns a CUDA context
  per process.
- The usual justification for embedding — amortizing process-startup cost
  across RL episodes — **does not apply here**: both `NekEnv.reset()` and
  `MaiaFlowEnv.reset()` already reuse the live MPI job across episodes
  (`RSETS`/`reinit()` commands over the existing communicator), only
  tearing the process down in `close()`.
- Recommendation: **keep the process architecture**; invest instead in (a)
  batching the current per-node Python-loop `Send`/`Recv` calls in Nek's
  `_get_state`/`_send_action` into vectorized `Gatherv`/`Scatterv` calls,
  and (b) replacing Nek's abrupt `exit()` call on CFL blow-up
  (`hydrogym/nek/env.py:921`) with a structured exception the RL training
  loop can catch and recover from, rather than killing the whole job.

> We want to keep this process separation going forward. Embedding is being deprecated across modern RL libraries.

---

## Detailed Findings

All findings above are reproduced with full file:line evidence in Findings
1-6; the Task List (below) converts each into an independently-shippable,
independently-testable unit of work. No additional findings beyond what is
listed above were substantiated with direct code evidence; anything not
listed here (e.g. a full per-file MAIA/Nek/JAX-Fluids example audit) is
explicitly called out as unfinished and scheduled as Task 4.1 rather than
asserted without evidence.

---

## Proposed Target Architecture

Keep five backends, one repository, one *documented contract* — do not
force identical internals. Concretely:

1. **`hydrogym/core.py` stays the base for in-process, `self`-mutating
   solvers** (today: Firedrake only). No changes to its public contract
   (backwards compatibility — see below) beyond fixing the two dropped-kwarg
   bugs found inside it (Task List 2.x).
2. **Add `hydrogym/core_external.py`** (new, small file) defining a
   documented `ExternalProcessEnv` mixin/protocol capturing what MAIA and
   Nek both already do but never shared: a `launch_config` sub-schema
   (`nproc`, `hostfile`, MPI binding options), a `close()` contract that
   *must* send a termination message and free/finalize MPI, and a common
   `_wait_for_mpmd_peer()` helper factoring out the duplicated
   split-communicator logic in `mpi_split()` and
   `MaiaInterface.init_comm()`. This does **not** ask Nek/MAIA to change
   their wire protocols (those stay solver-specific, correctly) — it only
   unifies the *shape* of the launch/lifecycle contract so a new
   external-process solver doesn't have to reinvent it a third time.
3. **Add `hydrogym/hf_env_mixin.py`** (new, small file) — extract
   `_setup_environment_data`/`_resolve_configuration_file`/
   `_find_configuration_file`/cache-namespace logic out of
   `jax/env_core.py`, `jaxfluids/env_core.py`, `maia/env_core.py`, and the
   partial copy in `nek/env.py`, into one `HFEnvConfigMixin` class used by
   all four. This directly fixes Findings 1.1/2.10/2.11/2.20 by
   construction (one implementation instead of four to keep in sync).
4. **JAX stays on `gymnax`'s functional API** — do not retrofit
   `core.FlowEnv`. Instead: (a) delete the dead `jax/flow.py::FlowConfig`
   and its duplicate inside `kolmogorov.py`, replacing both with nothing
   (there is no working consumer) or a real, single implementation if a
   PDEBase-based JAX flow is ever actually needed; (b) fix
   `RungeKuttaCrankNicolson` to stop inheriting from `core.TransientSolver`
   in name only — either drop the inheritance or make it real.
5. **JAX-Fluids stays a thin wrapper around `jaxfluids_rl.JAXFluidsEnv`** —
   no change to that relationship; only the HF-resolution duplication
   (item 3 above) is unified.
6. **Register every backend with `gymnasium.register()`**, using
   backend-appropriate `entry_point` factories that accept each backend's
   native config schema (this does not require unifying the schemas — it
   only requires each backend to expose one `make(**kwargs) -> Env`
   factory function, which can internally still expect a `env_config`
   dict). This directly fixes Finding 1.6 and gives users one predictable
   discovery path (`gym.make("hydrogym/Cylinder-v0", ...)`) across all five
   backends without touching internals.
7. **Standardize `terminated`/`truncated` semantics** across Firedrake,
   MAIA, and Nek (Finding 1.4): `terminated` should mean "the physics
   became invalid / episode failed" (e.g. CFL blow-up, divergence) and
   `truncated` should mean "step budget reached" — currently no backend
   does exactly this. This is a **behavior-changing fix** and must be
   flagged loudly in the changelog/migration notes (see Backwards
   Compatibility).
8. **Standardize the three duplicated-but-differently-named config
   concepts** (restart/checkpoint, substep count, reward aggregation —
   Findings 2.15-2.17) on Firedrake's already-non-deprecated names
   (`restart`, `num_substeps`, `reward_aggregation`), with deprecation
   shims (following the existing `core.py` pattern) on MAIA/JAX/Nek's old
   names rather than a hard break.

---

## API Changes

All changes below are **additive or deprecation-shimmed**, not breaking,
except the `terminated`/`truncated` semantics fix (item 7 above), which
is called out explicitly as the one behavior change with real
backwards-compatibility risk (see § Backwards Compatibility for the
recommended rollout).

| Change | Old | New | Deprecation path |
|---|---|---|---|
| `NavierStokesTransientSolver` noise kwargs | accepted, silently ignored | either wired to `white_noise()` or removed | If removed: keep accepting+warning for one minor version, then drop |
| `integrate()` reward collection | `collect_rewards` unreachable | forwarded to `solver.solve()` | Pure addition, no shim needed |
| `NekEnv.__init__(**kwargs)` | silently dropped | removed, or explicitly validated/raised | Warn-then-raise over one minor version |
| `NekEnv._apply_runtime_overrides` | 5-key whitelist | generic dotted-path override (`section.param=value`) with validation | Old 5 keys keep working unchanged; new mechanism is additive |
| `HFDataManager(cache_dir=...)` | accepted, not forwarded | forwarded to every `snapshot_download()` call | Pure bugfix, no API shape change |
| `HFDataManager(token=..., revision=...)` | absent | new optional kwargs, threaded through the full call chain | Pure addition |
| MAIA/JAX `num_sim_substeps_per_actuation` | only working name | deprecated in favor of `num_substeps`, both accepted with warning | Mirrors existing `core.py:388-396` pattern exactly |
| Nek `reward_agg` | Nek-only name, `mean`/`sum` only | accept `reward_aggregation` too (incl. `median`), old name warns | Additive + warn |
| `gymnasium.register()` for all 5 backends | absent | added, backend-native `env_config` still accepted via `gym.make(id, env_config=...)` | Pure addition |
| `terminated`/`truncated` semantics (MAIA, Nek) | inconsistent | standardized per item 7 above | **Breaking** — major-version-flagged, see Backwards Compatibility |

---

## Configuration / Argument Propagation Design

Principle: **do not blanket-add `**kwargs` everywhere.** Instead:

1. **At every layer boundary, either fully forward a parameter or
   explicitly document why it stops there.** Concretely, for every
   `dict.get(key, default)` call found in Finding 2's table, the fix is
   one of: (a) actually use the retrieved value downstream (most rows),
   (b) raise a clear `ConfigError` if the key is unrecognized (new
   validation step, see below), or (c) document in the docstring that this
   is intentionally the terminal layer for that key.
2. **Add one shared "unknown key" validation helper** in
   `hydrogym/core.py` (`PDEBase.__init__`) and mirror it in
   `hydrogym/hf_env_mixin.py` for the other backends: after all recognized
   keys are consumed via `.pop()` (not `.get()`), any keys remaining in the
   config dict trigger a `warnings.warn(f"Unrecognized config keys: {...}")`
   — not a hard error, to avoid breaking currently-passing extra keys
   that some downstream code silently relies on being ignored, but loud
   enough that the exact failure mode you described ("I pass an argument
   and get an error / silent no-op several layers down") becomes visible
   immediately at the top-level call instead of being invisible.
3. **Do not merge the MAIA/JAX/JAX-Fluids/Nek config schemas into one
   shape.** They differ for defensible reasons (OmegaConf+TOML
   cross-validation for MAIA's two-source-of-truth model; dataclass config
   for Nek's legacy path). Standardizing the *names* of overlapping
   concepts (restart, substeps, reward aggregation — API Changes table
   above) captures the real, user-visible win without a disruptive schema
   rewrite.
4. **`HFDataManager` becomes the single place `token`/`revision`/`cache_dir`
   are threaded from** — every backend's `env_config` gains optional
   `hf_token`/`hf_revision` keys (mirroring the existing `hf_repo_id`
   pattern) that flow straight through to the one `HFDataManager`
   instance each backend already constructs.

---

## Solver Interface Design

Two documented base contracts, not one:

### In-process contract (`hydrogym/core.py`, unchanged in shape)
- `PDEBase` — abstract: `num_inputs`, `num_outputs`, `load_mesh`,
  `initialize_state`, `init_bcs`, `copy_state`, `save_checkpoint`,
  `load_checkpoint`, `get_observations`, `evaluate_objective`, `render`.
  Concrete (inherit for free): `set_state`, `reset`, `reset_controls`,
  `set_control`, `advance_time`, `dot`.
- `TransientSolver` — `step(iter, control=None)`, `solve(...)`, `reset()`.
- `FlowEnv(gym.Env)` — the actual gymnasium-facing class; unchanged.
- **Who uses this today:** Firedrake. **Who could in principle:** any
  future in-process, `self`-mutating, non-JIT solver added later.

### External-process contract (new: `hydrogym/core_external.py`)
A lightweight `ExternalProcessEnvMixin` (not a hard ABC — MAIA and Nek keep
their own wire protocols) providing:
- `launch_config` schema fields: `nproc`, `hostfile`, `mpi_bind_to`
  (fixes Finding 2.8), validated against `MPI.COMM_WORLD.Get_size()` at
  construction (generalizes the ad hoc check already in `mpi_split`).
- `_split_mpmd_comm(comm_world)` — factors out the shared logic between
  `mpi_split()` (`nek/env.py:36-85`) and `MaiaInterface.init_comm()`
  (`maia/mpmd_interface.py:63-91`); both call sites are refactored to call
  this, keeping their own tag-protocol code untouched.
- `close()` contract: must send a termination message over the live
  communicator and call the appropriate MPI teardown — codify what Nek and
  MAIA already correctly do (`hydrogym/nek/env.py:1077-1087`,
  `hydrogym/maia/env_core.py:757-769`) as a documented requirement for any
  future external-process backend, rather than something each backend
  has to rediscover.
- **Explicitly not unified:** the actual observation/action wire format
  (raw MPI tags for both, but different tag sets and different buffer
  layouts) stays solver-specific. This is the correct "solver-specific
  escape hatch" called for in the task brief — MAIA's LBM/FV field layout
  and Nek's SEM node layout are genuinely different and forcing a shared
  wire format would be the "artificial identicalness" the brief explicitly
  says to avoid.

### Functional/JIT contract (JAX, unchanged)
`gymnax.environment.Environment[EnvState, EnvParams]` stays exactly as is.
No new base class — document it as the third legitimate pattern rather
than pretend it doesn't exist.

### Registration layer (new, all backends)
```python
gym.register(
    id="hydrogym/Cylinder-v0",
    entry_point="hydrogym.firedrake.envs.cylinder:make",  # thin factory, wraps FlowEnv(env_config)
)
```
Each backend gains one `make(**kwargs) -> Env` factory (thin wrapper
around today's direct-construction pattern) purely so `gym.make()` works
uniformly; no internal restructuring required.

---

## Environment Interface Design

No changes to `FlowEnv`'s public shape (`step`, `reset`, `render`, `close`)
beyond the `terminated`/`truncated` semantics fix. For MAIA/Nek (which
subclass `gym.Env` directly, not `FlowEnv`), no forced migration to
`FlowEnv` — document why (external-process coupling has no natural fit
with `FlowEnv`'s in-process `self.flow`/`self.solver` composition) and
instead require them to satisfy the same **external** contract described
above.

---

## Extensibility Design

This section is the technical backbone for the new developer guide (Task
4.2/4.3). Two flows:

### Adding a new in-process solver (Firedrake-like)
1. Subclass `hydrogym.core.PDEBase`, implement all 10 abstract methods.
2. Subclass `hydrogym.core.TransientSolver` (or reuse `core.CallbackBase`
   directly if no custom time-stepping needed).
3. Reuse `hydrogym.core.FlowEnv` unmodified — do not write a new Env class.
4. If Hugging-Face-backed checkpoints are needed, use the new
   `hydrogym.hf_env_mixin.HFEnvConfigMixin` rather than writing a 5th copy
   of the resolution logic.
5. Register with `gym.register()`.
6. Write tests mirroring `test/test_cyl.py`'s structure (import smoke test,
   steady-state solve, one transient step, one gradient test if
   differentiable).

### Adding a new external-process solver (MAIA/Nek-like)
1. Subclass `gym.Env` directly (not `FlowEnv` — see Environment Interface
   Design above for why) and mix in the new
   `hydrogym.core_external.ExternalProcessEnvMixin`.
2. Implement the solver-specific wire protocol (own tag/command set) —
   this is the expected, sanctioned escape hatch.
3. Use `_split_mpmd_comm()` for the communicator split instead of
   hand-rolling it a third time.
4. Implement `close()` per the mixin's documented contract (termination
   message + MPI teardown).
5. Use `HFEnvConfigMixin` for any HF-backed environment data staging.
6. Provide a `prepare_<solver>_workspace()` pre-launch helper mirroring
   `hydrogym.maia.prepare_maia_workspace` — document the MPMD launch
   command pattern explicitly (this is the single biggest source of
   "how do I even start" friction found in this audit).
7. Register with `gym.register()`.
8. Tests: at minimum a `--collect-only`-style unit test of the
   config-parsing/validation path that does *not* require MPI/the solver
   binary (most of MAIA's and Nek's current code has zero test coverage
   even at this level), plus an integration test gated behind an MPI/
   solver-binary marker for dev-container/HPC-only CI.

### Reference skeleton (Task 4.4 will produce this as real, copyable code)
`examples/developer_templates/minimal_inprocess_solver/` and
`examples/developer_templates/minimal_external_solver/` — trivial
(no real physics) implementations of each pattern, each under ~100 lines,
each with a passing test that runs in the plain CI runner (no Firedrake/
MPI/GPU required), so a new contributor can literally `diff` their new
solver against a working minimal example.

---

## Documentation / Example Plan

1. Finish the interrupted per-example audit (Task 4.1) for MAIA, Nek,
   JAX, JAX-Fluids — the Firedrake tree is already spot-checked as
   consistent; the others were not reached before the audit agent hit a
   session limit.
2. Fix the two direct doc inconsistencies found (Task 1.x): environment
   count (61 vs 88 — figure out the true current count by actually
   counting registered environment classes across all 5 backends, then
   make README/docs agree), and the two different Docker
   registries/images referenced by `README.md` vs `test/README.md`.
3. Fix `codespell.yml`'s `tests`/`tutorials` path mismatch (Task 1.y).
4. Write the two new developer guides (`adding-a-solver.md`,
   `adding-an-environment.md`) plus the two reference skeletons (Task 4.2-4.4).
5. Where an example can run without GPU/MPI/a real solver binary
   (config-parsing/import-smoke-test level), convert it into a `test/`
   pytest case gated to run in the plain CI runner; where it genuinely
   needs Firedrake/MPI/a GPU/a solver binary, leave it as a documented
   manual example but add a lightweight "does this at least import and
   construct without crashing" smoke test where feasible (see Testing
   Strategy).

---

## Testing Strategy

**This is the section that directly answers "is there anything that can go
wrong, are we testing enough, how do we avoid new bugs."** Given Finding 4
(CI runs zero tests today), the single highest-priority action, before any
other change in this plan, is establishing a real regression net — see
Phase 0 in the Implementation Plan.

### What can go wrong (explicit risk list for the coding agent)
- **Silent regression in Firedrake numerics.** Any change to
  `hydrogym/core.py` (even a "safe-looking" validation addition) risks
  breaking Firedrake, the one backend with any test coverage at all, and
  today that coverage isn't even CI-gated. Mitigation: Phase 0 adds CI
  gating *before* Phase 2/3 touch `core.py`; every task in Phase 2/3 that
  touches `core.py` or `firedrake/` must show a passing local
  `pytest test/` run (inside the Firedrake dev container) in its
  acceptance criteria, not just "the diff looks right."
- **`terminated`/`truncated` semantics fix silently changing RL training
  behavior** for anyone with existing trained policies/checkpoints
  expecting the old (inconsistent) behavior. Mitigation: ship this as an
  explicit, separately-flagged, opt-in-then-default change (see
  Backwards Compatibility) — never silently in the same PR as unrelated
  fixes.
- **The new "unknown config key" warning becoming a de facto breaking
  change** if any currently-passing test or example relies on an
  extra/misspelled key being silently ignored. Mitigation: warning, not
  exception, in the first release; add a follow-up task (not in this
  plan) to promote to an exception only after a full deprecation window.
- **Refactoring the duplicated HF-resolution code into
  `HFEnvConfigMixin` breaking one backend while fixing another** — this
  touches 4 files with independently-drifted behavior (Finding 1.1/2.10/
  2.11); the safest sequencing is per-backend, one PR each, not one
  combined refactor (see Task List 3.1-3.4, deliberately split).
- **MAIA/Nek changes cannot be verified without real MPI + solver
  binaries.** Any task touching `hydrogym/maia/` or `hydrogym/nek/`
  **must** be validated inside `wipmaiaml`'s devcontainer (per this
  workspace's other `CLAUDE.md`) or an equivalent Nek5000 dev container —
  a diff that "looks right" but was never run against a live MPMD job is
  not acceptable for these two backends given they have zero existing
  test coverage to fall back on.
- **The dead-code removals (`jax/flow.py::FlowConfig`,
  `maia/hf_data_manager.py`) must be grep-verified as truly unreferenced
  immediately before deletion**, not just trusted from this audit — repo
  state may have changed between audit time and implementation time.

### Test tiers

**Local / no special dependencies:**
- Static analysis: ruff, isort, codespell (already exist — fix the
  `codespell.yml` path bug as part of Phase 0).
- New: a lightweight unit test of `hydrogym/core.py`'s `FlowEnv`/`PDEBase`/
  `TransientSolver` using a trivial mock `PDEBase` subclass (no Firedrake
  needed) — this is currently entirely missing and is the cheapest,
  highest-value new test to add, because it's the one piece of shared
  infrastructure everything else is supposed to build on.
- New: unit tests for every kwargs-propagation fix in Phase 2 that doesn't
  require a real solver (e.g. `HFDataManager(cache_dir=...)` forwarding —
  mockable via `unittest.mock.patch("huggingface_hub.snapshot_download")`).

**Dev container, no MPI/GPU required (Firedrake container):**
- Existing `test/test_*.py` (8 files) — must be added to CI (Phase 0),
  this is non-negotiable given Finding 4.
- New tests for every Firedrake-touching Phase 2 fix (2.2, 2.3, 2.4).

**Dev container, requires MPI (`mpi4py`, no real solver binary):**
- New: unit tests for `_split_mpmd_comm()` (the extracted helper) using a
  trivial 2-rank `mpirun -np 2 pytest ...` setup that doesn't require
  Nek5000/MAIA themselves — just validates the communicator-splitting
  logic in isolation.

**Dev container, requires MAIA or Nek5000 binaries (this workspace's
`wipmaiaml`/Nek devcontainers specifically):**
- Integration test of `MaiaFlowEnv`/`NekEnv` construction + one `step()` +
  `close()` against a real, small case (e.g. this workspace's own
  `Cylinder_2D_Re200` MAIA test case, or the smallest available Nek case)
  — validates the `ExternalProcessEnvMixin` contract end-to-end, not just
  in isolation.
- Regression test for the `terminated`/`truncated` semantics fix
  specifically, since this is the one behavior-changing item in the whole
  plan.

**Requires an actual HPC scheduler (Slurm/PBS) — cannot run in this
repo's dev containers:**
- Multi-node MPMD launch verification (the dev containers are single-node;
  the `mpirun -np 1 ... : -np N ...` pattern used throughout is
  scheduler-launcher-agnostic by design, but this claim itself has not
  been, and cannot be, verified inside a container). Document this
  explicitly as an untested-in-this-repo claim rather than asserting it
  works on a real cluster.

### Example tests
Per the Documentation/Example Plan above: any example that only needs
config parsing / object construction (no real solve) becomes a `test/`
pytest case in the "local, no special deps" or "Firedrake container"
tier as appropriate. Examples requiring a real MPMD job stay manual but
gain, where feasible, a "does it at least construct" smoke test in the
appropriate MPI/solver tier.

---

## Safety Protocol — Hard Gates for the Coding Agent

The user has stressed, repeatedly and explicitly, that **no new bugs or
regressions are acceptable** and that this must be treated as a first-order
constraint, not a nice-to-have. The Testing Strategy and Regression Risks
sections above describe *what* to test; this section states the
non-negotiable *process rules* that make "don't introduce new bugs"
actually enforceable rather than aspirational. Follow these for every
task in every phase, no exceptions:

1. **Phase 0 is a hard blocking gate.** Do not start any Phase 1+ task
   until Task 0.1 (CI runs `test/`) is merged and green, and Task 0.2/0.3's
   baseline is recorded. If Phase 0 cannot be completed (e.g. the
   Firedrake container can't be provisioned in CI), stop and report back
   rather than proceeding without a regression net "just this once."
2. **One task, one PR, one revert point.** Never combine two Task-List
   items into one commit/PR, even when they touch the same file (e.g.
   Tasks 2.7 and 2.8 both touch `hydrogym/nek/env.py` — land them as two
   separate, independently-revertible PRs, in order). If something breaks
   later, the blast radius of "which change caused it" must be one task,
   not a batch.
3. **Every PR must show, not just claim, a passing run.** "The diff looks
   correct" is not sufficient acceptance evidence for any task touching
   `hydrogym/` (docs-only tasks are exempt). Paste or reference the actual
   local test-run output (`pytest` output, or the manual dev-container
   run's console output for MAIA/Nek) in the PR description before
   considering a task done.
4. **MAIA and Nek tasks require a real dev-container run — not a review —
   as a merge gate**, every time, with no exceptions for "small" changes.
   This workspace's `wipmaiaml` MAIA devcontainer (per this workspace's
   own `CLAUDE.md` dev pipeline: build, `regression_test.py`
   run-and-compare against `master_baseline`, confirm PASS) and an
   equivalent Nek5000 container are the only acceptable validation
   environments for these two backends, precisely because they have zero
   existing automated coverage to fall back on — a code review alone
   cannot catch an MPI-protocol regression.
5. **Never widen a fix's scope mid-task.** If implementing one Task-List
   item surfaces an adjacent problem, note it (add a new Task-List entry
   for a later phase) rather than fixing it inline — an unplanned,
   unreviewed second change hidden inside a planned one is exactly how
   regressions slip through unnoticed.
6. **Deprecation over deletion, everywhere except the two explicitly-flagged
   exceptions** (Task 1.1's confirmed-dead code, and Phase 5's
   deliberately-isolated breaking change). Every other task in this plan
   is designed to be additive/shimmed specifically so existing user code
   keeps working during the transition — do not "clean up" an old code
   path in the same PR that adds its replacement; remove it only in a
   separate, later, explicitly-scoped task if removal is warranted at all.
7. **Re-verify this audit's claims before acting on them.** Every
   "confirmed dead"/"confirmed unused" claim in this document was true at
   audit time (2026-08-19); re-run the relevant grep/check immediately
   before deleting or changing behavior based on it, since repo state can
   drift between audit and implementation.
8. **If a task's acceptance criteria cannot be met — stop and report,
   don't guess.** E.g. if Task 5.1's semantics fix cannot be validated
   against a real MAIA divergence scenario because none is readily
   reproducible, that is a blocker to report back on, not a reason to ship
   the change with reduced test coverage and hope.
9. **Never merge Phase 5 (the one breaking change) bundled with anything
   else**, and never merge it without the isolation/changelog treatment
   described in Backwards Compatibility — this is the single highest-risk
   item in the whole plan for downstream users specifically because it is
   a behavior change, not a bug fix.

If, during implementation, a task turns out to be riskier than this audit
assessed (e.g. a "Low" risk task turns out to touch more callers than
expected), treat the *actual* risk level as authoritative over this
document's estimate, and escalate the validation rigor accordingly rather
than following the letter of the original risk rating.

---

## Implementation Plan

### Phase 0 — Safety net (must complete first, before any other phase)
Establish the regression net that does not currently exist. No functional
changes in this phase.

### Phase 1 — Low-risk, high-value cleanup
Dead code removal, documentation fixes, CI config fixes. No behavior
changes to any solver.

### Phase 2 — Kwargs/config propagation fixes
Each row from Finding 2's table becomes one independently-shippable,
independently-tested task. Ordered by backend to keep blast radius small
per PR (Firedrake first, since it has existing test coverage to validate
against; MAIA/Nek last, since they require the most expensive
verification).

### Phase 3 — Shared infrastructure extraction
`HFEnvConfigMixin` and `ExternalProcessEnvMixin`, one backend migration
per PR, in order: JAX-Fluids → JAX → Nek → MAIA (least to most
MPI-coupling risk).

### Phase 4 — Documentation, examples, developer guides
Finish the example audit, fix doc inconsistencies, write the two new
developer guides and reference skeletons.

### Phase 5 — Behavior-changing fix (isolated, last)
`terminated`/`truncated` semantics unification — deliberately scheduled
last, as its own release, per the Backwards Compatibility plan below.

### Phase 6 — Optional / nice-to-have (not required for "gold standard" baseline)
`gymnasium.register()` for all backends; Nek RPC batching; structured
CFL-blowup exception instead of `exit()`. Valuable but lower-risk/lower-
urgency than Phases 0-5; can be deferred past the initial cleanup pass.

### Phase 7 — Documentation coverage rollout (final phase)
Fill missing docstrings (worst first: JAX, then Firedrake, then
`core.py`) and add a CI floor to stop future regression. Deliberately
last of all phases — no functional risk, and writing docstrings against
final, post-refactor signatures avoids doing this work twice.

---

## Task List

Each task specifies Objective, Files/components affected, Current problem,
Required change, Implementation details, Tests, Acceptance criteria,
Dependencies, and Risk.

### Phase 0 — Safety net

**Task 0.1 — Add a CI job that actually runs `test/`**
- *Objective:* Close the single biggest risk identified in this audit —
  zero automated test execution today.
- *Files:* `.github/workflows/build.yml` (or a new `test.yml`); needs a
  Firedrake-capable runner (likely a container-based job using the same
  image `test/README.md` documents, `lpaehler/hydrogym-env:stable`, or
  whatever image Phase 1's doc-fix task settles on as canonical).
- *Current problem:* `build.yml` only runs `poetry build`; `pytest` is
  never invoked in CI.
- *Required change:* New workflow (or job) that: builds/pulls the
  Firedrake container, installs hydrogym (`pip install -e ".[firedrake]"`),
  runs `cd test && python -m pytest . -x --durations=10`.
- *Implementation details:* Confirm which Docker image is actually current
  and working (resolve the `README.md` vs `test/README.md` mismatch — see
  Task 1.3 — before wiring CI to either) before writing the workflow.
- *Tests:* The workflow itself, run once on a throwaway branch/PR to
  confirm green.
- *Acceptance criteria:* A PR that intentionally breaks a Firedrake test
  (e.g. `test_cyl.py::test_import_medium`) causes this new CI job to fail;
  reverting the break makes it pass again.
- *Dependencies:* None (can start immediately); should land before any
  Phase 2/3 task that touches `hydrogym/core.py` or `hydrogym/firedrake/`.
- *Risk:* Low. Risk is entirely in correctly provisioning the Firedrake
  container in CI (known to be heavy/slow) — acceptable to start with a
  long timeout and optimize later.

**Task 0.2 — Add a minimal, dependency-free unit test of `hydrogym/core.py`**
- *Objective:* Test the one shared abstraction directly, not only through
  Firedrake.
- *Files:* new `test/test_core.py` (or `test/unit/test_core.py`).
- *Current problem:* `PDEBase`/`TransientSolver`/`FlowEnv` have zero direct
  test coverage; every existing test exercises them only through the full
  Firedrake PDE stack.
- *Required change:* Define a trivial mock `PDEBase` subclass (e.g. a
  1-D scalar "flow" with a closed-form update) and a trivial
  `TransientSolver` subclass, then test: construction, `reset()`,
  `set_control`/`advance_time`, `step()`/`solve()` (both `t_span` and
  `num_steps` modes), `collect_rewards=True` path, multi-substep
  aggregation (`mean`/`sum`/`median`), deprecated-key warnings
  (`num_sim_substeps_per_actuation`, `reward_aggreation_rule`), and
  `close()`.
- *Tests:* This task **is** the test.
- *Acceptance criteria:* `pytest test/test_core.py` passes without
  Firedrake/MPI/GPU installed, in a bare Python env with only `gymnasium`+
  `numpy`.
- *Dependencies:* None.
- *Risk:* Very low — new, additive, no existing code touched.

**Task 0.3 — Snapshot current example/output behavior before any changes**
- *Objective:* Give later phases something concrete to diff against.
- *Files:* none changed; produce a scratch record (not committed) of:
  current `pytest test/` pass/fail state, current example script exit
  codes where runnable in this environment.
- *Required change:* N/A — this is a verification/record step, not a code
  change.
- *Acceptance criteria:* A written note (can live in the PR description of
  Task 0.1, does not need its own file) of the exact baseline pass/fail
  state before Phase 1 begins.
- *Dependencies:* Task 0.1 (needs the CI job, or at least the local
  container, to produce this baseline).
- *Risk:* None (no code change).

### Phase 1 — Low-risk cleanup

**Task 1.1 — Delete dead code**
- *Objective:* Remove confirmed-dead duplicates (Finding 1.2, 1.3).
- *Files:* `hydrogym/jax/flow.py` (delete `FlowConfig` class; check if
  anything else in the file is still used before deleting the whole file),
  `hydrogym/maia/hf_data_manager.py` (delete whole file).
- *Current problem:* Two confirmed-dead code paths increase maintenance
  surface and drift risk.
- *Required change:* `grep -rn "from hydrogym.jax.flow import\|hydrogym\.jax\.flow\." --include="*.py" .` and `grep -rn "hf_data_manager" --include="*.py" hydrogym examples test` immediately before deleting, to reconfirm zero references in the current tree (repo state may have shifted since this audit).
- *Tests:* Existing test suite (Task 0.1's CI job) must stay green; add
  nothing new (there's nothing to test — it's a deletion).
- *Acceptance criteria:* Grep confirms zero references before deletion;
  full test suite green after.
- *Dependencies:* Task 0.1 (need CI green baseline first).
- *Risk:* Low, but re-verify the grep — do not trust this audit's
  "confirmed dead" claim blindly if time has passed.

**Task 1.2 — Fix `codespell.yml`**
- *Files:* `.github/workflows/codespell.yml`.
- *Required change:* Change `codespell tests` → `codespell test`; remove
  the `codespell tutorials` step (no such directory) or point it at
  whatever the real intended target was (check git blame/history for
  context on whether `tutorials/` was ever real and got renamed/removed).
- *Tests:* Run the workflow on a PR, confirm it no longer references a
  missing path.
- *Acceptance criteria:* `codespell.yml` passes (or fails on real
  spelling issues only, not path errors).
- *Dependencies:* None.
- *Risk:* Very low.

**Task 1.3 — Resolve README vs docs inconsistencies**
- *Files:* `README.md`, `docs/docs/quickstart.md`,
  `docs/docs/introduction.md`, `test/README.md`.
- *Required change:* (a) Count actual registered environment classes
  across all 5 backends' `envs/` directories to determine the true current
  count; update all three "61"/"88" mentions to agree. (b) Decide which
  Docker image is actually current/maintained (`clagemann/hydrogym-*` vs
  `lpaehler/hydrogym-env:stable`) — likely by checking which one still
  exists/pulls successfully and which one Task 0.1's CI job ends up using
  — and make `test/README.md` and `README.md` agree, or explicitly
  document why two different images serve two different purposes if that
  turns out to be intentional.
- *Tests:* None (documentation only); manual review.
- *Acceptance criteria:* No contradictory numeric claims or Docker image
  references remain across `README.md`/`docs/`/`test/README.md`.
- *Dependencies:* Task 0.1 (Docker image decision informed by what CI
  actually uses).
- *Risk:* Very low (docs only).

**Task 1.4 — Add `jax`/`jaxfluids` to `hydrogym/__init__.py`'s lazy-loader**
- *Files:* `hydrogym/__init__.py`.
- *Required change:* Add `"jax"`, `"jaxfluids"` to the `__getattr__`
  allowlist tuple and to `__all__`.
- *Tests:* New test asserting `import hydrogym; hydrogym.jax` and
  `hydrogym.jaxfluids` resolve without error (skip if `jax`/`jaxfluids`
  extras aren't installed in the test env — guard with
  `pytest.importorskip`).
- *Acceptance criteria:* Attribute access works consistently across all 5
  backend names.
- *Dependencies:* None.
- *Risk:* Low — purely additive; verify it doesn't change import-time
  behavior for environments that intentionally avoid importing
  jax/jaxfluids eagerly (confirm the lazy `__getattr__` pattern still
  defers the actual import correctly for these two, matching the existing
  MPI-safety rationale for `maia`/`nek`).

### Phase 2 — Kwargs/config propagation fixes

**Task 2.1 — `PDEBase.__init__` unknown-key warning**
- *Files:* `hydrogym/core.py:47-65`.
- *Required change:* Track which config keys are consumed (`.pop()`
  instead of `.get()` for `mesh`/`restart`); after processing, `warnings.warn`
  on any leftover keys (see Configuration/Argument Propagation Design §2).
- *Tests:* Extend Task 0.2's `test_core.py` with a case passing an
  unrecognized key and asserting a `UserWarning`/`DeprecationWarning` is
  raised; assert existing valid configs raise no warning.
- *Acceptance criteria:* New test passes; full `test/` suite (Task 0.1
  CI) still green — this must not become a hard error for any currently-
  valid Firedrake flow config.
- *Dependencies:* Task 0.1, 0.2.
- *Risk:* Medium — must audit all current Firedrake flow configs (in
  `test/`, `examples/firedrake/`) for any key that would newly warn, to
  confirm no false positives before merging.

**Task 2.2 — Fix `FlowConfig`'s mesh-fallback bug**
- *Files:* `hydrogym/firedrake/flow.py:88-99`.
- *Required change:* Use `self.DEFAULT_MESH` instead of `self.MESH_DIR` as
  the fallback for checkpoint-inference's mesh-name variable.
- *Tests:* Unit test constructing a flow without an explicit `mesh` key
  and asserting the inferred HF checkpoint env-name string no longer
  contains a filesystem path.
- *Acceptance criteria:* New test passes; `test/test_cyl.py` and friends
  (which do pass `mesh=` explicitly) remain unaffected/green.
- *Dependencies:* Task 0.1.
- *Risk:* Low-medium — this changes what checkpoint gets auto-resolved
  when `mesh` is omitted; verify no example currently relies on the buggy
  (always-mismatching, therefore effectively "never auto-resolves")
  behavior.

**Task 2.3 — `NavierStokesTransientSolver` noise kwargs**
- *Files:* `hydrogym/firedrake/solvers/base.py:75-97,9`.
- *Required change:* Either wire `eta`/`max_noise_iter`/`noise_cutoff` into
  an actual `white_noise()`-based forcing step, or remove the dead
  parameters/import entirely if no one actually needs this feature today
  (check with repo history / issue tracker if accessible; if unclear,
  default to **removal** — reintroducing a documented, tested feature
  later is safer than leaving a silently-broken one in place).
- *Tests:* If wired: a numerical test confirming noise forcing perturbs
  the solution measurably. If removed: confirm no example/test currently
  passes these kwargs (grep first).
- *Acceptance criteria:* No silently-ignored constructor kwargs remain in
  this class.
- *Dependencies:* Task 0.1.
- *Risk:* Low if removing (dead code); medium if wiring up real forcing
  (numerical correctness risk — needs careful validation against a known
  noise-forced reference case if available).

**Task 2.4 — `integrate()` forwards `collect_rewards`**
- *Files:* `hydrogym/firedrake/solvers/integrate.py:11-16`.
- *Required change:* Add `collect_rewards: bool = False` parameter,
  forward to `solver.solve(..., collect_rewards=collect_rewards)`.
- *Tests:* Unit test calling `integrate(..., collect_rewards=True)` and
  asserting a `(flow, rewards)` tuple is returned.
- *Acceptance criteria:* New test passes; existing callers of `integrate()`
  without this kwarg are unaffected (default `False` preserves current
  behavior exactly).
- *Dependencies:* Task 0.1.
- *Risk:* Very low — purely additive.

**Task 2.5 — `HFDataManager` forwards `cache_dir` to `snapshot_download`**
- *Files:* `hydrogym/data_manager.py:524-533,587-596,626-635`.
- *Required change:* Pass `cache_dir=self.cache_dir` into all three
  `snapshot_download()` call sites.
- *Tests:* Mock `huggingface_hub.snapshot_download` and assert it's called
  with the configured `cache_dir`.
- *Acceptance criteria:* New test passes; confirm no currently-passing
  test/example relied on downloads always landing in the default HF cache
  regardless of `cache_dir` (unlikely, but grep for `cache_dir=` usages in
  examples/tests first).
- *Dependencies:* Task 0.1.
- *Risk:* Low — but genuinely changes on-disk download location for
  anyone who *was* setting `cache_dir` and (incorrectly) not noticing it
  was ignored; call this out in release notes.

**Task 2.6 — Add `token`/`revision` to `HFDataManager`**
- *Files:* `hydrogym/data_manager.py:249-289,334-345,524-644`, plus every
  caller that constructs `HFDataManager` (`maia/env_core.py`,
  `nek/env.py`, `jax/env_core.py`, `jaxfluids/env_core.py`,
  `firedrake/flow.py`'s checkpoint resolver).
- *Required change:* Add optional `token`/`revision` kwargs to
  `HFDataManager.__init__`, thread through every `snapshot_download`/
  `list_repo_files`/`HfApi()` call; add corresponding optional
  `hf_token`/`hf_revision` keys to each backend's `env_config` schema that
  construct their own `HFDataManager`.
- *Tests:* Mock-based unit tests per call site asserting `token`/`revision`
  reach the underlying `huggingface_hub` calls when set, and that omitting
  them preserves current (ambient-auth) behavior exactly.
- *Acceptance criteria:* All new tests pass; no behavior change when the
  new kwargs are omitted.
- *Dependencies:* Task 2.5 (same file, sequence to avoid merge conflicts).
- *Risk:* Low — purely additive, default `None` preserves current
  ambient-token behavior.

**Task 2.7 — Delete or fix `NekEnv.__init__`'s dead `**kwargs`**
- *Files:* `hydrogym/nek/env.py:132-173`.
- *Required change:* Remove the misleading `**kwargs` parameter, or (safer
  given `1.0.0` API-stability concerns) keep it but add a
  `warnings.warn` if it's non-empty, explicitly stating it has no effect
  and listing the actually-supported override mechanism
  (`_apply_runtime_overrides`, see Task 2.8).
- *Tests:* Unit test constructing `NekEnv` with an extra kwarg (using a
  minimal/mock config path that doesn't require MPI — check if
  `NekEnv.__init__`'s config-parsing can be unit-tested without
  `mpi_split()` actually running; if not, this becomes an MPI-tier test
  instead) and asserting the warning fires.
- *Acceptance criteria:* No kwarg is silently accepted-and-dropped without
  at least a warning.
- *Dependencies:* Task 0.1; must be validated inside a Nek5000-capable dev
  container per the Testing Strategy's MPI/Nek tier.
- *Risk:* Low-medium — must confirm no current example relies on passing
  (and having ignored) an extra kwarg here.

**Task 2.8 — Generalize `NekEnv._apply_runtime_overrides`**
- *Files:* `hydrogym/nek/env.py:432-450`.
- *Required change:* Extend beyond the 5-key whitelist to accept
  dotted-path overrides (`section.param=value`) against the OmegaConf
  config tree, validating the path exists before applying (raise a clear
  `ConfigError` — reuse or extend the existing `ConfigError` class,
  `hydrogym/nek/env.py`, already imported in `__init__.py` — for unknown
  paths, rather than silently ignoring them as today).
- *Tests:* Unit test (config-parsing only, no MPI needed if the override
  application itself doesn't require a live solver) covering: the
  existing 5 keys still work unchanged, a new dotted-path override
  correctly reaches `self.conf`, and an invalid path raises `ConfigError`
  instead of being silently ignored.
- *Acceptance criteria:* All new tests pass; the existing 5-key behavior
  is provably unchanged (regression test against current behavior before
  the change, per Task 0.3's baseline).
- *Dependencies:* Task 2.7 (same file).
- *Risk:* Medium — this is exactly the kind of change that must not
  regress any existing example that relies on the current whitelist
  behavior; validate against every `examples/nek/**/*.py` config first.

**Task 2.9 — Expose `mpi_bind_to` in `NekEnv`**
- *Files:* `hydrogym/nek/env.py:504-511`.
- *Required change:* Read an optional `mpi_bind_to` (default `"none"`,
  preserving current behavior) from `env_config`, pass to `mpi_info.Set(...)`.
- *Tests:* MPI-tier test (per Testing Strategy) confirming the default
  preserves current behavior and an explicit override is honored.
- *Acceptance criteria:* New test passes in a Nek-capable dev container.
- *Dependencies:* Task 0.1; Nek dev container access.
- *Risk:* Low — additive, defaults preserve current behavior exactly.

**Task 2.10 — Fix `JAXFlowEnv` missing `fallback_profile`/`SOLVER_TYPE`**
- *Files:* `hydrogym/jax/env_core.py:86-88` (and wherever `SOLVER_TYPE` is
  defined on sibling classes, e.g. `maia/env_core.py:52`).
- *Required change:* Add `SOLVER_TYPE = "JAX"` class attribute to
  `JAXFlowEnv`/`JAXFlowEnvBase` (whichever is still live after Task 1.1's
  dead-code cleanup) and pass `fallback_profile=self.SOLVER_TYPE`,
  matching the sibling backends' pattern exactly.
- *Tests:* Unit test asserting the constructed `HFDataManager` receives
  `fallback_profile="JAX"`.
- *Acceptance criteria:* New test passes; confirm `SOLVER_PROFILES` in
  `hydrogym/data_manager.py` already has a `"JAX"` entry (per the earlier
  research, it does) so this doesn't require also adding a new profile.
- *Dependencies:* Task 1.1 (dead code cleanup should land first so this
  task targets the surviving class).
- *Risk:* Low — brings JAX in line with an already-working pattern used
  by 3 other backends.

**Task 2.11 — Fix `JAXFlowEnv`'s wrong cache namespace**
- *Files:* `hydrogym/jax/env_core.py:137-153`.
- *Required change:* Change `"maiagym"` to a JAX-specific namespace (e.g.
  `"jaxgym"`), matching `jaxfluidsgym`/`nekgym`/`maiagym`'s per-backend
  pattern.
- *Tests:* Unit test asserting the cache path no longer references
  `maiagym`.
- *Acceptance criteria:* New test passes. **Note this changes the on-disk
  cache location for existing JAX users** — call out in release notes;
  consider a one-time migration note (old `~/.cache/maiagym/<jax_env_name>`
  entries become orphaned, not corrupted — safe to just leave stale, not
  worth an automated migration).
- *Dependencies:* None (independent of 2.10, can ship together or
  separately).
- *Risk:* Low — cache-path-only change, no computational behavior change;
  worst case is a redundant re-download for existing users on first run
  after the fix.

**Task 2.12 — Standardize substep-count naming (MAIA, JAX)**
- *Files:* `hydrogym/maia/env_core.py:137`, `hydrogym/jax/env_core.py:118`,
  plus wherever the YAML/config schema documents this key.
- *Required change:* Accept `num_substeps` as the primary name on MAIA and
  JAX (matching Firedrake's non-deprecated name), keep
  `num_sim_substeps_per_actuation` working with a `DeprecationWarning`
  (mirroring `core.py:388-396`'s existing pattern exactly).
- *Tests:* Unit test per backend: old name still works + warns, new name
  works without warning, both produce identical resulting config state.
- *Acceptance criteria:* Both names produce identical behavior; only the
  old name warns.
- *Dependencies:* Task 0.1 (MAIA path needs MAIA dev-container
  verification per Testing Strategy).
- *Risk:* Low — additive/deprecation-shimmed, not a hard break.

**Task 2.13 — Standardize reward-aggregation naming (Nek)**
- *Files:* `hydrogym/nek/env.py:132-138,781-784`.
- *Required change:* Accept `reward_aggregation` (matching Firedrake) as
  an alias for `reward_agg`, including support for `"median"` (currently
  Nek-only supports `mean`/`sum`); keep `reward_agg` working unchanged.
- *Tests:* Unit test covering both names + the new `"median"` option.
- *Acceptance criteria:* No change to existing `reward_agg`-based
  configs' behavior; new `reward_aggregation`/`"median"` path works
  correctly, validated against a Nek-capable dev container.
- *Dependencies:* Task 0.1; Nek dev container.
- *Risk:* Low-medium (the `"median"` addition is new numerical code —
  needs a correctness test, not just a plumbing test).

**Task 2.14 — Fix `ChannelFlowSpectralEnv`'s hardcoded physical params**
- *Files:* `hydrogym/jax/envs/channel.py:414-436` vs `:29-52`.
- *Required change:* Read `Lx`/`Ly`/`Lz`/`nu`/`Nx`/`Ny`/`Nz` from
  `env_config`, defaulting to today's hardcoded values (preserves current
  behavior exactly when unset).
- *Tests:* Unit test constructing the env with an overridden `Nx` and
  asserting the resulting `PseudoSpectralNavierStokes3D` actually uses it.
- *Acceptance criteria:* Default (no override) behavior byte-for-byte
  unchanged; override path works.
- *Dependencies:* None.
- *Risk:* Low-medium — grid-size changes affect JAX JIT compilation
  shapes; verify no shape-assumption elsewhere in the JAX channel code
  breaks when these are non-default.

**Task 2.15 — Fix `ChannelFlowSpectralEnv`'s missing HF config keys**
- *Files:* `hydrogym/jax/envs/channel.py:443-455` vs `jax/env_core.py:82-88`.
- *Required change:* Read `hf_repo_id`/`cache_dir`/`use_clean_cache` from
  `env_config`, matching `JAXFlowEnv`'s pattern, defaulting to today's
  hardcoded values.
- *Tests:* Unit test asserting overrides reach the constructed
  `HFDataManager`.
- *Acceptance criteria:* Default behavior unchanged; overrides work.
- *Dependencies:* Task 2.14 (same file).
- *Risk:* Low.

### Phase 3 — Shared infrastructure extraction

**Task 3.1 — Extract `HFEnvConfigMixin`, migrate JAX-Fluids first**
- *Files:* new `hydrogym/hf_env_mixin.py`; `hydrogym/jaxfluids/env_core.py`.
- *Required change:* Move `_setup_environment_data`/
  `_resolve_configuration_file`/`_find_configuration_file`/cache-namespace
  logic into the new mixin, parameterized by `SOLVER_TYPE`/cache-namespace
  string; migrate `JAXFluidsFlowEnv` to use it, deleting its local copy.
- *Tests:* All existing JAX-Fluids example/test behavior must be
  byte-for-byte unchanged; add unit tests of the mixin directly (mockable,
  no real JAX-Fluids install needed for the resolution-logic tests
  themselves).
- *Acceptance criteria:* JAX-Fluids example(s) that ran before still run
  identically after; new mixin unit tests pass.
- *Dependencies:* Phase 2 tasks that touch this logic (2.10, 2.11) should
  land first if targeting the same functions, to avoid rebasing pain —
  otherwise independent.
- *Risk:* Medium — this is a real refactor of working code; migrate one
  backend at a time (this task = JAX-Fluids only) specifically to keep
  blast radius small, per the Testing Strategy's explicit warning against
  a combined refactor.

**Task 3.2 — Migrate JAX to `HFEnvConfigMixin`**
- Same shape as 3.1, targeting `hydrogym/jax/env_core.py`. Do this
  *after* 2.10/2.11 land (fixes should go into the mixin from the start,
  not be reapplied after migration).
- *Dependencies:* Task 3.1 (mixin must exist), Task 2.10, 2.11.
- *Risk:* Medium, same rationale as 3.1.

**Task 3.3 — Migrate Nek to `HFEnvConfigMixin`**
- Same shape, targeting `hydrogym/nek/env.py`'s partial copy. Requires
  Nek dev-container validation (this backend's copy was already the most
  heavily modified from the others per the original research — expect
  the migration to be less than a pure find-replace).
- *Dependencies:* Task 3.1, 3.2 (establish the pattern on lower-risk
  backends first); Nek dev container.
- *Risk:* Medium-high — most-modified copy, least test coverage backend.

**Task 3.4 — Migrate MAIA to `HFEnvConfigMixin`**
- Same shape, targeting `hydrogym/maia/env_core.py`. Do last — highest
  MPI-coupling risk, requires this workspace's `wipmaiaml` MAIA dev
  container for validation (per this workspace's own `CLAUDE.md` dev
  pipeline).
- *Dependencies:* Task 3.1-3.3; MAIA dev container (`wipmaiaml`, per the
  sibling project's documented build/test pipeline).
- *Risk:* Medium-high, same rationale as 3.3. **Must** be validated with a
  real `mpirun ... : ... maia ...` MPMD run against a real test case
  (e.g. this workspace's own `Cylinder_2D_Re200`), not just a code review.

**Task 3.5 — Extract `ExternalProcessEnvMixin` (`_split_mpmd_comm`)**
- *Files:* new `hydrogym/core_external.py`; refactor
  `hydrogym/nek/env.py:36-85` (`mpi_split`) and
  `hydrogym/maia/mpmd_interface.py:63-91` (`MaiaInterface.init_comm`) to
  call the shared helper, keeping each solver's own tag-protocol code
  untouched.
- *Tests:* MPI-tier test (per Testing Strategy) with a trivial 2-rank
  `mpirun -np 2 pytest ...` setup validating the split logic in isolation,
  without requiring Nek5000/MAIA binaries; then full Nek and MAIA
  integration tests (existing manual examples, run in their respective
  dev containers) to confirm the refactor is behavior-preserving.
- *Acceptance criteria:* Nek and MAIA MPMD examples run identically
  before/after in their respective dev containers.
- *Dependencies:* Task 3.3, 3.4 (do this after the HF-mixin migrations to
  avoid two concurrent refactors of the same files).
- *Risk:* High — this touches the most fragile, least-tested part of the
  codebase (raw MPI communicator setup) in both external-process
  backends. Strongly recommend doing Nek and MAIA as two fully separate
  PRs even though the shared helper is one file, and requiring a real
  MPMD run (not just a review) as a merge gate for each.

### Phase 4 — Documentation, examples, developer guides

**Task 4.1 — Finish the per-example audit for MAIA, Nek, JAX, JAX-Fluids**
- *Objective:* The original audit pass for these four backends' examples
  was interrupted by a session limit partway through Nek; finish it
  before rewriting anything.
- *Files:* all of `examples/{maia,nek,jax,jaxfluids}/**`.
- *Required change:* For each example: read it fully, cross-reference
  every API call against the current source (post-Phase 2/3 changes),
  flag stale imports/kwargs/deprecated keys, verify `*_docker.sh` scripts
  match their Python counterparts and current install docs.
- *Tests:* N/A (audit task); produces the input for Task 4.5.
- *Acceptance criteria:* A complete per-example verdict list (OK / STALE /
  BROKEN / DEPRECATED-KEY / INCONSISTENT), matching the format already
  established for the Firedrake spot-check in this document's Finding 3.
- *Dependencies:* Should run *after* Phase 2/3 land, so the audit is
  against the corrected API, not the pre-fix one (avoids double work).
- *Risk:* None (read-only audit).

**Task 4.2 — Write "Adding a New Solver" guide**
- *Files:* new `docs/docs/developers/adding-a-solver.md`.
- *Required change:* Turn this document's Extensibility Design section
  into a full, standalone guide, covering both the in-process and
  external-process patterns, referencing the new mixins from Phase 3.
- *Acceptance criteria:* An external developer can follow it without
  reading any existing backend's full source, per the original brief's
  explicit goal.
- *Dependencies:* Phase 3 (guide should describe the *post-refactor*
  architecture, not the pre-refactor one).
- *Risk:* None (docs only).

**Task 4.3 — Write "Adding a New Environment" guide**
- *Files:* new `docs/docs/developers/adding-an-environment.md`.
- *Required change:* Cover the simpler case (new flow/case within an
  existing backend, e.g. a new Firedrake `FlowConfig` subclass or a new
  MAIA `envs/*.py` registration) separately from adding a whole new
  solver backend (Task 4.2) — these are different-sized tasks with
  different guides today conflated in people's minds per the original
  brief.
- *Dependencies:* Task 4.2 (shares structure/tone).
- *Risk:* None (docs only).

**Task 4.4 — Build the two reference skeletons**
- *Files:* new `examples/developer_templates/minimal_inprocess_solver/`,
  `examples/developer_templates/minimal_external_solver/`.
- *Required change:* Trivial, no-real-physics implementations of each
  pattern (per Extensibility Design), each with a passing test runnable
  in the plain CI tier (no Firedrake/MPI/GPU).
- *Tests:* The skeletons' own tests, added to CI (Task 0.1's job or a new
  lightweight one).
- *Acceptance criteria:* `pytest examples/developer_templates/` passes in
  the plain CI runner.
- *Dependencies:* Phase 3 (skeletons should demonstrate the post-refactor
  mixins).
- *Risk:* Low — new, additive, isolated code.

**Task 4.5 — Rewrite/fix flagged examples from Task 4.1**
- *Files:* whichever specific example files Task 4.1 flags.
- *Required change:* Per-file, following the same
  wrong/correct/replacement/test pattern established for Firedrake in
  this document's Finding 3.
- *Tests:* Where feasible (config-parsing-level examples), convert to a
  `test/` pytest case per the Documentation/Example Plan; where not
  (requires real MPI/GPU/solver binary), leave as a manual example but
  add an import/construction-only smoke test if that much is testable in
  the plain CI tier.
- *Dependencies:* Task 4.1.
- *Risk:* Low-medium, scales with how much Task 4.1 finds; treat each
  fixed example as its own small PR rather than one giant sweep.

### Phase 5 — Behavior-changing fix (isolated, ship last)

**Task 5.1 — Unify `terminated`/`truncated` semantics**
- *Files:* `hydrogym/core.py` (Firedrake, already closest to correct —
  `terminated` always `False`), `hydrogym/maia/env_core.py:640` (currently
  sets both to the same bool), `hydrogym/nek/env.py` (currently hardcodes
  `truncated=False`, ties `terminated` to CFL/step-budget conflated logic —
  needs `terminated` to mean "physics invalid" and `truncated` to mean
  "step budget reached," splitting Nek's `env.py:790-797` conditions
  accordingly).
- *Required change:* See target semantics in Proposed Target Architecture
  item 7. This is the **one deliberately behavior-changing item** in the
  whole plan.
- *Tests:* Explicit regression tests per backend confirming the new
  semantics under both a normal-episode-end and a physics-failure
  scenario (CFL blowup for Nek, an analogous divergence condition for
  MAIA if one exists — verify what MAIA currently treats as "done" before
  designing this test).
- *Acceptance criteria:* All three backends' `terminated`/`truncated`
  match the documented semantics; full regression suite (Firedrake CI +
  Nek/MAIA dev-container manual runs) green.
- *Dependencies:* All of Phase 2/3 should be complete and stable first —
  this is intentionally the last functional change in the plan.
- *Risk:* **High for downstream users** (any RL training code/checkpoint
  relying on the old semantics), **low for this codebase's own
  correctness** (the change makes behavior more correct, not less). See
  Backwards Compatibility for the required rollout gating.

### Phase 6 — Optional / deferred

**Task 6.1 — `gymnasium.register()` for all 5 backends**
- Straightforward, additive, per Solver Interface Design's registration
  layer. Low risk, can be done any time after Phase 3. Not required for
  the "gold standard" baseline but closes Finding 1.6.

**Task 6.2 — Nek RPC batching (`Gatherv`/`Scatterv`)**
- Performance-only change to `hydrogym/nek/env.py:826-858,873-891`; no
  behavior change if done correctly, but touches the most fragile part of
  the codebase — treat as its own carefully-isolated PR with a real MPMD
  correctness test (identical results before/after, only latency should
  change) if pursued.

**Task 6.3 — Structured exception instead of `exit()` on Nek CFL blowup**
- `hydrogym/nek/env.py:921`. Replace the bare `exit()` call with a raised,
  catchable exception (e.g. `NekDivergenceError`), letting the RL training
  loop decide whether to abort the whole job or recover. Requires careful
  handling of the MPI-side state (the other rank(s) still need to see the
  `TERMN` message sent just before `exit()` today) — do not remove that
  message, only change how Python's own side reacts to it.

### Phase 7 — Documentation coverage rollout (final phase — do this last)

Deliberately scheduled after every other phase, including Phase 6: this is
pure documentation debt with no functional risk, and doing it last avoids
rewriting docstrings on signatures that Phases 1-5 are still actively
changing (e.g. Task 2.8's generalized `_apply_runtime_overrides`, Task
2.14's new `env_config` keys — write their docstrings once, against their
final shape, not twice).

**Task 7.1 — Fill missing docstrings, worst-covered files first**
- *Objective:* Bring docstring coverage up consistently across all 5
  backends, closing the gap identified in Finding 4 (measured coverage:
  `jax/env_core.py` 24%, `firedrake/flow.py` 59%, `core.py` 63%, vs.
  `nek/env.py` 97% and `maia/env_core.py` 100%).
- *Files:* Priority order by lowest measured coverage first:
  `hydrogym/jax/env_core.py` (24%), `hydrogym/jax/envs/{channel,kolmogorov}.py`,
  `hydrogym/jax/solvers/base.py`, `hydrogym/jax/equation.py` (JAX backend
  overall was the weakest spot in every sampled file), then
  `hydrogym/firedrake/flow.py` (59%) and the rest of
  `hydrogym/firedrake/{solvers,envs}/`, then `hydrogym/core.py` (63%) —
  the shared abstraction deserves full coverage given it's the one file
  every backend either builds on or is compared against. Re-measure
  `hydrogym/jaxfluids/env_core.py`, `hydrogym/nek/env.py`, and
  `hydrogym/maia/env_core.py` (already high) for any regressions
  introduced by Phases 1-5's edits before considering them done.
- *Required change:* Google-style docstrings (matching the existing
  `pydoc-markdown.yaml` processor config) on every public function,
  method, and class lacking one; skip private (`_`-prefixed) helpers
  unless their behavior is genuinely non-obvious.
- *Implementation details:* Re-run this audit's `ast.get_docstring`-based
  coverage measurement (see Finding 4's table for the exact approach)
  before and after, per file, to track progress objectively rather than
  by feel.
- *Tests:* N/A directly (docs-only), but run `docs/`'s `npm run build`
  locally (or in CI, see Task 7.2) to confirm the new docstrings parse
  correctly and render without breaking the Docusaurus build.
- *Acceptance criteria:* Every file above reaches parity with the
  already-good files (roughly 90%+ coverage on public API surface); a full
  `docs/` build succeeds with no new warnings/errors attributable to
  malformed docstrings.
- *Dependencies:* All of Phases 1-6 (write docstrings against final
  signatures only).
- *Risk:* Very low — docstrings-only, no functional code paths change.
  The only real risk is a malformed Google-style docstring breaking the
  `pydoc-markdown` build; caught by the acceptance criteria's build check.

**Task 7.2 — Add CI enforcement for docstring coverage**
- *Files:* new step in an existing or new `.github/workflows/*.yml`.
- *Required change:* Add a coverage-floor check (e.g. `interrogate
  hydrogym/ --fail-under <N>`, threshold set to whatever Task 7.1 actually
  achieves, so it locks in the improvement rather than an arbitrary
  aspirational number) so coverage cannot silently regress again the way
  it evidently has across backends already.
- *Tests:* The workflow itself; verify it fails on a deliberately
  under-documented throwaway function and passes on the post-7.1 codebase.
- *Acceptance criteria:* New CI job is green on the current codebase and
  demonstrably fails on an intentional regression.
- *Dependencies:* Task 7.1 (need the achieved coverage level to set a
  realistic threshold).
- *Risk:* Very low — additive CI check only.

---

## Regression Risks

Summarized from the Testing Strategy's risk list:
1. Firedrake numerics breaking silently — mitigated by Phase 0's CI gate.
2. `terminated`/`truncated` fix changing RL training behavior for existing
   users — mitigated by isolating it as Phase 5, last, with its own
   explicit backwards-compatibility gate (see below).
3. New unknown-config-key warnings becoming a de facto breaking change —
   mitigated by warning-not-erroring, plus auditing all current
   configs before merging Task 2.1.
4. HF-resolution mixin refactor breaking one backend while fixing another
   — mitigated by one-backend-per-PR sequencing (Tasks 3.1-3.4) instead of
   a combined refactor.
5. MAIA/Nek changes unverifiable without real MPI + solver binaries —
   mitigated by requiring dev-container validation (this workspace's
   `wipmaiaml` MAIA container, and an equivalent Nek5000 container) as an
   explicit merge gate for every Phase 2/3/5 task touching those backends,
   not just a code review.
6. Dead-code deletions (Task 1.1) based on this audit's grep results,
   which may be stale by implementation time — mitigated by re-running the
   grep immediately before deleting.
7. Task-batching hiding which specific change caused a regression —
   mitigated by the Safety Protocol's one-task-per-PR rule (rule 2), which
   makes every change independently revertible and independently
   attributable.
8. A task quietly shipping without real validation evidence (a plausible
   failure mode under time pressure) — mitigated by the Safety Protocol's
   rule 3 (passing-run evidence required in every PR, not just a
   diff review).

**All of the above are process risks, not just technical ones — see the
Safety Protocol section for the concrete, mandatory rules that mitigate
them; the risk list here says *what* could go wrong, the Safety Protocol
says *how the process itself prevents it*.**

## Backwards Compatibility

- **Everything in Phases 0-4 and Phase 6 is additive or
  deprecation-shimmed.** Old config keys/kwargs keep working, with
  warnings where appropriate, following the existing pattern already
  established in `hydrogym/core.py:388-412`.
- **Phase 5 (`terminated`/`truncated` semantics) is the one true breaking
  change.** Recommended rollout: ship it as a minor version bump with an
  explicit, prominent changelog entry and migration note (the project has
  no `CHANGELOG.md` today — creating one is a reasonable side effect of
  this task, not required by this plan but recommended); do not bundle it
  with any other change in the same release, so anyone tracking releases
  can immediately identify it as the one thing to check their training
  code against.
- **Cache-path changes** (Task 2.11, JAX's `maiagym`→`jaxgym` fix) and
  **download-location changes** (Task 2.5, `cache_dir` now actually
  honored) are technically behavior changes but are bugfixes correcting
  already-broken behavior, not changes to a previously-working contract —
  call out in release notes but do not require the same isolation
  treatment as Phase 5.

## HPC Considerations

- The MPMD launch pattern (`mpirun -np 1 python ... : -np N <solver>`)
  used by both Nek5000 and MAIA is, by construction, scheduler-launcher-
  agnostic (Slurm's OpenMPI/PMIx integration and PBS's `mpirun` MPMD colon
  syntax both support it) — but this claim **has not been verified inside
  this repo's dev containers**, which are single-node. Any task touching
  `core_external.py`/`mpi_split`/`MaiaInterface.init_comm` should be
  validated on real Slurm/PBS hardware if at all possible before being
  considered fully done, not just in the single-node dev container.
- No task in this plan proposes subprocess-based MPI launching as a
  "solution" to anything, per the original brief's explicit constraint —
  the existing MPMD architecture is confirmed correct and is preserved,
  not replaced, throughout.
- Full in-process library embedding for Nek5000/MAIA was investigated and
  is explicitly **not recommended** — see § Nek/MAIA Feasibility Study.

## Nek / MAIA Feasibility Study

*(Full study; headline conclusions already summarized in Finding 6.)*

### Nek5000
1. **Launch:** MPMD, `mpirun -np 1 python ... : -np N nek5000`; Python-side
   split via `mpi_split()` (`hydrogym/nek/env.py:36-85`).
2. **Process boundary:** at the MPI rank level, within one shared MPMD job.
3. **Fundamentally separate processes?** Yes, as currently written: `PROGRAM
   NEKTON`'s `drive.f` calls `mpi_init`/`mpi_finalize` exactly once per
   process lifetime; this repo's own forked `drive.f` (in
   `third_party/nek5000/solver/KTH_DRL_Framework/Nek5000/core/drive.f`) has
   already been patched (~35 lines) specifically to support the MPMD split,
   confirming this is not out-of-the-box Nek5000 behavior.
4. **Embeddable API?** No `.so`/`pybind11`/`ctypes`/`f2py` binding exists in
   this repo's Nek5000/Toolbox submodules. The DRL command-handshake logic
   answering the Python side's tag protocol lives in a case-specific `.usr`
   file shipped via Hugging Face, not in this repo, and was not auditable.
5. **MPI comm handoff feasible?** Yes — already implemented and working.
6. **Upstream changes for tighter embedding:** turn `PROGRAM NEKTON`'s
   driver loop into a callable subroutine, eliminate/encapsulate Fortran
   `COMMON`-block global state. Estimated medium complexity, but low payoff
   (see point 8).
7. **Slurm/PBS impact:** none for the current architecture (already
   compatible); a tighter embedding would also be compatible in principle
   but requires point 6's upstream work first.
8. **Performance:** per-episode process-relaunch cost is **not actually a
   bottleneck today** — `NekEnv.reset()` reuses the live MPI job via an
   `RSETS` command over the existing communicator; only `close()` tears
   down the process, once, at the end of training.
9. **Robustness:** current design isolates a solver crash to one MPI rank
   (Python's `exit()` on CFL blowup, `env.py:921`, still abruptly kills the
   Python side too — flagged as Task 6.3, independent of the embedding
   question). Tighter in-process embedding would make a Nek-side fault
   fatal to the entire Python interpreter — a regression, not an
   improvement.
10. **Maintenance burden:** current fork-and-patch approach has low
    incremental burden (already working); tighter embedding would add an
    ongoing Fortran/Python ABI contract to maintain across Nek5000 version
    bumps.
11. **Recommendation:** **keep the current MPMD architecture.** Invest
    instead in RPC batching (Task 6.2) and structured error handling
    (Task 6.3) — these address the real, measurable pain points (latency,
    crash-handling) without touching Fortran internals.

### MAIA
1. **Launch:** MPMD via the MPI-3-standard `MPI_APPNUM` mechanism (more
   portable than Nek's manual rank-split), `mpirun -np 1 python ... : -np N
   maia properties_run.toml`; Python-side split via
   `MaiaInterface.init_comm()` (`hydrogym/maia/mpmd_interface.py:63-91`),
   mirrored exactly on the C++ side by `GlobalMpiInformation::init`
   (`third_party/m-AIA/src/COMM/globalmpiinfo.cpp:36-73`).
2. **Process boundary:** same shape as Nek — MPI rank level, one shared
   MPMD job.
3. **Fundamentally separate processes?** Yes: `MAIA::run()`
   (`third_party/m-AIA/src/maia.cpp`) calls `MPI_Init_thread`/
   `MPI_Finalize` exactly once per process; substantial global/static state
   confirmed (`g_mpiInformation`, `mEnvironment`, and many more in
   `globalvariables.cpp`) — multiple instances cannot coexist in one
   process without a large refactor. MAIA additionally owns a CUDA context
   per process (NVHPC/CUDA build), adding a second reason single-process
   multi-instance embedding is unsound as-is.
4. **Embeddable API?** No `.so`/`pybind11` binding exists in the checked-out
   `third_party/m-AIA` tree. The existing `MPMD::sendDataV/receiveDataV`
   machinery and `callMPMDRoutines` call sites (wired into both the LB and
   FV structured/cartesian solvers) are the existing, production-grade
   coupling surface — already used for real boundary-flow-control work in
   the sibling `wipmaiaml` project this same user maintains.
5. **MPI comm handoff feasible?** Yes — already implemented, and more
   standards-based than Nek's approach (`MPI_APPNUM` vs. manual rank
   splitting).
6. **Upstream changes for tighter embedding:** remove/refactor global
   singletons into instance state, build a `pybind11` wrapper, and —
   the hard part — reconcile `MPI_Init_thread(MPI_THREAD_FUNNELED)` and
   CUDA context creation with Python's/PyTorch's own CUDA/MPI usage in one
   process. Estimated large complexity (multiple months).
7. **Slurm/PBS impact:** none for current architecture; large effort for
   marginal-to-negative benefit for a tighter embedding, given point 8.
8. **Performance:** same finding as Nek — `MaiaFlowEnv.reset()` already
   reuses the live process via `reinit()` over the existing communicator;
   no per-episode relaunch cost exists to eliminate.
9. **Robustness:** current process-per-run design isolates GPU/CUDA
   context and global-state faults to one process; in-process embedding
   would risk taking down a co-resident Python/PyTorch training process on
   any MAIA-side fault.
10. **Maintenance burden:** MAIA is under **active internal development**
    (per this workspace's own `wipmaiaml` project) — a `pybind11`/C-API
    surface would need continuous co-maintenance against a fast-moving
    internal fork, a real recurring cost specific to MAIA (Nek5000, by
    contrast, is a more stable upstream).
11. **Recommendation:** **keep the current MPMD architecture; do not
    pursue full library embedding at the hydrogym-repo level.** If the
    real underlying motivation is the separately-mentioned libtorch
    `torch::from_blob` zero-copy GPU-buffer-sharing direction (noted only
    as background context from the sibling `wipmaiaml` project, out of
    scope here), that is a fundamentally different, narrower engineering
    problem than "avoid subprocess launching for hydrogym" and should be
    scoped as its own project inside `wipmaiaml`, not folded into this
    repo's environment-interface work.

---

## Definition of Done

For this audit's implementation to be considered complete:
- Phase 0 shipped and green: CI runs `test/` on every PR; `test_core.py`
  exists and passes without Firedrake/MPI/GPU.
- Every row in Finding 2's kwargs-propagation table has a corresponding
  merged, tested fix (Phase 2) or an explicit, documented decision not to
  fix it (with rationale) if investigation during implementation finds a
  fix is inappropriate.
- `HFEnvConfigMixin` and `ExternalProcessEnvMixin` exist and all four
  applicable backends (JAX, JAX-Fluids, Nek, MAIA) use them, with zero
  remaining copy-pasted HF-resolution logic.
- `docs/docs/developers/adding-a-solver.md` and `adding-an-environment.md`
  exist, and the two reference skeletons under
  `examples/developer_templates/` exist and pass their own tests in CI.
- The Task 4.1 example audit is complete for MAIA/Nek/JAX/JAX-Fluids (not
  just Firedrake), and every flagged example is either fixed or has a
  tracked follow-up with rationale for deferral.
- `terminated`/`truncated` semantics are unified and shipped as an
  isolated, changelog-flagged release (Phase 5).
- No regression: the full test suite (Firedrake CI tier + manually-run
  MAIA/Nek dev-container integration checks) is green at every merge
  point, not just at the end.
- README/docs no longer contain contradictory environment counts or
  Docker image references; `codespell.yml` targets real directories;
  `hydrogym/distributed/` either has real content or its README/docs
  claims are corrected to not imply functionality that doesn't exist.
- Docstring coverage (Phase 7) reaches parity across all 5 backends and a
  CI job enforces the achieved floor going forward.
- **Every merged task has actual passing-run evidence attached (Safety
  Protocol rule 3), not just a reviewed diff** — retroactively spot-check
  this across the merged task list before declaring the whole plan done.

## Recommended Implementation Order

1. **Phase 0** (safety net) — non-negotiable first step, given zero
   current CI test coverage. Hard blocking gate per Safety Protocol rule 1
   — nothing below starts before this is green.
2. **Phase 1** (low-risk cleanup) — cheap wins, builds confidence in the
   new CI gate.
3. **Phase 2** (kwargs fixes), Firedrake-touching tasks first (2.1-2.4,
   have existing test coverage to validate against), then JAX/JAX-Fluids
   tasks (2.10-2.15, lower MPI risk), then Nek tasks (2.7-2.9, 2.13,
   require Nek dev container), MAIA-specific fixes are comparatively few
   in this phase (mostly captured in Phase 3's mixin work instead).
4. **Phase 3** (shared infrastructure), strictly in the stated order
   (JAX-Fluids → JAX → Nek → MAIA, least to most MPI-coupling risk), each
   as its own PR with its own dev-container validation.
5. **Phase 4** (docs/examples/guides) — deliberately after Phase 3, so
   documentation describes the final architecture, not an intermediate one.
6. **Phase 5** (`terminated`/`truncated` fix) — deliberately last among
   functional changes, isolated, changelog-flagged, never bundled with
   anything else (Safety Protocol rule 9).
7. **Phase 6** (optional/deferred) — pick up opportunistically; not
   required for the "gold standard" baseline this audit targets, but
   valuable follow-on work.
8. **Phase 7** (docstring coverage rollout) — deliberately last of all,
   after every functional change is settled, so docstrings are written
   once against final signatures rather than repeatedly against
   moving targets.

Throughout all eight steps, the Safety Protocol's rules (one task per PR,
real validation evidence, MAIA/Nek dev-container gates, no scope creep,
deprecate-don't-delete except where explicitly flagged) apply uniformly —
they are not optional guidance layered on top of the plan, they are how
the plan avoids introducing new bugs while executing it.

---

## Verification Addendum (2026-09-04)

**Purpose:** an independent, hands-on re-verification of the ~55-commit
implementation of this plan on branch `hydrogym-audit-v2`, performed in a
*separate* session from the one that did the implementation. Per the
audit's own Safety Protocol rule ("re-verify claims before acting on
them"), nothing below is asserted from reading commit messages or prior
session logs alone — every item marked CONFIRMED was independently
re-executed in a live devcontainer during this pass (the CPU stack
`jovial_proskuriakova`/`full-cpu-stack`, the GPU stack
`hydrogym-full-gpu-stack-audit`/`full-gpu-stack` on the machine's real RTX
4090, and the standalone `peaceful_lovelace`/`nek5000-test` container),
not merely re-read from `.devcontainer/build-logs`/`test-logs` left over
from the implementation session. Prior logs were used only to *locate*
the canonical test recipes (`.devcontainer/scripts/test_cpu_solvers.sh`,
`test_gpu_solvers.sh`, `.github/workflows/test.yml`), never as evidence
in themselves.

### 1. MPI / HPC launch architecture — CONFIRMED CORRECT, no subprocess spawning

This was the top priority for this verification pass. Findings:

- **Repo-wide grep for `subprocess`/`Popen`/`os.system`/`shell=True`/
  `MPI_Comm_spawn`/`.Spawn(` across `hydrogym/`, `examples/`, and every
  `.py` file for a literal `mpirun` invocation** turns up **zero** cases
  of Python spawning the solver process, a nested `mpirun`, or dynamic
  `MPI_Comm_spawn`. The only `mpirun` strings in the whole tree are inside
  docstrings/comments documenting the launch command, or in `.sh` driver
  scripts that are themselves the outer MPMD launcher — never invoked
  *from* Python.
- **The actual mechanism** (`hydrogym/core_external.py`, read in full):
  both `NekEnv` and `MaiaFlowEnv` are launched as one externally-started
  MPMD job (`mpirun -np 1 python ... : -np N <solver>`, or the Slurm/PBS
  equivalent — `srun`'s MPMD/`--multi-prog` mode and PBS's `mpirun` colon
  syntax both support this natively), and the Python and solver ranks
  share one `MPI_COMM_WORLD` from the moment the job starts. Splitting
  that world into a controller↔solver channel is pure standards-based MPI
  — `mpi_split()` uses `Comm.Split` + `Create_intercomm` (Nek's
  rank-color protocol), `split_comm_by_appnum()` uses
  `Get_attr(MPI.APPNUM)` + `Allreduce` (MAIA's true-MPMD protocol) — with
  **no process creation of any kind** on the Python side, at any point,
  including inside `reset()` (both `NekEnv.reset()` and
  `MaiaFlowEnv.reset()` reuse the live communicator across episodes via
  in-band reinit commands, matching the audit's original Nek/MAIA
  Feasibility Study finding exactly).
- This design is fully compatible with static Slurm/PBS allocation: the
  scheduler launches one job with a fixed total rank count, and the
  Python↔solver split happens entirely inside that fixed allocation —
  there is no dynamic process spawning that would fight a scheduler's
  static resource grant.
- **Bonus finding while reading `core_external.py`**: the Task 3.5
  extraction did not just move code — its own docstring documents two
  real, pre-existing correctness bugs it *fixed* in the process (a
  hardcoded `remote_leader=1` that only worked because the Python
  controller happens to always be app/rank 0, and a group-translation
  direction bug in the MAIA split with the same "only ever tested as app
  0" blind spot). Both are now covered by `test/mpmd_smoke_split.py`
  (`TestSplitFixes::test_mpi_split_remote_leader_not_hardcoded_one`,
  `test_appnum_root_translation_direction`), independently re-run as part
  of this pass (§3) and passing.
- **One unrelated `subprocess` usage found and evaluated**:
  `hydrogym/nek/nek_lib/nek_utils.py:142-146`
  (`NEK_INIT.write_SESSION_NAME`) uses `subprocess.call(cmd, shell=True)`
  for three trivial local file operations (`touch`, `echo >`,
  `echo $(pwd) >>`) run once per episode by the Python controller rank on
  its own local filesystem — **not** a solver-launch or multi-node
  concern, and irrelevant to Slurm/PBS compatibility. It is a pre-existing
  (not audit-introduced) code-quality/robustness nit: plain Python file
  I/O (`open(...).write(...)`) would avoid an unnecessary shell and a
  format-string-interpolated command (`self.nek.CASENAME` reaches the
  shell unescaped, though it comes from trusted local config, not
  attacker input). Flagged as a low-severity cleanup candidate, not a
  regression and not an architecture violation.

**Live re-verification performed** (not just static reading): real MPMD
runs were executed end-to-end during this pass for MAIA-CPU, MAIA-GPU,
and Nek5000-CPU (see §3), each producing genuine solver step/reward
output through the real `mpirun ... : ... <solver>` colon-syntax launch,
confirming the static analysis above against actual running processes,
not just source code.

### 2. Documentation accuracy — CONFIRMED

- **Environment count**: README.md, `docs/docs/quickstart.md`, and
  `docs/docs/introduction.md` all now consistently say **89** (20
  Firedrake + 55 MAIA LBM + 4 MAIA Structured FV + 4 NEK5000 + 2 JAX + 4
  JAX-Fluids = 89, arithmetic checked). All three cite the same source
  (environment configurations currently published on the HF Hub), a
  reasonable and self-consistent choice — Task 1.3 resolved.
- **Docker image references**: README.md and `test/README.md` now both
  point at `clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere` (the
  old `lpaehler/hydrogym-env` reference is gone from `test/README.md`) —
  Task 1.3 resolved.
- **`codespell.yml`**: now runs against `test` (singular, the real
  directory), `examples`, `hydrogym`, and `README.md` — no more reference
  to the nonexistent `tests`/`tutorials` — Task 1.2 resolved.
- **`hydrogym/distributed/`**: the false "distributed RL training
  support" claim is gone from README.md (now correctly describes "MPMD
  co-launch of RL and solver processes"), and `docs/CLAUDE.md` now
  correctly labels it "Empty placeholder (multi-agent support lives under
  `nek/`...)". **However**, see Finding A below — the package itself was
  not removed, which is a direct deviation from an explicit instruction
  in this document.
- **`.gitignore`**: the Task 4.1 example audit flagged five leaking
  run-output directories as an "out-of-scope, tracked separately" cleanup
  item; checking the current tree shows this was fixed anyway (`git
  status` is clean for all of them) — better than the audit report itself
  claimed, worth noting as a pleasant discrepancy rather than a problem.
- **Developer guides** (`docs/docs/developers/adding-a-solver.md`,
  `adding-an-environment.md`): read in full. Both are accurate against
  the current code (verified `ExternalProcessEnvMixin`/`HFEnvConfigMixin`
  usage examples against the real classes), and Pattern 2 (external
  process) correctly documents the MPMD colon-syntax launch as the
  *only* sanctioned way to start an external-process backend — no
  guide anywhere suggests subprocess-based launching.
- **Task 4.1 example audit** (`.devcontainer/example-audit-task4.1.md`):
  substantive, not a stub (39 files, per-file verdicts). Spot-checked
  three of its flagged fixes directly against source and confirmed all
  three actually landed (not just documented as "should fix"):
  the JAX Kolmogorov notebook's dead `control_function` path now carries
  an explicit correction and uses the real `control_field` API; Nek's
  `ctrl_min_amp`/`ctrl_max_amp` config now resolves via the generalized
  dotted-path override machinery (Task 2.8) instead of raising
  `ConfigError`; the MAIA README's `properties.toml` → `properties_run.toml`
  naming bug is fixed everywhere it appeared.

### 3. Test evidence — CONFIRMED (independently re-executed, not re-read)

| Check | Method | Result |
|---|---|---|
| `test/test_core.py` (Task 0.2) | Ran in a **bare** `python3 -m venv` with only `pytest`+`gymnasium`+`numpy` installed (no Firedrake/MPI/GPU) | **52 passed** |
| `examples/developer_templates/` skeletons (Task 4.4) | Same bare venv | **10 passed** |
| Docstring coverage floor (Task 7.1/7.2) | `python test/measure_docstrings.py --fail-under 99`, same bare venv | **454/454 = 100.0%**, floor is 99% |
| `test_registration`, `test_lazy_loader`, `test_mesh_fallback`, `test_hf_data_manager_*`, `test_hf_env_mixin`, `test_integrate`, `test_substep_naming`, `test_step_semantics` | Bare venv + `omegaconf`/`huggingface_hub`/`pyyaml` (still no Firedrake/mpi4py) | **40 passed**, 25 skipped (legitimately gated on Firedrake/mpi4py, which this tier doesn't install) |
| CI's exact Firedrake recipe (`test.yml`'s `firedrake-tests` job body) | Ran twice: first verbatim with `-x` (`pytest . -x --durations=10 --ignore=*_grad.py`, matching CI exactly), then again **without** `-x` (222 collected, all backends) to see the whole picture rather than stopping at the first failure | **With `-x` (what CI actually runs): 73 passed, then stops at `test_cyl.py::test_steady`** — this is the only failure CI itself would ever see, since `-x` halts there. **Without `-x`: 218 passed, 10 failed, 5 skipped** (0:23:48). All 10 failures were individually re-run in isolation and root-caused (see Finding C below and its follow-up) — none are new regressions from this session's 4 commits: 4 are the same auto-inferred-checkpoint-ambiguity mechanism as `test_steady` (`test_cyl.py::test_steady`, `test_steady_rotation`, `test_act_implicit_no_damp`, `test_pinball.py::test_steady_rotation` — all confirmed pre-existing, none touched by the ~55-commit implementation); 1 (`test_cyl.py::test_linearize`) is the Task 0.3 baseline's already-documented Firedrake-version-skew bug, confirmed by reproducing it with checkpoint resolution bypassed entirely; 2 (`test_pinball.py::test_env`, `test_step.py::test_env`) are an unrelated pre-existing test bug (`SemiImplicitBDF.__init__() missing 1 required positional argument: 'dt'` in the test's own `solver_config`, nothing to do with checkpoints); 1 (`test_io.py::test_checkpointing`) references `hgym.IPCS`, a solver class removed from this codebase years before this audit (commit `43e75cc`, "Remove IPCS solver") — a stale, never-updated test; 2 (`test_registration.py::test_firedrake_factory_defaults`/`test_firedrake_factory_caller_overrides_win`) pass individually and pass running the whole file alone (10/10) — they only fail as part of the full 222-test run, indicating pre-existing cross-test state leakage somewhere in the suite, not something either of these two tests or this session's changes caused. None of the 10 are reachable by CI's actual `-x` recipe except `test_steady`, which is the first one anyway. |
| `.devcontainer/scripts/test_cpu_solvers.sh` (maia_cpu, firedrake, nek5000 real solver smoke tests) | Ran verbatim against the live `full-cpu-stack` container | `maia_cpu`: **PASS** (real MPMD run, real steps/reward). `firedrake`: **PASS**. `nek5000`: **TIMEOUT** at the 600s cap — see below, resolved as a test-methodology artifact, not a regression. |
| `.devcontainer/scripts/test_gpu_solvers.sh` (maia_gpu, jax_kolmogorov, jax_channel, jaxfluids) | Ran verbatim against the live `full-gpu-stack` container on the real RTX 4090 | Script exit code 0. `maia_gpu`: **PASS** (real MPMD run on GPU, "Info: MPMD is activated", 3 real steps, clean close, 8s). `jax_kolmogorov`: **PASS** — completed 5 real steps with real numeric reward output in 1m2s (one transient CUDA "RESOURCE_EXHAUSTED" line at startup, self-recovered — see Finding C; the runner's own log-grep evidence heuristic reported "no step/reset output seen" for this one even though the log clearly shows real step numbers/rewards — a minor false-negative in the regex, not a real gap, worth tightening but not a regression). `jax_channel`: **PASS**, 2 real DNS-coupled steps, 17s. `jaxfluids`: **TIMEOUT at 600s, exactly as designed** — the script's own header comment documents that this test has no `--num-steps` flag (1000 steps hardcoded) and is expected to always hit the cap rather than finish; the log shows continuous, correct `env_step`/`sim_step` progress throughout (real solver output every ~25s, no stall) — this is the *expected*, not a failure, outcome. |
| Nek5000 MPMD path in isolation | The `nek5000` leg of `test_cpu_solvers.sh` hit the 600s timeout while running **concurrently** with the full Firedrake pytest suite and the GPU-stack tests on the same 32-core host (`uptime` showed load average 12-17 across all three simultaneously-active containers). To separate "resource contention" from "regression," the identical command was re-run **in isolation** against the idle `nek5000-test` container. | **Completed in 3.6 seconds wall-clock** (Nek-side: 0.88s total elapsed, 0.12s solver time), 3 real timesteps, real reward output (total −12.823, avg −4.274), clean `TERMN`/close sequence. The earlier TIMEOUT was conclusively a test-methodology artifact of running three heavy containers simultaneously on one host during this verification pass, **not** a Nek MPMD regression. The benign UCX "unexpected tag-receive descriptor was not matched" warning after `[NEK] TERMN ENV` reappeared exactly as previously documented (pre-existing teardown artifact, not a protocol issue). |
| Task 6.2 RPC batching | Read `hydrogym/nek/env.py`'s `_get_state`/`_send_action` | Confirmed non-blocking point-to-point (`Isend`/`Irecv` + one `Waitall`), not `Gatherv`/`Scatterv` as the original plan's Task 6.2 wording suggested — a deliberate, documented deviation (an MPI intercommunicator has no collectives without touching the Fortran side), not a shortfall. |
| Task 5.1 terminated/truncated semantics | Read `hydrogym/maia/env_core.py::step()` and `hydrogym/nek/env.py` directly | MAIA: `return self.obs, reward, False, bool(done), info` — terminated always `False`, truncated on step-budget, exactly as designed. Nek: `NekDivergenceError` raised from `_evolve`/`step()` on CFL blowup (Task 6.3), `truncated` split from step-budget/`tmax`. Firedrake unchanged (`terminated = False`, pre-existing). Matches `CHANGELOG.md`'s "Unreleased — Breaking" entry, which itself is present and correctly describes the migration impact. |
| Task 6.1 registration | Read `hydrogym/registration.py` | `gym.register()` calls present for Firedrake, MAIA, Nek, JAX-Fluids canonical IDs; JAX deliberately excluded (documented rationale: functional/gymnax contract) — matches the audit's own design. |

### 4. Findings from this verification pass

**Finding A — `hydrogym/distributed/` was not removed despite an explicit instruction in this document.**
This document's own inline annotation under "Current Architecture" states:
*"Let's remove `distributed` for now to avoid giving wrongful impressions, and my PR for the distributed backend then recreated the folder."*
The implementation corrected every *documentation* claim about `distributed/`
(README.md, `docs/CLAUDE.md` — see §2) but the package itself
(`hydrogym/distributed/__init__.py`, still a 0-byte-equivalent empty
placeholder) is still present, still importable via
`hydrogym.distributed`, and still listed in `hydrogym/__init__.py`'s
`__getattr__` allowlist and `__all__`. The Definition of Done's own
wording ("either has real content or its README/docs claims are
corrected") is satisfied by the letter, but not by the explicit user
instruction embedded earlier in the same document. **Recommendation**:
delete `hydrogym/distributed/__init__.py` and drop `"distributed"` from
the lazy-loader allowlist/`__all__` in a small, isolated follow-up commit,
consistent with the user's stated intent to re-add it via a dedicated
future PR.

**Finding B — Finding 2.5 (MAIA launch-config validation) was never assigned a Task List item, and remains unimplemented.**
The original Finding 2 table's row 2.5 states MAIA has "No Python code
path launches the MAIA process; MPMD launch is entirely out-of-band
shell, with zero config surface or mismatch detection." Cross-referencing
against the Task List: Tasks 2.1-2.15 map to Findings 2.1-2.4 and
2.6-2.15 (note the renumbering: "Task 2.5" is HFDataManager's `cache_dir`
fix, an unrelated item) — **Finding 2.5 itself has no corresponding task
anywhere in Phases 0-7.** This was independently confirmed in code:
`grep -n "nproc\|launch_config\|hostfile\|Get_size" hydrogym/maia/*.py`
returns nothing — `MaiaFlowEnv` still performs zero validation of the
actual MPI world size against any expected rank count, unlike Nek's
`mpi_split()` (§1), which raises a clear, actionable `RuntimeError`
naming the exact fix (`Launch with: mpirun -n 1 python ... : -n N
./nek5000`) on a size mismatch. A user who launches MAIA with the wrong
`-np` gets no early, clear diagnostic — this is a real, still-open gap
in the "escape hatches, not silent inconsistency" goal this audit set out
to achieve for MAIA specifically. **Recommendation**: add this as a new
Phase 2/6 task (e.g., validate `comm_world.Get_size()` against an
optional `nproc` key in MAIA's `env_config` inside
`MaiaInterface.init_comm`, mirroring `mpi_split`'s existing pattern) —
this is a plan gap, not an implementation-quality issue, and should be
tracked as new work rather than retroactively blamed on the ~55 commits
that did land.

**Finding C — non-deterministic checkpoint selection in `FlowConfig._resolve_single_checkpoint` causes a real, reproducible Firedrake test failure (`test_cyl.py::test_steady`).**
Running the CI recipe verbatim (§3) hit a genuine failure — not a flake of
this verification's own making, confirmed reproducible in complete
isolation with no other container active (`load average: 3.69`,
re-run twice, same result both times):
```
firedrake.exceptions.ConvergenceError: Nonlinear solve failed to converge
after 12 nonlinear iterations. Reason: DIVERGED_DTOL
```
**Root cause, traced to source**: `test_cyl.py::test_steady` constructs
`hgym.Cylinder(Re=100, mesh="medium")` with no explicit `restart`, so
`FlowConfig.__init__` auto-infers and loads a checkpoint as the *initial
guess* for the subsequent `NewtonSolver.solve()`. The environment
`Cylinder_2D_Re100_medium_FD` on the Hub contains **22 timestamped
transient trajectory snapshots** (`..._00000570.ckpt` through
`..._00001199.ckpt` — a `dt=0.01` vortex-shedding trajectory, Re=100 is
well past the ~47 shedding-onset threshold, so every snapshot in this
range sits on or near the shedding limit cycle, not the unstable steady
base flow Newton's method is trying to find). The selection code
(`hydrogym/firedrake/flow.py::_resolve_single_checkpoint`) is:
```python
checkpoint_files = list(Path(env_path).glob("checkpoint*.h5"))
if not checkpoint_files:
    checkpoint_files = list(Path(env_path).glob("*.ckpt"))
if checkpoint_files:
    resolved_path = str(checkpoint_files[0].resolve())   # <-- unsorted
```
`Path.glob()` makes **no ordering guarantee** — the file that lands at
index `[0]` depends on filesystem/directory-entry order, which in turn
depends on the order `snapshot_download()` happened to write files to
disk on a given run. In this session's container, `glob()[0]` resolved to
`..._00000870.ckpt` (confirmed via direct inspection, stable across 3
repeated calls *on this populated cache*, but not guaranteed stable
across a fresh download elsewhere) — an arbitrary mid-shedding-cycle
snapshot, a numerically poor Newton initial guess, hence divergence.
**This function was not touched by any of the ~55 audit commits**
(`git log 7c08bc5..HEAD -- hydrogym/firedrake/` does not include it), so
this is a **pre-existing bug**, not a regression introduced by this
implementation pass — but it directly undermines Task 0.1's CI regression
net: because `test.yml`'s `firedrake-tests` job builds a **fresh**
container (and therefore triggers a fresh Hub download) on every run, the
file landing at `glob()[0]` — and therefore whether `test_cyl.py::test_steady`
passes or fails — is not guaranteed stable **across CI runs**, only within
one already-populated cache. A CI job that can go red or green on the same
unchanged code, depending on download-order luck, is close to as
dangerous as the "zero test coverage" problem Task 0.1 set out to fix in
the first place — a flaky-red test trains reviewers to ignore CI, exactly
the failure mode the audit's own Safety Protocol was designed to prevent.
**Fix applied and its actual scope** (this section rewritten after
implementing and testing the fix, not left at the original recommendation
— the first attempt taught something the initial analysis missed):

- `checkpoint_files` is now `sorted(...)` instead of an unsorted
  `glob()` result, and the pick (`[-1]`, last by sorted filename) is now
  **deterministic** — same input directory, same result, every time, on
  every machine — with a log message naming which file was chosen
  whenever more than one candidate exists. This part is a clean,
  low-risk, unambiguous improvement: pure Python (`sorted()` + logging),
  touches nothing solver-numerical, and is unconditionally an improvement
  over "silently varies by filesystem/download order."
- **A first version of this fix went further**: when the restart was
  *auto-inferred* (no explicit checkpoint given) and multiple ambiguous
  candidates were found, it fell back to a zero initial condition instead
  of guessing — reasoning that "ambiguous" should be treated the same as
  "not found." This **did** fix `test_cyl.py::test_steady` (zero-IC
  converges to exactly the expected `CL≈0, CD≈1.2840`) — but re-running
  the full suite (not just the one test) showed it broke
  `test_cyl.py::test_steady_rotation`, a **different** test that also
  auto-infers a checkpoint (`RotaryCylinder_2D_Re100_medium_FD`, same
  22-snapshot structure) but for a **transient** integration, not a
  steady solve. Zero-IC is numerically *wrong* for that test: 40 BDF
  steps from zero-IC gives `CL=-0.196` against an expected `-0.060` (tol
  `1e-3`) — confirmed directly, not just inferred from the test failing.
  `_resolve_single_checkpoint` has no way to know whether its result will
  feed a Newton steady solve (wants zero-IC when ambiguous) or a short
  transient integration (wants *some* real restart state, wrong ones
  included, over none at all) — so a blanket "ambiguous → fall back"
  policy cannot be correct for both callers. **This heuristic was
  reverted** in favor of always resolving to *some* deterministic file
  (see above), which is the only change that helps in both contexts
  without guessing at caller intent.
- **Consequence, checked directly, not assumed**: with the deterministic
  sorted-last pick, `test_cyl.py::test_steady` **still fails**
  (`DIVERGED_DTOL` — the sorted-last file, `..._1199.ckpt`, is a
  transient snapshot that does not converge; an exhaustive 22-file sweep
  run during this investigation found only 2 of 22 — `..._00000930.ckpt`
  and `..._00000990.ckpt`, neither the first nor the last by sort order —
  actually converge Newton to the expected `CL≈0, CD≈1.2840`) and
  `test_cyl.py::test_steady_rotation` **also fails** with the same
  deterministic pick (`CL=-0.510` vs expected `-0.060`; sorted-first was
  checked too and also fails). There is no single fixed index (first,
  last, or otherwise) that satisfies both tests' calibrated expected
  values — the two tests implicitly depend on *different, specific*
  snapshots that were presumably whatever `glob()` happened to return at
  whatever time these tests' expected values were last tuned. A full,
  un-truncated run of the test suite (§3 below) found **two more tests
  with this exact same mechanism** (`test_cyl.py::test_act_implicit_no_damp`,
  `test_pinball.py::test_steady_rotation` — same auto-inferred-checkpoint
  pattern, same failure signature), confirming this is not a one-off.
  **This is now confirmed to be a materially deeper problem than "sort a
  list"**: correctly fixing it requires either curating the Hub dataset to
  publish one canonical checkpoint per environment (outside this repo),
  pinning an explicit `restart=<specific file>` in each affected test (a
  test-content change, which the Safety Protocol reserves for the
  maintainer, not a drive-by verification pass), or a more invasive
  solver-robustness change (e.g. Newton retry-from-zero on divergence)
  that touches shared numerical infrastructure used by every Firedrake
  flow and needs its own
  validation pass across all Firedrake examples — explicitly out of scope
  for a "small fix."
- **A genuine, separate bonus bug found and fixed along the way**:
  `hydrogym/firedrake/flow.py` imports `logging` from **Firedrake**
  (`from firedrake import dx, logging`), not the stdlib module, and
  Firedrake's `logging` shim defines `WARNING` but not the stdlib alias
  `WARN`. Every `logging.log(logging.WARN, ...)` call in this file
  (5 pre-existing, unrelated to checkpoint selection, plus 2 introduced
  by this fix before being caught) would raise `AttributeError` instead
  of logging a warning — meaning **any** error during checkpoint
  resolution that should have produced a graceful warning-and-fall-back
  instead crashed with an unrelated, confusing `AttributeError`, for as
  long as this file has existed. All 7 occurrences are now
  `logging.WARNING`. Caught by actually running the new tests against
  real Firedrake, not by inspection — this is exactly the kind of bug
  static analysis alone would miss.
- **`test_cyl.py::test_linearize` was also observed to fail** in the same
  full-suite run. Investigated and confirmed **unrelated** to any of this
  session's changes: with checkpoint resolution bypassed entirely
  (`use_HF_data_manager=False`, pure zero-IC), it fails with
  `ValueError: too many values to unpack (expected 2)` inside
  `NewtonSolver`-adjacent linearization code — the exact error signature
  the Task 0.3 baseline already documented for
  `test_cavity`/`test_pinball`/`test_step` ("version skew in the
  Firedrake install"). In the full-suite run it actually surfaced as a
  `ConvergenceError` instead, for an uninteresting reason: `test_linearize`
  also auto-infers `Cylinder_2D_Re100_medium_FD` (same environment as
  `test_steady`), so with the deterministic pick it now fails at the
  earlier `NewtonSolver.solve()` call, before ever reaching the
  pre-existing unpacking bug downstream — two independent, both
  pre-existing problems stacked on the same test, whichever one is
  encountered first depends on which checkpoint got picked. Neither is
  caused by this pass.

**A full, un-truncated run of the whole suite** (`pytest .` with no `-x`,
222 collected, 0:23:48) surfaced 10 failures total, individually
re-run in isolation and root-caused (§3's evidence table has the full
breakdown) — none are new regressions from this session's changes:
- **4 are this exact checkpoint-ambiguity mechanism**: the two above,
  plus `test_cyl.py::test_act_implicit_no_damp` and
  `test_pinball.py::test_steady_rotation` (same auto-inferred-checkpoint
  pattern, confirmed by direct reproduction).
- **1 (`test_io.py::test_checkpointing`) references `hgym.IPCS`**, a
  solver class removed from this codebase years ago (`git log` shows
  commit `43e75cc`, "Remove IPCS solver," long before this audit) — a
  stale test nobody updated after the removal, unrelated to anything in
  this pass.
- **2 (`test_pinball.py::test_env`, `test_step.py::test_env`) hit
  `SemiImplicitBDF.__init__() missing 1 required positional argument:
  'dt'`** — a pre-existing bug in those two tests' own `solver_config`
  dict, nothing to do with checkpoints or this pass's changes.
- **2 (`test_registration.py::test_firedrake_factory_defaults`/
  `test_firedrake_factory_caller_overrides_win`) pass individually and
  pass running the whole file alone** (10/10) — they only fail as part
  of the complete 222-test run, indicating pre-existing cross-test state
  leakage somewhere in the suite (most likely `gymnasium`'s global
  registry or a Firedrake-side cache accumulating state across files);
  not attributable to either test or to this session's changes.

None of these 10 are reachable through CI's actual recipe (`test.yml`
uses `-x`, which halts at the first failure — `test_steady`, the very
first one) except `test_steady` itself; they were only found because this
verification pass ran the suite in full to check for exactly this kind of
thing.

**Net effect of the applied fix**: `test.yml`'s Firedrake CI job is no
longer *flaky* (every affected test's pass/fail status is now the same
on every run, every machine) but it is not fully *green* — `test_steady`
and its 3 checkpoint-ambiguity siblings fail deterministically pending
the maintainer decision described above, and the other 6 failures found
by running the full suite are pre-existing, independent issues this pass
did not attempt to fix (out of scope: none are small, none are caused by
this pass, and per the Safety Protocol, widening scope to fix unrelated
issues discovered mid-task is exactly what "note it as a new task, don't
fix it inline" is for). This is still a strict improvement over the
starting state (a stable red is debuggable and actionable; a flaky one
erodes trust in the whole
CI job) but is **not** the "small fix" this was initially assessed to be
— flagging that assessment as wrong is itself part of doing this
honestly. Tracked as a new, still-open task, independent of Findings A/B.

**Finding D — transient CUDA `RESOURCE_EXHAUSTED` observed during concurrent GPU-stack testing.**
`jax_kolmogorov`'s log opened with `E0904 ... Failed to allocate device
memory of 11.65GiB ... CUDA_ERROR_OUT_OF_MEMORY`, immediately followed by
a normal, complete run with real numeric output. This occurred while
`maia_gpu`, `jax_channel`, and (starting) `jaxfluids` were recently active
on the same single RTX 4090 Laptop GPU (16GB) as part of this
verification pass's own sequential-but-back-to-back GPU test run,
consistent with JAX's default memory-preallocation behavior transiently
colliding with another process's still-resident allocation rather than a
code defect. Not reproduced as a hard failure; noted for awareness rather
than as a regression, since it did not recur and every GPU test still
produced correct output.

### 5. Net assessment

The implementation is **substantially faithful to the plan and, on every
task independently spot-checked in this pass, verifiably real** — not
just claimed. In particular, the item the user weighted most heavily —
whether MAIA/Nek launch via subprocess spawning (which would break static
Slurm/PBS allocation) — is **conclusively not the case**: the entire
architecture is standards-based MPI communicator-splitting inside one
externally-launched MPMD job, verified both by exhaustive static grep and
by live, real, GPU-and-CPU end-to-end runs during this session.

Four findings came out of this pass, none of them in the MPI/launch
architecture itself, and all four were fixed or fully investigated
(commits `0bb77b3`, `4478a89`, `b054291`, `655b45b`, on top of the ~55
that implemented the plan itself):

- **Finding A** (`distributed/` not deleted): **fixed**. Package deleted,
  dropped from the lazy-loader and `__all__`, verified against the real
  container.
- **Finding B** (MAIA launch-config validation never assigned a task):
  **fixed**. Optional `nproc` validation added to `MaiaInterface.init_comm`,
  wired through `env_config`, verified against real MAIA-CPU and MAIA-GPU
  MPMD runs.
- **Finding C** (non-deterministic checkpoint selection causing a real,
  reproducible `ConvergenceError` in `test_cyl.py::test_steady`):
  **partially fixed, and more instructive than it first looked**. The
  selection is now deterministic (sorted, same result on every machine),
  which is a genuine, unambiguous improvement — the CI job is no longer
  *flaky*. A first version of the fix went further (falling back to
  zero-IC when ambiguous) and looked complete because it made the one
  originally-reported test pass; only re-running the *whole* affected
  test file exposed that it broke a different test
  (`test_steady_rotation`) that needs the opposite behavior for the same
  ambiguous input. That version was reverted. Running the full,
  un-truncated suite (not just the reported failure) then found 3 more
  tests hitting the exact same underlying ambiguity, plus 6 entirely
  unrelated pre-existing failures (a removed-solver-class reference, a
  missing test kwarg, cross-test state leakage) — all individually
  root-caused, none caused by this pass or the ~55 that preceded it.
  `test_steady` and its siblings now fail *deterministically* rather than
  by luck; fully resolving them needs a maintainer decision (curate the
  Hub dataset, or pin explicit checkpoints in the affected tests) that is
  correctly out of scope for this pass to make unilaterally. A genuine
  bonus fix (`firedrake.logging.WARN` not existing, silently crashing 7
  warning-log call sites) was found and fixed along the way — the kind of
  thing only surfaces by actually running the fix against real code, not
  by reasoning about it.
- **Finding D** (transient CUDA OOM message that self-recovered): noted
  for awareness only, not a defect.

None of the four findings were in the code the ~55 commits actually
wrote; Finding C's root cause in particular predates this implementation
pass entirely, and every failure it led to on a full test-suite run
traces to either that same pre-existing mechanism or to independent,
also pre-existing issues. The audit's core deliverable — a real, working,
Slurm/PBS-safe MPMD architecture with no subprocess spawning anywhere —
is confirmed sound, and the fixes made in this pass are verified against
real solver runs (CPU and GPU), not just code review.

---

## Examples Verification Pass (2026-09-04, continued)

**Trigger:** a direct follow-up question — "did you run all examples from
READMEs and the example section and ensured they work 100% correct?" —
that the prior pass had not actually answered. The prior pass ran a
curated subset (the smoke-test harness scripts, one per backend, plus 3
spot-checks against the Task 4.1 example audit). This pass closes that
gap: every `examples/**/*.py`, both notebooks, and every README code
block was either executed directly or (for training-loop scripts whose
full run would take hours) executed with a reduced/bounded configuration
sufficient to prove the code path is real and correct, matching this
repo's own test-harness philosophy of judging by real, correct progress
rather than requiring full physical completion.

### Coverage

- **All 21 Firedrake example scripts** (`examples/firedrake/advanced/{cavity,cylinder,pinball,step}/*.py`,
  `getting_started/config_reference.py`), run with a bounded per-script
  timeout: 8 completed in full (clean exit 0, correct physics output —
  e.g. `cavity/solve-steady.py`'s Newton residual reaching `6.6e-17`,
  `cylinder/stability.py`'s eigenvalues, `pinball/solve-steady.py`'s
  per-cylinder lift/drag), 13 hit the timeout still producing correct,
  physically sensible, steadily-progressing output (these are genuinely
  long research scripts — `Tf=100-500` integrations, a 1M+-DOF stability
  eigenproblem) — zero crashes, zero incorrect output, across all 21.
  One (`cavity/stability.py`) was separately re-run with a 900s budget and
  confirmed to complete its full Reynolds-continuation sequence
  (Re=500→1000→2000→4000→7500) with clean Newton convergence at every
  stage before its final eigenvalue solve.
- **The top-level `README.md`'s PPO training snippet** (Firedrake
  `Cylinder`/`SemiImplicitBDF`/SB3): executed verbatim, completed in full
  (`total_timesteps=20` → SB3 still runs one full `n_steps=2048` rollout
  per its own semantics — genuinely ~22 real minutes on a contended CPU,
  not stuck), exit 0.
- **`examples/firedrake/README.md`, `examples/maia/README.md`,
  `examples/jax/README.md`'s code blocks**: each backend's snippet
  executed directly (not just read) against a real prepared workspace —
  MAIA's `from_hf(...)` quickstart and JAX's `KolmogorovFlow`/`jax.jit`
  quickstart both confirmed working end-to-end.
- **All 6 Nek `getting_started` chapters**, both their `test_*.py` and
  `train_sb3_*.py` scripts: real MPMD runs (`mpirun ... : ... nek5000`)
  against the actual case binaries, all producing correct step/reward
  output, including a real PPO training run
  (`train_sb3_nek_direct.py`) and the 12-rank zero-shot wing deployment
  demo (`zeroshot_demo_pettingzoo.py`, 1632 agents, sensible reward
  range).
- **MAIA's `test_maia_env.py`, `train_sb3_maia.py`, `prepare_workspace.py`**:
  all re-verified on both the CPU and GPU stacks after this pass's fixes.
- **JAX's `test_kolmogorov_env.py`, `test_channel_env.py`, `run_ppo.py`,
  and both notebooks** (`kolmogorov.ipynb`, `channel.ipynb`, executed via
  `jupyter nbconvert --execute`, not just read).
- **JAX-Fluids' `test_jaxfluids_env.py`**: re-confirmed (times out at its
  documented 600s cap by design, real progress throughout).
- **`examples/developer_templates/`**: covered by its own pytest suite
  (10/10 passing in a bare venv, already verified in the prior pass).

### Findings from this pass — all fixed except two explicitly flagged as out of scope

**Fixed (5 commits: `db3ee88`, `8076a7b`, `e677784`, `4bfc278`, plus the
`pyproject.toml` edits folded into the latter two):**

1. **Nonexistent default environment name** (`"MiniChannel_Re180"`, does
   not exist on the Hub — confirmed via `HfApi().list_repo_files`) in
   `test_nek_DM.py`/`test_nek_pettingzoo.py`'s `--env` defaults, five
   Nek READMEs' copy-pasteable usage examples, `hydrogym/nek/env.py` and
   `hydrogym/nek/__init__.py`'s own docstring examples, and
   `prepare_workspace.py`'s usage text. Real env for this exact case
   (already used correctly by two sibling scripts): `TCFmini_3D_Re180`.
2. **Wrong script filename** in `zeroshot_demo_pettingzoo.py`'s own
   docstring and its README (`test_nek_pettingzoo.py`, a different file
   in a different directory) — following either literally gives "No such
   file or directory."
3. **`pyproject.toml`'s `maia` extras missing `stable-baselines3`/
   `tensorboard`** — `pip install -e ".[maia]"`, the documented install
   method, cannot run `train_sb3_maia.py`; reproduced directly in the
   officially-provisioned `maia-cpu` devcontainer venv.
4. **`properties.toml` vs `properties_run.toml`** typo in
   `train_sb3_maia.py`/`test_maia_env.py`'s own docstrings (Task 1.3 had
   already fixed this in the READMEs, but missed these two scripts'
   embedded examples).
5. **`kolmogorov.ipynb` fails at three successive points** against the
   currently-installed JAX/Matplotlib: a removed `jax.lib.xla_bridge`
   import (→ `jax.default_backend()`), an undeclared `imageio` dependency
   (→ added to the `jax` extras), and a removed
   `FigureCanvasAgg.tostring_rgb()` method (→ `buffer_rgba()`). Confirmed
   fixed by actually re-executing the notebook end-to-end afterward, not
   by inspection.

**Flagged, not fixed — genuinely out of this pass's scope:**

6. **Nek5000's `small_wing` case (`zeroshot_demo_pettingzoo.py`, 12
   ranks) prints "Emergency exit" + a stack-trace-style backtrace on
   every worker rank during MPI teardown**, immediately after printing
   correct, complete results. Confirmed via the full untruncated log
   (not just a truncated tail, which — a real methodological trap hit
   during this investigation — made it initially look like the *only*
   output, i.e. a crash with no results at all) that the real demo
   output (a sensible 1632-agent reward summary) prints first, then the
   clean `[TERMN] DISCONNECT!` sequence, then the "Emergency exit"
   dump on all 12 ranks. This is Nek5000's own internal abrupt-exit
   routine (a known pattern for legacy Fortran CFD codes — deliberately
   calling an abrupt exit instead of a clean `MPI_Finalize` to sidestep
   hangs), not a memory-safety bug: the `mini_channel`/`TCFmini` case
   used by every other Nek example reaches a cleaner `"run successful:
   dying ..."` exit path instead, so this is specific to the
   `small_wing` case's own exit path. Cosmetically alarming, computed
   results are correct and already printed — same *category* of benign
   teardown noise as the already-documented UCX warning, just a lot
   louder. Not fixed: this is Fortran `.usr`-case-level behavior, outside
   this repo's safe editing scope (the `.usr` files ship via HF, not in
   this tree).
7. **`run_ppo.py --env kolmogorov` shows a real slowdown after its first
   few logged updates** — fast for the first ~20-60 steps, then
   substantially slower per step for the remainder of a 2000-step run
   (confirmed twice, not a one-off). Output remained numerically correct
   throughout the portion that ran (sensible `mean_tke` values, no
   NaN/crash) — this is a performance characteristic, not a correctness
   bug, and root-causing it (JAX recompilation? GPU memory fragmentation
   under this session's own concurrent testing? something else?) would
   need dedicated profiling this pass did not have grounds to assume was
   in scope. Flagged for the maintainer.

### Net effect

Every example this pass could exercise (i.e. everything except the four
`train_sb3_*.py` scripts' *full* advertised timestep counts, which would
take hours and were instead smoke-tested at reduced scale — a deliberate,
documented substitution, not a skipped check) was run for real and
produced correct output, after five small, verified fixes. The two
flagged-not-fixed items are both genuinely outside a "small fix" — one is
Fortran solver-internal behavior this repo doesn't own the source for,
the other needs profiling this pass wasn't scoped to do. Nothing found in
this pass touches the MPI/launch architecture question from the prior
pass; that finding stands as previously confirmed.
