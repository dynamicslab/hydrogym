# Coding Agent Prompt — HydroGym Engineering Audit v2 Implementation

You are implementing the changes specified in
**`HYDROGYM_ENGINEERING_AUDIT_v2.md`** (repo root of this branch). That
document is your **complete and authoritative instruction**: it contains the
findings, the target architecture, the per-task list (Objective / Files /
Required change / Tests / Acceptance criteria / Dependencies / Risk), the
Safety Protocol, and the Definition of Done. Read it in full before touching
any code, and re-consult the relevant task entry before starting each task —
do not work from memory or from this prompt's summary.

## Repository & branch

- Repo: `/home/christian/maia_solver/hydrogym` (fork/working tree of
  `dynamicslab/hydrogym`).
- Branch: **`hydrogym-audit-v2`** (already created off `devcontainer-update`
  tip `2af9303`). Do all work on this branch; do not commit to `main` or
  `devcontainer-update`.
- The working tree has pre-existing, uncommitted local modifications under
  `third_party/` (Nek5000/m-AIA submodules). These are **not yours** and are
  unrelated to this effort: do not stage, commit, revert, or "clean up" them.
  Only commit files you intentionally changed for a task.
- Audit baseline: the doc was written against `main` tip `bf2c2dd`. Verify
  file states against the current checkout before acting (Safety Protocol
  rule 7).

## Execution order (from the doc, §Recommended Implementation Order)

1. **Phase 0 — Safety net** (CI runs `test/`; `test_core.py`). **Hard
   blocking gate**: nothing in Phases 1+ starts until Phase 0 is merged and
   green. If it cannot be completed, stop and report — do not proceed
   without a regression net.
2. **Phase 1** — low-risk cleanup (dead code, doc/CI config fixes).
3. **Phase 2** — kwargs/config propagation fixes, one task = one PR, ordered
   Firedrake → JAX/JAX-Fluids → Nek → MAIA.
4. **Phase 3** — shared-infrastructure extraction (`HFEnvConfigMixin`,
   `ExternalProcessEnvMixin`), strictly JAX-Fluids → JAX → Nek → MAIA.
5. **Phase 4** — docs, examples, developer guides.
6. **Phase 5** — the single behavior-changing fix
   (`terminated`/`truncated`), isolated, changelog-flagged, last, never
   bundled with anything else.
7. **Phase 6** — optional/deferred items; pick up only when Phases 0–5 are
   done and only as separate tasks.
8. **Phase 7** — docstring coverage rollout, last of all.

## Nature of the work — interface, not functionality

This effort is a **pure interface/refactoring change: functionality must be
bit-for-bit preserved.** Every task either fixes plumbing (kwargs reaching
the place they were already meant to reach), extracts duplicated code into a
shared mixin, or adds documentation/tests — no solver's numerics, stepping
semantics, or outputs change. Where Phase 2 fixes a kwargs defect, the
"new" behavior is the behavior the code already claimed to have; flag
explicitly in the PR if you believe a fix changes observable behavior, and
treat that as a blocker to escalate, not a detail to note.

Consequence: **every change must produce the same results as before.** Any
test that starts differing in solver output is a regression to stop on, not
a new baseline to adopt.

## Continuous testing — non-negotiable

Progress is tested continuously, per backend, in the **local devcontainers**
defined in `.devcontainer/`:

| Backend | Validation environment | Devcontainer config |
|---|---|---|
| MAIA | **GPU** container | `maia-gpu-test.devcontainer.json` (or `full-gpu-stack`) |
| JAX | **GPU** container | `full-gpu-stack.devcontainer.json` (maia-gpu + jax-gpu) |
| JAX-Fluids | **GPU** container | `full-gpu-stack.devcontainer.json` (`jax-fluids` feature) |
| Firedrake | **CPU** container | `firedrake-test.devcontainer.json` |
| Nek5000 | **CPU** container | `nek5000-test.devcontainer.json` |

Rules that follow from this mapping:

- Run the relevant backend's tests/tests-tier **before and after every task
  that touches that backend**, and re-run the untouched backends' tiers
  before declaring a phase done (shared code — e.g. `core.py`, the new
  mixins, `data_manager.py` — counts as touching *every* backend).
- **Record an explicit pre-change baseline first** (per backend: test output
  +, for MAIA and Nek, a reference run's numerical output), then compare
  every task's post-change run against that baseline. Identical results is
  the pass condition — not "tests pass."
- MAIA and Nek additionally require a real end-to-end dev-container run
  (per the doc's Safety Protocol rule 4) as a merge gate, every time.
- If the GPU or CPU container for a backend can't be run at a given moment,
  that backend's work pauses until it can — never merge untested-in-env
  changes with the intention of "testing later."

## Commit discipline

- **Commit progress regularly to `hydrogym-audit-v2`** — small, task-scoped
  commits (one Task-List item per commit, per Safety Protocol rule 2), each
  carrying its own validation evidence in the message body (which tests ran,
  in which container, against which baseline, pass/fail).
- Never leave validated-but-uncommitted work sitting in the tree across
  tasks, and never let a commit land untested with the plan to validate it
  afterward.
- Do not push anywhere unless explicitly asked; local commits on the branch
  are the deliverable cadence.

## Non-negotiable rules (summary — full text in the doc's
## "Safety Protocol — Hard Gates" section; it governs every task)

- **One task, one commit/PR, one revert point.** Never batch two Task-List
  items, even in the same file.
- **Show passing-run evidence, not claims.** Every non-docs task ships with
  actual test output in its description.
- **MAIA/Nek tasks require a real dev-container run** (build + integration
  check) as a merge gate — a code review is never sufficient for those
  backends.
- **Never widen scope mid-task.** Adjacent problems get new Task-List
  entries, not inline fixes.
- **Deprecate, don't delete**, except Task 1.1's confirmed-dead code and
  Phase 5's isolated change.
- **Re-verify every audit claim** (greps for "dead"/"unused" code, current
  line numbers, etc.) immediately before acting on it.
- **If acceptance criteria can't be met, stop and report** — don't ship
  with reduced validation and hope.
- If reality disagrees with the doc's risk rating, the *actual* risk wins:
  escalate validation rigor accordingly.

## Environment notes

- All validation containers are already built locally (the `full-gpu-stack`
  image `vsc-hydrogym-…:latest` is verified working with GPU passthrough);
  launch configs live in `.devcontainer/` per the matrix above. The
  `pstlPreset` is pinned to `"ada"` (RTX 40-series) — do not change it.
- `.devcontainer/README.md` documents feature composition
  (`maia-gpu`, `maia-cpu`, `nek5000`, `petsc`, `firedrake`, `python-ml`,
  `hydrogym-gpu`, `jax-fluids`) if a variant needs to be rebuilt.
- `third_party/m-AIA` here points at the private `wipmaiaml` dev tree;
  MAIA-side work may interact with the MAIA workspace conventions in
  `/home/christian/maia_solver/CLAUDE.md` — consult it if a task touches
  MAIA or Nek integration.
- CI workflows live in `.github/workflows/`. Phase 0's work happens there
  (`build.yml` currently runs `poetry build` only and never runs `test/`).

## Deliverables & Definition of Done

Track progress against the doc's "Definition of Done" section. Every merged
task needs: the code change, its tests, and real passing-run evidence, on
this branch, as separate reviewable commits/PRs. When a phase completes,
state so explicitly and list the evidence before moving to the next phase.

If you find a task in the doc that is wrong, obsolete, or unsafe as
specified, do not silently deviate — report the discrepancy with evidence
and get a decision before implementing anything different from the doc.
