# Minimal in-process solver (reference skeleton)

A no-real-physics implementation of HydroGym's **in-process** backend
pattern (~100 lines): a `core.PDEBase` subclass + a
`core.TransientSolver` subclass, reused through `core.FlowEnv` unchanged.
Runs in plain Python — no Firedrake, no MPI, no GPU.

```bash
pytest test_trivial_backend.py
```

Read `trivial_backend.py` top to bottom, then see
`docs/docs/developers/adding-a-solver.md` (Pattern 1) for the full
walkthrough. `firedrake/` is the production example of this pattern.
