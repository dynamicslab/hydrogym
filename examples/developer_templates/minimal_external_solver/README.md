# Minimal external-process solver (reference skeleton)

A no-real-physics implementation of HydroGym's **external-process** backend
pattern: a `gym.Env` subclass mixed with `ExternalProcessEnvMixin`, with
stubbed wire methods where a real backend talks to its solver over MPMD
MPI. Imports and tests run in plain Python — no mpi4py, no solver binary,
no GPU.

```bash
pytest test_solver_env.py
```

In production the solver runs as a second MPI application in the same MPMD
world:

```bash
mpirun -np 1 python controller.py ... : -np N ./your_solver
```

Read `solver_env.py` top to bottom, then see
`docs/docs/developers/adding-a-solver.md` (Pattern 2) for the full
walkthrough. `hydrogym/maia/` and `hydrogym/nek/` are the production
examples (real wire protocols, `close()` termination contract, workspace
prep helpers).
