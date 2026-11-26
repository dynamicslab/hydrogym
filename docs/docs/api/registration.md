---
sidebar_label: registration
title: hydrogym.registration
---

gymnasium registration for HydroGym environments.

Importing this module registers a canonical set of environment IDs so
`gym.make(...)` works uniformly across the gymnasium-API backends.

gymnasium requires IDs of the form `[namespace/]name-vN` (one namespace
segment), so each backend gets its own namespace: `hydrogym` (Firedrake,
the historical IDs), `hydrogym-maia`, `hydrogym-nek`, `hydrogym-jaxfluids`.

**Examples**:

  >>> import hydrogym.registration  # registers the IDs below (once)
  >>> import gymnasium as gym
  >>> env = gym.make('hydrogym/Cylinder-v0')
  >>> env = gym.make('hydrogym-maia/Cylinder_2D_Re200-v0', nproc=4)
  >>> env = gym.make('hydrogym-nek/TCFmini_3D_Re180-v0', nproc=10)
  >>> env = gym.make('hydrogym-jaxfluids/Nozzle2D-v0', env_config=\{\})
  
  Design notes:
  - Registration is deliberately lazy: every entry point imports its backend
  only when the environment is actually constructed, so
  `import hydrogym.registration` never pays for mpi4py / jax / jaxfluids
  imports and never initializes MPI.
  - Backend-native config schemas are preserved: `gym.make` passes its
  keyword arguments straight through to each backend's own factory
  (`from_hf` for MAIA/Nek, `env_config` dicts for Firedrake and
  JAX-Fluids).
  - The JAX backend is intentionally NOT registered: it implements the
  functional/gymnax contract (`gymnax.environment.Environment`), whose
  `reset`/`step` signatures differ from `gymnasium.Env`. Wrapping it
  in `gym.make` would present a misleading API; use
  `hydrogym.jax.envs.*` directly (see docs/docs/developers/adding-a-solver.md,
  Pattern 3).
  - The MAIA/NEK/JAX-Fluids IDs below are representative entry points, not an
  exhaustive catalog: the HF Hub hosts many more environments per backend,
  and each backend's own `from_hf` accepts any registered environment
  name (see docs/docs/developers/adding-an-environment.md).

#### register\_all

```python
def register_all() -> None
```

Register every built-in environment ID. Idempotent; called on module
import. New IDs should follow the same lazy-factory pattern.

