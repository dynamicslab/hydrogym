---
sidebar_label: registration
title: hydrogym.registration
---

gymnasium registration for HydroGym environments (audit Task 6.1).

Importing this module registers a canonical set of environment IDs so
`gym.make(...)` works uniformly across the gymnasium-API backends.

gymnasium requires IDs of the form `[namespace/]name-vN` (one namespace
segment), so each backend gets its own namespace: `hydrogym` (Firedrake,
the historical IDs), `hydrogym-maia`, `hydrogym-nek`, `hydrogym-jaxfluids`.

**Examples**:

  &gt;&gt;&gt; import hydrogym.registration  # registers the IDs below (once)
  &gt;&gt;&gt; import gymnasium as gym
  &gt;&gt;&gt; env = gym.make(&#x27;hydrogym/Cylinder-v0&#x27;)
  &gt;&gt;&gt; env = gym.make(&#x27;hydrogym-maia/Cylinder_2D_Re200-v0&#x27;, nproc=4)
  &gt;&gt;&gt; env = gym.make(&#x27;hydrogym-nek/TCFmini_3D_Re180-v0&#x27;, nproc=10)
  &gt;&gt;&gt; env = gym.make(&#x27;hydrogym-jaxfluids/Nozzle2D-v0&#x27;, env_config=`{}`)
  
  Design notes (following the audit&#x27;s Solver Interface Design):
  - Registration is deliberately lazy: every entry point imports its backend
  only when the environment is actually constructed, so
  `import hydrogym.registration` never pays for mpi4py / jax / jaxfluids
  imports and never initializes MPI.
  - Backend-native config schemas are preserved: `gym.make` passes its
  keyword arguments straight through to each backend&#x27;s own factory
  (`from_hf` for MAIA/Nek, `env_config` dicts for Firedrake and
  JAX-Fluids).
  - The JAX backend is intentionally NOT registered: it implements the
  functional/gymnax contract (`[namespace/]name-vN`0), whose
  `[namespace/]name-vN`1/`[namespace/]name-vN`2 signatures differ from `[namespace/]name-vN`3. Wrapping it
  in `gym.make` would present a misleading API; use
  `[namespace/]name-vN`5 directly (see docs/docs/developers/adding-a-solver.md,
  Pattern 3).
  - The MAIA/NEK/JAX-Fluids IDs below are representative entry points, not an
  exhaustive catalog: the HF Hub hosts many more environments per backend,
  and each backend&#x27;s own `from_hf` accepts any registered environment
  name (see docs/docs/developers/adding-an-environment.md).

#### register\_all

```python
def register_all() -> None
```

Register every built-in environment ID. Idempotent; called on module
import. New IDs should follow the same lazy-factory pattern.

