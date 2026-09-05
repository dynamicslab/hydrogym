"""gymnasium registration for HydroGym environments (audit Task 6.1).

Importing this module registers a canonical set of environment IDs so
``gym.make(...)`` works uniformly across the gymnasium-API backends:

    import hydrogym.registration  # registers the IDs below (once)
    import gymnasium as gym

    env = gym.make("hydrogym/Cylinder-v0")
    env = gym.make("hydrogym-maia/Cylinder_2D_Re200-v0", nproc=4)
    env = gym.make("hydrogym-nek/TCFmini_3D_Re180-v0", nproc=10)
    env = gym.make("hydrogym-jaxfluids/Nozzle2D-v0", env_config={...})

gymnasium requires IDs of the form ``[namespace/]name-v<version>`` (one
namespace segment), so each backend gets its own namespace: ``hydrogym``
(Firedrake, the historical IDs), ``hydrogym-maia``, ``hydrogym-nek``,
``hydrogym-jaxfluids``.

Design notes (following the audit's Solver Interface Design):
- Registration is deliberately lazy: every entry point imports its backend
  only when the environment is actually constructed, so
  ``import hydrogym.registration`` never pays for mpi4py / jax / jaxfluids
  imports and never initializes MPI.
- Backend-native config schemas are preserved: ``gym.make`` passes its
  keyword arguments straight through to each backend's own factory
  (``from_hf`` for MAIA/Nek, ``env_config`` dicts for Firedrake and
  JAX-Fluids).
- The JAX backend is intentionally NOT registered: it implements the
  functional/gymnax contract (``gymnax.environment.Environment``), whose
  ``reset``/``step`` signatures differ from ``gymnasium.Env``. Wrapping it
  in ``gym.make`` would present a misleading API; use
  ``hydrogym.jax.envs.*`` directly (see docs/docs/developers/adding-a-solver.md,
  Pattern 3).
- The MAIA/NEK/JAX-Fluids IDs below are representative entry points, not an
  exhaustive catalog: the HF Hub hosts many more environments per backend,
  and each backend's own ``from_hf`` accepts any registered environment
  name (see docs/docs/developers/adding-an-environment.md).
"""

import gymnasium as gym

__all__ = ["register_all"]

_REGISTERED = False

# Canonical Firedrake flow/solver pairs (flow from hydrogym.firedrake.envs,
# solver from hydrogym.firedrake.solvers; dt defaults to each flow's own).
_FIREDRAKE_ENVS = {
    "hydrogym/Cylinder-v0": "Cylinder",
    "hydrogym/RotaryCylinder-v0": "RotaryCylinder",
    "hydrogym/Cavity-v0": "Cavity",
    "hydrogym/Pinball-v0": "Pinball",
    "hydrogym/Step-v0": "Step",
}

# Representative MAIA environment names (verified in the from_hf docstring).
# `probe_locations` has no universal default (the sensible probe grid is a
# per-flow-geometry choice -- MaiaFlowEnv raises a clear ConfigError if it's
# omitted, see env_core.py), so only environments with a verified-working
# default grid get one supplied here; the others require the caller to pass
# `probe_locations=` explicitly, same shape of requirement as Nek's `nproc`.
# Cylinder_2D_Re200's default is the wake-sampling grid from
# examples/maia/getting_started/test_maia_env.py's create_probe_locations
# (8 downstream x 5 crossflow, x in [1, 8], y in [-1, 1]) -- run repeatedly
# against the real solver with this exact grid, real sensible reward output.
_MAIA_ENVS = {
    "Cylinder_2D_Re200": {
        "probe_locations": [
            coord
            for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            for y in [-1.0, -0.5, 0.0, 0.5, 1.0]
            for coord in (x, y)
        ],
        "obs_normalization_strategy": "U_inf",
    },
    "RotaryCylinder_2D_Re1000": {},
    "Cavity_2D_Re4140": {},
}  # ns: hydrogym-maia

# Representative NEK environment (the documented smoke-test case; nproc must
# match the MPMD launch, so it is a required make() kwarg here).
_NEK_ENVS = ["TCFmini_3D_Re180"]  # ns: hydrogym-nek

# Representative JAX-Fluids environments (constructed from an env_config
# dict, as in examples/jaxfluids/). Maps gym ID suffix -> (env_class_name,
# default HF environment_name). There is no plain "Nozzle2D"/"Nozzle3D" HF
# environment -- only resolution-suffixed variants (confirmed via
# HfApi().list_repo_files: Nozzle2D_coarse/_fine, Nozzle3D_coarse/_fine) --
# so the gym ID's default must be one specific published environment;
# "_coarse" matches examples/jaxfluids/test_jaxfluids_env.py's own default.
_JAXFLUIDS_ENVS = {
    "Nozzle2D": ("Nozzle2D", "Nozzle2D_coarse"),
    "Nozzle3D": ("Nozzle3D", "Nozzle3D_coarse"),
}  # ns: hydrogym-jaxfluids


def _firedrake_make(flow_name: str):
    def _make(env_config: dict = None):
        import hydrogym as hgym
        from hydrogym.core import FlowEnv

        env_config = dict(env_config or {})
        env_config.setdefault("flow", getattr(hgym.firedrake, flow_name))
        env_config.setdefault("solver", hgym.firedrake.SemiImplicitBDF)
        env_config.setdefault("flow_config", {})
        env_config.setdefault("solver_config", {})
        return FlowEnv(env_config)

    return _make


def _maia_make(environment_name: str, defaults: dict):
    def _make(**kwargs):
        # Import via the package (attribute access), not
        # `from hydrogym.maia.env_core import from_hf`: hydrogym.maia lazily
        # populates its MPI-dependent names -- including the env-class
        # registration side effect of importing hydrogym.maia.envs.* -- only
        # through its own __getattr__, which a direct submodule import
        # bypasses entirely. Importing env_core directly leaves
        # _ENVIRONMENT_REGISTRY empty, so from_hf() can't recognize any
        # environment type ("Cylinder", "Pinball", ...) even though the
        # class exists -- confirmed live: raises immediately outside MPMD,
        # or strands the paired MAIA rank waiting forever for a handshake
        # the crashed Python side never sends, inside MPMD.
        import hydrogym.maia as maia

        merged = {**defaults, **kwargs}
        return maia.from_hf(environment_name, **merged)

    return _make


def _nek_make(environment_name: str):
    def _make(**kwargs):
        from hydrogym.nek.env import NekEnv

        if "nproc" not in kwargs:
            raise TypeError(
                f"gym.make('hydrogym/nek/{environment_name}') requires nproc=... "
                "(it must match the MPMD launch's worker count)."
            )
        return NekEnv.from_hf(environment_name, **kwargs)

    return _make


def _jaxfluids_make(env_class_name: str, default_environment_name: str):
    def _make(env_config: dict = None, **kwargs):
        from hydrogym.jaxfluids import envs as jxf_envs

        env_cls = getattr(jxf_envs, env_class_name)
        merged = {**(env_config or {}), **kwargs}
        merged.setdefault("environment_name", default_environment_name)
        return env_cls(env_config=merged)

    return _make


def register_all() -> None:
    """Register every built-in environment ID. Idempotent; called on module
    import. New IDs should follow the same lazy-factory pattern."""
    global _REGISTERED
    if _REGISTERED:
        return

    for env_id, flow_name in _FIREDRAKE_ENVS.items():
        gym.register(id=env_id, entry_point=_firedrake_make(flow_name))

    for env_name, defaults in _MAIA_ENVS.items():
        gym.register(id=f"hydrogym-maia/{env_name}-v0", entry_point=_maia_make(env_name, defaults))

    for env_name in _NEK_ENVS:
        gym.register(id=f"hydrogym-nek/{env_name}-v0", entry_point=_nek_make(env_name))

    for env_name, (env_class_name, default_environment_name) in _JAXFLUIDS_ENVS.items():
        gym.register(
            id=f"hydrogym-jaxfluids/{env_name}-v0",
            entry_point=_jaxfluids_make(env_class_name, default_environment_name),
        )

    _REGISTERED = True


register_all()
