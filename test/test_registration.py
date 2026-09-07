"""Tests for hydrogym.registration (audit Task 6.1).

Plain-CI tier: importing hydrogym.registration must cost nothing but
gymnasium (no firedrake / mpi4py / jax / jaxfluids / MPI init), and each
registered entry point must lazily import its backend only at construction.
Backend imports are faked via sys.modules (both the leaf module and its
parent package, so heavy backend __init__s never execute) so no solver
stack is needed.
"""

import subprocess
import sys
import types

import gymnasium as gym
import numpy as np
import pytest

import hydrogym.registration  # noqa: F401  (import performs the registration)

EXPECTED_IDS = {
    "hydrogym/Cylinder-v0",
    "hydrogym/RotaryCylinder-v0",
    "hydrogym/Cavity-v0",
    "hydrogym/Pinball-v0",
    "hydrogym/Step-v0",
    "hydrogym-maia/Cylinder_2D_Re200-v0",
    "hydrogym-maia/RotaryCylinder_2D_Re1000-v0",
    "hydrogym-maia/Cavity_2D_Re4140-v0",
    "hydrogym-nek/TCFmini_3D_Re180-v0",
    "hydrogym-jaxfluids/Nozzle2D-v0",
    "hydrogym-jaxfluids/Nozzle3D-v0",
}


def test_all_ids_registered_and_callable():
    registry = gym.registry
    for env_id in EXPECTED_IDS:
        assert env_id in registry, f"{env_id} not registered"
        assert callable(registry[env_id].entry_point)


def test_no_jax_functional_env_registered():
    """The JAX backend implements the gymnax functional contract, not
    gymnasium.Env -- deliberately NOT registered (documented in the module
    docstring)."""
    assert not any(env_id.startswith("hydrogym/jax/") for env_id in gym.registry)


def test_register_all_is_idempotent():
    # gym.register raises on a duplicate id, so a second call must be a no-op.
    from hydrogym.registration import register_all

    register_all()  # must not raise


def test_import_is_lazy():
    """A fresh interpreter importing only hydrogym.registration must not pull
    in any solver backend or MPI."""
    code = (
        "import sys, hydrogym.registration; "
        "banned = ['mpi4py', 'firedrake', 'jax', 'jaxfluids']; "
        "leaked = [m for m in banned for k in sys.modules if k == m or k.startswith(m + '.')]; "
        "print('LEAKED:' + ','.join(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    leaked = out.stdout.strip().split("LEAKED:")[-1]
    assert leaked == "", f"lazy import violated, pulled in: {leaked}"


# ---------------------------------------------------------------------------
# Factory behavior, with each backend faked in sys.modules so the lazy
# `from hydrogym.X import ...` inside the entry point resolves to a stub.
# disable_env_checker=True because the factories' return values below are
# plain sentinels, not full gymnasium.Env instances.
# ---------------------------------------------------------------------------


def _fake_module(**attrs):
    mod = types.ModuleType("fake")
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


class _SentinelEnv(gym.Env):
    """A real gymnasium.Env that records what it was constructed with.

    gym.make validates that entry points return gymnasium.Env instances, so
    the fakes must too. ``marker`` lets each test assert which factory ran.
    """

    metadata = {"render_modes": []}
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

    def __init__(self, marker, payload=None):
        self.marker = marker
        self.payload = payload

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, False, False, {}


class _SentinelEnv(gym.Env):
    """A real gymnasium.Env that records what it was constructed with.

    gym.make validates that entry points return gymnasium.Env instances, so
    the fakes must too. ``marker`` lets each test assert which factory ran.
    """

    metadata = {"render_modes": []}
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

    def __init__(self, marker, payload=None):
        self.marker = marker
        self.payload = payload

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, False, False, {}


@pytest.fixture
def fake_firedrake(monkeypatch):
    captured = {}

    class FakeFlowEnv(_SentinelEnv):
        def __init__(self, env_config):
            super().__init__(marker="FAKE_FIREDRAKE_ENV", payload=env_config)
            captured["env_config"] = env_config

    flow = _fake_module(
        Cylinder="FAKE_CYLINDER_FLOW",
        RotaryCylinder="FAKE_ROT_FLOW",
        SemiImplicitBDF="FAKE_SEMI_IMPLICIT_BDF",
    )
    core = _fake_module(FlowEnv=FakeFlowEnv)
    # Faking the parents too: `from hydrogym.core import FlowEnv` must not
    # execute the real (backend-heavy) parent/module chain here.
    monkeypatch.setitem(sys.modules, "hydrogym.firedrake", flow)
    monkeypatch.setitem(sys.modules, "hydrogym.core", core)
    # _firedrake_make does `import hydrogym as hgym; ... hgym.firedrake`
    # (attribute access on the *package*, not a submodule import) --
    # hydrogym/__init__.py's lazy __getattr__ caches the real module into
    # hydrogym.__dict__ the first time anything ever does
    # `hydrogym.firedrake`, permanently, for the rest of the process. If any
    # other collected test module (e.g. test_pinball.py) already triggered
    # that real import before this fixture runs -- which pytest's collection
    # phase does for every test file up front, regardless of which tests you
    # actually select -- patching sys.modules alone is not enough: the cached
    # attribute on the real `hydrogym` module object still wins over it.
    # Patch that attribute directly too, so this fixture is correct
    # regardless of what else got collected/imported first.
    import hydrogym as _hydrogym_pkg

    monkeypatch.setattr(_hydrogym_pkg, "firedrake", flow, raising=False)
    return captured


def test_firedrake_factory_defaults(fake_firedrake):
    env = gym.make("hydrogym/Cylinder-v0", disable_env_checker=True)
    assert env.unwrapped.marker == "FAKE_FIREDRAKE_ENV"
    cfg = fake_firedrake["env_config"]
    assert cfg["flow"] == "FAKE_CYLINDER_FLOW"
    assert cfg["solver"] == "FAKE_SEMI_IMPLICIT_BDF"
    assert cfg["flow_config"] == {}
    assert cfg["solver_config"] == {}


def test_firedrake_factory_caller_overrides_win(fake_firedrake):
    gym.make(
        "hydrogym/Cylinder-v0",
        disable_env_checker=True,
        env_config={"solver_config": {"dt": 0.5}, "flow_config": {"x": 1}},
    )
    cfg = fake_firedrake["env_config"]
    assert cfg["solver_config"] == {"dt": 0.5}
    assert cfg["flow_config"] == {"x": 1}
    assert cfg["flow"] == "FAKE_CYLINDER_FLOW"  # default survives partial override


@pytest.fixture
def fake_maia(monkeypatch):
    captured = {}

    def fake_from_hf(name, **kwargs):
        captured["call"] = (name, kwargs)
        return _SentinelEnv(marker="MAIA_ENV_SENTINEL")

    # _maia_make imports `hydrogym.maia` as a package and calls
    # `maia.from_hf(...)` (attribute access, not a submodule import) -- see
    # its comment for why: importing hydrogym.maia.env_core directly
    # bypasses the package's lazy __getattr__-triggered env-class
    # registration and leaves _ENVIRONMENT_REGISTRY empty. The fake module
    # must expose `from_hf` as its own attribute to match.
    maia = _fake_module(from_hf=fake_from_hf)
    monkeypatch.setitem(sys.modules, "hydrogym.maia", maia)
    # `import hydrogym.maia as maia` resolves through attribute access on the
    # real `hydrogym` package, not a fresh sys.modules lookup each call --
    # confirmed empirically: two sequential gym.make() calls for two
    # different MAIA ids, each with its own sys.modules swap, both ended up
    # invoking the *first* call's fake (proven with a minimal repro outside
    # pytest entirely, so this is not a monkeypatch/pytest artifact -- it's
    # hydrogym/__init__.py's own lazy loader, which caches
    # `globals()["maia"] = module` permanently the first time anything ever
    # resolves `hydrogym.maia`, same mechanism already worked around for
    # `fake_firedrake` above). Patch the attribute directly too.
    import hydrogym as _hydrogym_pkg

    monkeypatch.setattr(_hydrogym_pkg, "maia", maia, raising=False)
    return captured


def test_maia_factory_forwards_kwargs(fake_maia):
    env = gym.make("hydrogym-maia/Cylinder_2D_Re200-v0", disable_env_checker=True, nproc=4)
    assert env.unwrapped.marker == "MAIA_ENV_SENTINEL"
    name, kwargs = fake_maia["call"]
    assert name == "Cylinder_2D_Re200"
    assert kwargs["nproc"] == 4
    # Cylinder_2D_Re200 ships a verified default probe grid (see
    # registration.py's _MAIA_ENVS) so gym.make() works with no probe_locations.
    assert "probe_locations" in kwargs


def test_maia_factory_env_without_default_probes_requires_them(fake_maia):
    # RotaryCylinder_2D_Re1000 / Cavity_2D_Re4140 have no verified default
    # probe grid -- the caller must supply one (same shape of requirement as
    # Nek's nproc). No defaults means an empty kwargs dict reaches from_hf,
    # which is where the real (unmocked) MaiaFlowEnv raises ConfigError.
    gym.make("hydrogym-maia/RotaryCylinder_2D_Re1000-v0", disable_env_checker=True)
    name, kwargs = fake_maia["call"]
    assert name == "RotaryCylinder_2D_Re1000"
    assert "probe_locations" not in kwargs


@pytest.fixture
def fake_nek(monkeypatch):
    captured = {}

    class FakeNekEnv:
        @classmethod
        def from_hf(cls, name, **kwargs):
            captured["call"] = (name, kwargs)
            return _SentinelEnv(marker="NEK_ENV_SENTINEL")

    nek = _fake_module()
    env = _fake_module(NekEnv=FakeNekEnv)
    nek.env = env
    monkeypatch.setitem(sys.modules, "hydrogym.nek", nek)
    monkeypatch.setitem(sys.modules, "hydrogym.nek.env", env)
    # Same hydrogym.__init__ lazy-loader caching risk as fake_maia/
    # fake_firedrake/fake_jaxfluids above ("nek" is also in the parent's
    # lazy __getattr__ allowlist) -- defensive.
    import hydrogym as _hydrogym_pkg

    monkeypatch.setattr(_hydrogym_pkg, "nek", nek, raising=False)
    return captured


def test_nek_factory_requires_nproc(fake_nek):
    with pytest.raises(TypeError, match="nproc"):
        gym.make("hydrogym-nek/TCFmini_3D_Re180-v0", disable_env_checker=True)
    assert "call" not in fake_nek  # refused before touching the backend


def test_nek_factory_forwards_kwargs(fake_nek):
    env = gym.make("hydrogym-nek/TCFmini_3D_Re180-v0", disable_env_checker=True, nproc=10)
    assert env.unwrapped.marker == "NEK_ENV_SENTINEL"
    assert fake_nek["call"] == ("TCFmini_3D_Re180", {"nproc": 10})


@pytest.fixture
def fake_jaxfluids(monkeypatch):
    captured = {}

    class FakeNozzle2D(_SentinelEnv):
        def __init__(self, env_config):
            super().__init__(marker="FAKE_NOZZLE2D", payload=env_config)
            captured["call"] = env_config

    envs = _fake_module(Nozzle2D=FakeNozzle2D)
    jxf = _fake_module(envs=envs)
    monkeypatch.setitem(sys.modules, "hydrogym.jaxfluids", jxf)
    monkeypatch.setitem(sys.modules, "hydrogym.jaxfluids.envs", envs)
    # Same hydrogym.__init__ lazy-loader caching risk as fake_maia/
    # fake_firedrake above -- defensive, not yet proven necessary for this
    # fixture specifically, but the mechanism is identical (`hydrogym.
    # jaxfluids` is also a lazy `_MPI_ATTRS`-style name in the parent's
    # __getattr__), so patch it the same way rather than wait to hit it.
    import hydrogym as _hydrogym_pkg

    monkeypatch.setattr(_hydrogym_pkg, "jaxfluids", jxf, raising=False)
    return captured


def test_jaxfluids_factory_merges_env_config(fake_jaxfluids):
    gym.make(
        "hydrogym-jaxfluids/Nozzle2D-v0",
        disable_env_checker=True,
        env_config={"a": 1},
        b=2,
    )
    # environment_name defaults to the registered HF environment (there is
    # no plain "Nozzle2D" HF environment, only resolution-suffixed variants
    # -- see registration.py's _JAXFLUIDS_ENVS comment) unless overridden.
    assert fake_jaxfluids["call"] == {"a": 1, "b": 2, "environment_name": "Nozzle2D_coarse"}


def test_jaxfluids_factory_caller_can_override_environment_name(fake_jaxfluids):
    gym.make(
        "hydrogym-jaxfluids/Nozzle2D-v0",
        disable_env_checker=True,
        env_config={"environment_name": "Nozzle2D_fine"},
    )
    assert fake_jaxfluids["call"]["environment_name"] == "Nozzle2D_fine"
