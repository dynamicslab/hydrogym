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

    maia = _fake_module()
    env_core = _fake_module(from_hf=fake_from_hf)
    maia.env_core = env_core  # `from hydrogym.maia import env_core` needs the attr
    monkeypatch.setitem(sys.modules, "hydrogym.maia", maia)
    monkeypatch.setitem(sys.modules, "hydrogym.maia.env_core", env_core)
    return captured


def test_maia_factory_forwards_kwargs(fake_maia):
    env = gym.make("hydrogym-maia/Cylinder_2D_Re200-v0", disable_env_checker=True, nproc=4)
    assert env.unwrapped.marker == "MAIA_ENV_SENTINEL"
    assert fake_maia["call"] == ("Cylinder_2D_Re200", {"nproc": 4})


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
    return captured


def test_jaxfluids_factory_merges_env_config(fake_jaxfluids):
    gym.make(
        "hydrogym-jaxfluids/Nozzle2D-v0",
        disable_env_checker=True,
        env_config={"a": 1},
        b=2,
    )
    assert fake_jaxfluids["call"] == {"a": 1, "b": 2}
