"""Unit tests for hydrogym.jax.env_core.

Skipped entirely when jax is not installed (hydrogym.jax.env_core imports
jax at import time), e.g. in the bare test venv / the Firedrake CI
container.

Scope: the shipped JAX environments subclass JAXFlowEnvBase (no HF data);
JAXFlowEnv (the HF-integrated base class) is exercised here only up to the
expected ConfigError ("No configuration file found"), and the assertions are
on what happened before that error.

"""

import pytest

jax = pytest.importorskip("jax")  # noqa: F401


@pytest.fixture
def offline_data_manager(monkeypatch):
    """Replace HFDataManager with a recorder whose get_environment_path
    always fails (simulates being offline / no HF data available)."""
    from hydrogym.jax import env_core

    init_kwargs: dict = {}

    class RecordingDataManager:
        def __init__(self, *args, **kwargs):
            init_kwargs.update(kwargs)

        def get_environment_path(self, environment_name):
            raise RuntimeError(f"offline: no data for {environment_name!r}")

    monkeypatch.setattr(env_core, "HFDataManager", RecordingDataManager)
    return init_kwargs


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Point Path.home() at tmp_path for the ~/.cache/<namespace> lookup.

    NOTE: this temporarily patches pathlib.Path.home() process-wide (the
    HFEnvConfigMixin resolution logic lives in hydrogym.hf_env_mixin, whose
    Path is the same pathlib.Path class); monkeypatch restores it afterwards.
    """
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


class TestFallbackProfile:
    def test_hf_data_manager_receives_jax_fallback_profile(self, offline_data_manager, fake_home):
        """JAXFlowEnv must pass fallback_profile='JAX' to
        HFDataManager, matching the MAIA/NEK sibling pattern."""
        from hydrogym.jax import env_core

        # Pre-create the local cache dir so _setup_environment_data
        # short-circuits there and construction proceeds to the (expected)
        # missing-config-file error instead of the offline fallback.
        for namespace in ("jaxgym", "maiagym"):
            (fake_home / ".cache" / namespace / "unit_test_env").mkdir(parents=True)

        with pytest.raises(env_core.ConfigError, match="No configuration file"):
            env_core.JAXFlowEnv({"environment_name": "unit_test_env"})

        assert offline_data_manager.get("fallback_profile") == "JAX"

    def test_solver_type_is_a_valid_profile_name(self):
        """SOLVER_TYPE must be a key of SOLVER_PROFILES, or HFDataManager's
        fallback path would raise KeyError on first use."""
        from hydrogym.data_manager import SOLVER_PROFILES
        from hydrogym.jax.env_core import JAXFlowEnv

        assert JAXFlowEnv.SOLVER_TYPE in SOLVER_PROFILES


class TestCacheNamespace:
    def test_cache_namespace_is_jax_specific(self, offline_data_manager, fake_home, capsys):
        """JAXFlowEnv's local cache namespace must be JAX-specific
        ("jaxgym"), not 'maiagym'."""
        from hydrogym.jax import env_core

        (fake_home / ".cache" / "jaxgym" / "unit_test_env").mkdir(parents=True)
        (fake_home / ".cache" / "maiagym" / "unit_test_env").mkdir(parents=True)

        with pytest.raises(env_core.ConfigError, match="No configuration file"):
            env_core.JAXFlowEnv({"environment_name": "unit_test_env"})

        out = capsys.readouterr().out
        assert "jaxgym" in out
        assert "maiagym" not in out


class TestGymnaxVersion:
    """gymnax >= 1.0 changed the Environment contract (6-tuple step, is_terminated, truncation at
    max_steps_in_episode); hydrogym.jax pins and checks gymnax < 1."""

    def test_installed_version_is_supported_and_steps(self):
        from importlib.metadata import version

        from hydrogym.jax.env_core import check_gymnax_version

        installed = version("gymnax")
        assert int(installed.split(".")[0]) < 1
        check_gymnax_version()  # does not raise
        check_gymnax_version("0.0.9")

    @pytest.mark.parametrize("v", ["1.0.0", "1.0.1", "2.3"])
    def test_gymnax_1_is_rejected_with_a_clear_message(self, v):
        from hydrogym.jax.env_core import check_gymnax_version

        with pytest.raises(ImportError, match="gymnax < 1.0"):
            check_gymnax_version(v)

    def test_kolmogorov_step_returns_the_gymnax_0_contract(self):
        """The public gymnax ``step`` returns 5 items and does not reset after the first step."""
        import jax.numpy as jnp

        from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

        env = KolmogorovFlow({"dt": 1e-1})
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key, params)
        out = env.step(key, state, jnp.zeros(4), params)
        assert len(out) == 5
        assert int(out[1].time) == 1 and not bool(out[3])
