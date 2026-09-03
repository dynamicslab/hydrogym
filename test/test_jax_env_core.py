"""Unit tests for hydrogym.jax.env_core (audit Tasks 2.10 / 2.11).

Skipped entirely when jax is not installed (hydrogym.jax.env_core imports
jax at import time), e.g. in the bare test venv / the Firedrake CI
container.

CAVEAT on scope: JAXFlowEnv (the HF-integrated base class) is currently
instantiated by no shipped code -- both concrete JAX environments
(KolmogorovFlow, ChannelFlowSpectralEnv) subclass JAXFlowEnvBase, the
no-HF variant. JAXFlowEnv.__init__ is moreover broken independently of
what these tests cover: it calls _read_property_file/_get_property, which
only exist on the MAIA side (hydrogym/maia/env_core.py). The tests here
therefore exercise construction only up to the point they care about and
expect a ConfigError ("No configuration file found") to surface afterwards;
the assertions are on what happened BEFORE that error.

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
        """Task 2.10: JAXFlowEnv must pass fallback_profile='JAX' to
        HFDataManager, matching the MAIA/NEK sibling pattern."""
        from hydrogym.jax import env_core

        # Pre-create the local cache dir so _setup_environment_data
        # short-circuits there and construction proceeds to the (expected)
        # missing-config-file error instead of the offline fallback.
        # (Both namespace spellings created: the namespace itself is Task
        # 2.11's subject -- this test only needs *a* cache hit and must not
        # care which name is current.)
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
        """Task 2.11: JAXFlowEnv's local cache namespace must be JAX-specific
        ("jaxgym"), not "maiagym" (copy-paste from the MAIA backend). The
        maiagym dir is a decoy: with the old code it would be used and the
        capsys output would mention it."""
        from hydrogym.jax import env_core

        (fake_home / ".cache" / "jaxgym" / "unit_test_env").mkdir(parents=True)
        (fake_home / ".cache" / "maiagym" / "unit_test_env").mkdir(parents=True)

        with pytest.raises(env_core.ConfigError, match="No configuration file"):
            env_core.JAXFlowEnv({"environment_name": "unit_test_env"})

        out = capsys.readouterr().out
        assert "jaxgym" in out
        assert "maiagym" not in out
