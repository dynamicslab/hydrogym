"""Unit tests for hydrogym.jax.envs.channel (audit Tasks 2.14 / 2.15).

Task 2.14: Lx/Ly/Lz/nu/Nx/Ny/Nz must be readable from env_config with
defaults identical to the previously hardcoded values.
Task 2.15: hf_repo_id/cache_dir/use_clean_cache must be forwarded to
HFDataManager with defaults matching JAXFlowEnv's pattern.

Skipped when jax is not installed (e.g. bare venv / Firedrake CI
container). Hermetic by design: every test passes initial_field_dir
pointing at tiny synthetic numpy fields, so nothing touches the HF Hub.
Construction is cheap because no reset()/JIT-compile of the solver
operator is performed -- only wavenumber-grid setup.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F401 -- env module imports jax at import time


@pytest.fixture
def synthetic_field_dir(tmp_path):
    """A minimal initial_field directory; shapes are irrelevant because
    these tests never call reset()."""
    field_dir = tmp_path / "initial_field"
    field_dir.mkdir(parents=True)
    for name in ("U", "V", "W"):
        np.save(field_dir / f"{name}.npy", np.zeros((8, 8, 8), dtype=np.float32))
    return field_dir


class TestEnvConfigOverrides:
    def test_default_params_match_previous_hardcoded_values(self, synthetic_field_dir):
        """Task 2.14 acceptance: default (no override) behavior is
        byte-for-byte the old hardcoded configuration."""
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        env = ChannelFlowSpectralEnv({"initial_field_dir": str(synthetic_field_dir)})

        assert env.Nx == 72 and env.Ny == 72 and env.Nz == 72
        assert env.Lx == pytest.approx(2.0 * np.pi)
        assert env.Ly == pytest.approx(1.0 * np.pi)
        assert env.Lz == pytest.approx(2.0)
        assert env.nu == pytest.approx(1.9e-3)
        assert env.equation.Nx == 72 and env.equation.Ny == 72 and env.equation.Nz == 72
        assert env.equation.nu == pytest.approx(1.9e-3)

    def test_grid_override_reaches_pseudo_spectral_solver(self, synthetic_field_dir):
        """Task 2.14 acceptance: an overridden Nx actually reaches the
        constructed PseudoSpectralNavierStokes3D."""
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        env = ChannelFlowSpectralEnv({"initial_field_dir": str(synthetic_field_dir), "Nx": 48, "Ny": 48, "Nz": 48})

        assert env.Nx == 48
        assert env.equation.Nx == 48 and env.equation.Ny == 48 and env.equation.Nz == 48

    def test_physical_params_overridable(self, synthetic_field_dir):
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        env = ChannelFlowSpectralEnv(
            {
                "initial_field_dir": str(synthetic_field_dir),
                "Lx": 4.0,
                "Ly": 2.0,
                "Lz": 3.0,
                "nu": 5.0e-3,
            }
        )

        assert env.Lx == pytest.approx(4.0)
        assert env.Ly == pytest.approx(2.0)
        assert env.Lz == pytest.approx(3.0)
        assert env.nu == pytest.approx(5.0e-3)
        assert env.equation.Lx == pytest.approx(4.0)
        assert env.equation.nu == pytest.approx(5.0e-3)


class TestHFConfigKeys:
    @pytest.fixture
    def recording_data_manager(self, monkeypatch, tmp_path):
        """HFDataManager subclass that records __init__ kwargs and returns
        a synthetic env dir from get_environment_path (never downloads)."""
        import hydrogym.jax.envs.channel as channel_mod

        fake_env_dir = tmp_path / "fake_hf_env"
        (fake_env_dir / "initial_field").mkdir(parents=True)
        for name in ("U", "V", "W"):
            np.save(fake_env_dir / "initial_field" / f"{name}.npy", np.zeros((4, 4, 4), dtype=np.float32))

        recorded = {}

        class RecordingDataManager(channel_mod.HFDataManager):
            def __init__(self, *args, **kwargs):
                recorded.update(kwargs)
                super().__init__(*args, **kwargs)

            def get_environment_path(self, environment_name):
                return str(fake_env_dir)

        monkeypatch.setattr(channel_mod, "HFDataManager", RecordingDataManager)
        return recorded

    def test_hf_overrides_reach_data_manager(self, recording_data_manager, synthetic_field_dir):
        """Task 2.15 acceptance: overrides reach the constructed
        HFDataManager."""
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        ChannelFlowSpectralEnv(
            {
                "hf_repo_id": "unit-test/repo",
                "cache_dir": "/tmp/unit_test_cache",
                "use_clean_cache": False,
            }
        )

        assert recording_data_manager["repo_id"] == "unit-test/repo"
        assert recording_data_manager["cache_dir"] == "/tmp/unit_test_cache"
        assert recording_data_manager["use_clean_cache"] is False

    def test_hf_defaults_match_jaxflowenv(self, recording_data_manager, synthetic_field_dir):
        """Task 2.15 acceptance: default behavior unchanged -- the kwargs
        passed when nothing is overridden must equal JAXFlowEnv's defaults."""
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        ChannelFlowSpectralEnv({})

        assert recording_data_manager["repo_id"] == "dynamicslab/HydroGym-environments"
        assert recording_data_manager["cache_dir"] is None
        assert recording_data_manager["local_fallback_dir"] is None
        assert recording_data_manager["use_clean_cache"] is True
        assert recording_data_manager["fallback_profile"] == "JAX"
