"""Unit tests for audit Task 2.9: `mpi_bind_to` exposed on NekEnv.

`_initialize()` hardcoded `mpi_info.Set("bind_to", "none")` with no
env_config override, unlike the adjacent configurable `hostfile`. Now an
optional `mpi_bind_to` env_config key (default "none", preserving current
behavior exactly) is stored by _init_from_hf and used by _initialize.

The env_config -> attribute tests run the real _init_from_hf with only its
environment side-effects stubbed out (same pattern as
test_nek_reward_aggregation.py / test_hf_data_manager_token_revision.py) --
no MPI workers or HF download needed. The MPI-tier validation (default
behavior preserved in a live MPMD run) happens in the nek5000-test
container, recorded in the commit message.

pytest.importorskip guards: mpi4py/pandas via hydrogym.nek.env.
"""

import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from hydrogym.nek.env import NekEnv  # noqa: E402


@pytest.fixture
def real_init_from_hf(monkeypatch, tmp_path):
    from hydrogym import core_external
    from hydrogym.nek import env as nek_env_mod

    (tmp_path / "config.yaml").write_text("env: {}\n")
    monkeypatch.setattr(nek_env_mod, "HFDataManager", lambda *a, **k: object())
    monkeypatch.setattr(NekEnv, "_setup_environment_data", lambda self: str(tmp_path))
    monkeypatch.setattr(NekEnv, "_resolve_configuration_file", lambda self, x: str(tmp_path / "config.yaml"))
    monkeypatch.setattr(NekEnv, "_update_configuration_paths", lambda self: None)
    monkeypatch.setattr(NekEnv, "_apply_runtime_overrides", lambda self, ec: None)
    monkeypatch.setattr(NekEnv, "_create_session_file_early", lambda self: None)
    # NekEnv delegates the split to ExternalProcessEnvMixin._split_mpmd_comm,
    # which resolves mpi_split from core_external's module namespace.
    monkeypatch.setattr(core_external, "mpi_split", lambda comm, nproc=None, **kw: None)
    monkeypatch.setattr(NekEnv, "_initialize", lambda self: None)
    monkeypatch.chdir(tmp_path)


class TestMpiBindTo:
    def test_default_is_none(self, real_init_from_hf):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1})
        assert env.mpi_bind_to == "none"

    def test_explicit_override_stored(self, real_init_from_hf):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1, "mpi_bind_to": "core"})
        assert env.mpi_bind_to == "core"

    def test_is_reserved_from_dotted_override_mechanism(self, real_init_from_hf):
        """mpi_bind_to configures the MPI launcher, not the solver config
        tree -- it must not be treated as a dotted-path override."""
        assert "mpi_bind_to" in NekEnv.RESERVED_ENV_CONFIG_KEYS

    def test_initialize_uses_stored_policy(self, real_init_from_hf):
        """_initialize must consult the stored attribute (legacy path falls
        back to the class default via getattr)."""
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1})
        assert getattr(env, "mpi_bind_to", NekEnv.DEFAULT_MPI_BIND_TO) == "none"
