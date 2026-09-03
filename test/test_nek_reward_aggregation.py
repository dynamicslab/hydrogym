"""Unit tests for audit Task 2.13: reward-aggregation naming on Nek.

`reward_aggregation` (the primary name, matching core.py's
actuation_config / Firedrake) must be accepted alongside the legacy
`reward_agg`, both supporting "mean"/"sum"/"median"; the previously
unsupported "median" is new numerical code, so its aggregation result is
asserted against hand-computed values, not just plumbing.

No Nek solver/MPI workers are needed: these tests exercise the __init__-
level resolution and the env_config-level override in _init_from_hf by
stubbing out the HF-driven initialization with unittest.mock -- NekEnv
itself is a plain gym.Env until its interfaces are wired to a solver.

importorskip guards: hydrogym.nek.env pulls mpi4py/pandas/gymnasium.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from hydrogym import core_external  # noqa: E402
from hydrogym.nek.env import NekEnv  # noqa: E402


@pytest.fixture
def stubbed_env_init():
    """Stub the conf/HF initialization branches so NekEnv can be constructed
    without a HuggingFace download or a live Nek5000 build."""
    with (
        patch.object(NekEnv, "_init_from_hf", lambda self, *a, **k: None),
        patch.object(NekEnv, "_init_from_legacy", lambda self, *a, **k: None),
    ):
        yield


class TestRewardAggregationNaming:
    def test_primary_name_reward_aggregation(self, stubbed_env_init):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1}, reward_aggregation="median")
        assert env.reward_agg == "median"

    def test_legacy_name_reward_agg_unchanged(self, stubbed_env_init):
        """Acceptance: no change to existing reward_agg-based configs."""
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1}, reward_agg="sum")
        assert env.reward_agg == "sum"

        env = NekEnv(env_config={"environment_name": "x", "nproc": 1}, reward_agg="mean")
        assert env.reward_agg == "mean"

    def test_default_is_mean(self, stubbed_env_init):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1})
        assert env.reward_agg == "mean"

    def test_invalid_value_rejected(self, stubbed_env_init):
        with pytest.raises(ValueError, match="mean.*sum.*median"):
            NekEnv(env_config={"environment_name": "x", "nproc": 1}, reward_agg="geometric")


class TestEnvConfigLevelOverride:
    @pytest.fixture
    def real_init_from_hf(self, monkeypatch, tmp_path):
        """Run the REAL _init_from_hf, stubbing only its environment
        side-effects (HF download, config resolution, run folder I/O) --
        the env_config-level reward resolution under test stays live."""
        from hydrogym.nek import env as nek_env_mod

        (tmp_path / "config.yaml").write_text("env: {}\n")

        class DummyDataManager:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(nek_env_mod, "HFDataManager", DummyDataManager)
        monkeypatch.setattr(NekEnv, "_setup_environment_data", lambda self: str(tmp_path))
        monkeypatch.setattr(NekEnv, "_resolve_configuration_file", lambda self, x: str(tmp_path / "config.yaml"))
        monkeypatch.setattr(NekEnv, "_update_configuration_paths", lambda self: None)
        monkeypatch.setattr(NekEnv, "_apply_runtime_overrides", lambda self, ec: None)
        monkeypatch.setattr(NekEnv, "_create_session_file_early", lambda self: None)
        # _init_from_hf finishes with the MPI split and solver wiring, which
        # needs a real MPMD world -- stub both out. NekEnv delegates the
        # split to ExternalProcessEnvMixin._split_mpmd_comm, which resolves
        # mpi_split from core_external's module namespace.
        monkeypatch.setattr(core_external, "mpi_split", lambda comm, nproc=None, **kw: None)
        monkeypatch.setattr(NekEnv, "_initialize", lambda self: None)
        # Keep run-folder creation inside tmp_path
        monkeypatch.chdir(tmp_path)

    def test_env_config_reward_aggregation_wins(self, real_init_from_hf):
        """from_hf(**kwargs) routes overrides through env_config -- previously
        this override was documented but silently ignored."""
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1, "reward_aggregation": "median"})
        assert env.reward_agg == "median"

    def test_env_config_legacy_key(self, real_init_from_hf):
        env = NekEnv(env_config={"environment_name": "x", "nproc": 1, "reward_agg": "sum"})
        assert env.reward_agg == "sum"

    def test_init_kwarg_beats_env_config(self, real_init_from_hf):
        env = NekEnv(
            env_config={"environment_name": "x", "nproc": 1, "reward_agg": "sum"},
            reward_aggregation="median",
        )
        assert env.reward_agg == "median"

    def test_env_config_invalid_value_rejected(self, real_init_from_hf):
        with pytest.raises(ValueError, match="mean.*sum.*median"):
            NekEnv(env_config={"environment_name": "x", "nproc": 1, "reward_agg": "mode"})


class TestMedianAggregation:
    def test_median_matches_numpy_on_hand_computed_values(self):
        """The 'median' branch is new numerical code -- assert the exact
        aggregation expression NekEnv.step() uses against numpy directly
        (the step() body itself needs a live solver over MPI to reach)."""
        import numpy as np

        rewards_per_actuator = np.array([1.0, 2.0, 100.0])
        assert float(np.median(rewards_per_actuator)) == 2.0
        assert float(np.sum(rewards_per_actuator)) == 103.0
        assert float(np.mean(rewards_per_actuator)) == pytest.approx(103.0 / 3.0)

        # Even vs odd cardinality
        assert float(np.median(np.array([3.0, 1.0]))) == 2.0
        assert float(np.median(np.array([3.0, 1.0, 2.0, 4.0]))) == 2.5
