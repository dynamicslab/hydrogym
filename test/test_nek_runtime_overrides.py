"""Unit tests for audit Task 2.8: generalized NekEnv._apply_runtime_overrides.

Previously only 5 hardcoded shorthand keys were accepted and anything else
(WALLTIME, dt, CASENAME, ...) was silently dropped -- contradicting from_hf's
broader-sounding docstring. Now any dotted config-tree path
(section.param=value) can be overridden, the path's existence is validated
(unknown paths raise ConfigError instead of being silently ignored), and the
five legacy shorthands keep working unchanged.

Config-parsing only: _apply_runtime_overrides is exercised on a bare NekEnv
(object.__new__) with an OmegaConf tree -- no MPI workers, HF download, or
solver needed.

pytest.importorskip guards: mpi4py/pandas via hydrogym.nek.env.
"""

import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from omegaconf import OmegaConf  # noqa: E402

from hydrogym.nek.env import ConfigError, NekEnv  # noqa: E402

BASE_CONF = {
    "normalization": {"normalize_input": False},
    "episode": {"max_interactions": 100, "reward_mode": "default"},
    "initial_conditions": {"random_init": False},
    "rl_interface": {"rescale_actions": True},
    "simulation": {"walltime": 3600},
}


@pytest.fixture
def env():
    """Bare NekEnv with only what _apply_runtime_overrides touches."""
    env = object.__new__(NekEnv)
    env.conf = OmegaConf.create(BASE_CONF)
    env.environment_name = "TestEnv"
    return env


class TestShorthandKeysUnchanged:
    """The five legacy keys must keep working exactly as before."""

    def test_all_five_shorthands_apply(self, env):
        env._apply_runtime_overrides(
            {
                "normalize_input": True,
                "nb_interactions": 7,
                "random_init": True,
                "rescale_actions": False,
                "rew_mode": "drag",
            }
        )
        assert env.conf.normalization.normalize_input is True
        assert env.conf.episode.max_interactions == 7
        assert env.conf.initial_conditions.random_init is True
        assert env.conf.rl_interface.rescale_actions is False
        assert env.conf.episode.reward_mode == "drag"

    def test_reserved_keys_ignored(self, env):
        """Structural keys configure the env machinery, not the config tree --
        they must neither raise nor touch the config."""
        before = OmegaConf.to_container(env.conf)
        env._apply_runtime_overrides({"environment_name": "other", "nproc": 3, "hf_token": "t"})
        assert OmegaConf.to_container(env.conf) == before


class TestDottedPathOverrides:
    def test_dotted_path_reaches_conf(self, env):
        env._apply_runtime_overrides({"simulation.walltime": 100})
        assert env.conf.simulation.walltime == 100

    def test_multiple_dotted_paths(self, env):
        env._apply_runtime_overrides({"episode.max_interactions": 5, "simulation.walltime": 10})
        assert env.conf.episode.max_interactions == 5
        assert env.conf.simulation.walltime == 10

    def test_unknown_path_raises_config_error(self, env):
        with pytest.raises(ConfigError, match="simulation.nope"):
            env._apply_runtime_overrides({"simulation.nope": 1})

    def test_unknown_section_raises_config_error(self, env):
        with pytest.raises(ConfigError, match="no_such_section.param"):
            env._apply_runtime_overrides({"no_such_section.param": 1})

    def test_unknown_bare_key_raises_config_error(self, env):
        """Previously silently dropped; now a clear error naming the valid forms."""
        with pytest.raises(ConfigError, match="WALLTIME"):
            env._apply_runtime_overrides({"WALLTIME": 100})

    def test_shorthand_maps_to_same_path_as_dotted(self, env):
        """Both forms must target the same config entry."""
        env._apply_runtime_overrides({"nb_interactions": 42})
        assert env.conf.episode.max_interactions == 42
        env._apply_runtime_overrides({"episode.max_interactions": 43})
        assert env.conf.episode.max_interactions == 43
