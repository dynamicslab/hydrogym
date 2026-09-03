"""
Core Jax Gym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

"""

import os
from functools import partial
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
import omegaconf
import toml
from flax import struct
from gymnax.environments import environment, spaces

from hydrogym.data_manager import SOLVER_PROFILES, HFDataManager  # noqa: F401
from hydrogym.hf_env_mixin import ConfigError, HFEnvConfigMixin


class EnvParams(environment.EnvParams):
    """Gymnax environment parameters extended with the environment config dictionary.

    Attributes:
        config: Environment configuration (grid sizes, control bounds, etc.)
            as loaded from the environment's configuration file.
    """

    config: dict


EnvState = TypeVar("EnvState", bound=environment.EnvState)


class JAXFlowEnv(HFEnvConfigMixin, environment.Environment[EnvState, EnvParams]):
    """
    Base JAXFlowEnv with Hugging Face Hub integration for configuration management.

    This environment provides a Gymnax-compatible interface for CFD simulations
    using JAX solvers. It handles:
    - Environment data management via Hugging Face Hub
    - Configuration file resolution and loading
    - Action space configuration

    Attributes:
        environment_name: Name of the CFD environment configuration.
        env_data_path: Path to the local environment data directory.
        cfg: OmegaConf configuration object.
        observation_space: Gymnax observation space.
        action_space: Gymnax action space.
    """

    # Solver profile used by HFDataManager when no sentinel file is found
    # (offline / legacy data). Must be a key of data_manager.SOLVER_PROFILES.
    SOLVER_TYPE: str = "JAX"

    # Per-backend cache directory under ~/.cache
    HF_CACHE_NAMESPACE = "jaxgym"

    @staticmethod
    def _resolve_num_substeps(cfg_section) -> int:
        """
        Resolve the per-actuation substep count from the environment's YAML
        config section (e.g. cfg.jax).

        ``num_substeps`` is the primary key (matching Firedrake's
        non-deprecated name and core.py's actuation_config); the old
        ``num_sim_substeps_per_actuation`` key keeps working with a
        DeprecationWarning.

        Raises:
            ConfigError: If the section defines neither key.
        """
        import warnings

        if "num_substeps" in cfg_section:
            return cfg_section["num_substeps"]
        if "num_sim_substeps_per_actuation" in cfg_section:
            warnings.warn(
                "num_sim_substeps_per_actuation in the environment config is deprecated, rename it to num_substeps",
                DeprecationWarning,
                stacklevel=3,
            )
            return cfg_section["num_sim_substeps_per_actuation"]
        raise ConfigError(
            "Environment config section defines neither 'num_substeps' nor (deprecated) "
            "'num_sim_substeps_per_actuation'"
        )

    def __init__(self, env_config: Dict):
        """
        Initialize the JAXFlowEnv environment.

        Args:
            env_config: Configuration dictionary containing:
                - environment_name (str): Required. Name of the environment.
                - hf_repo_id (str): HF repository ID. Default: 'dynamicslab/HydroGym-environments'
                - local_fallback_dir (str): Optional local fallback directory.
                - use_clean_cache (bool): Whether to use clean cache. Default: True
                - configuration_file (str): Optional path to config file.
                - is_testing (bool): Whether in testing mode. Default: False
                - probe_locations (list): Probe coordinate locations.
                - obs_normalization_strategy (str): One of 'U_inf', 'probewise_mean_std',
                    'none', 'customized'.
                - obs_loc (list): Required if strategy is 'customized'.
                - obs_scale (list): Required if strategy is 'customized'.
        Raises:
            ConfigError: If required configuration is missing or invalid.
        """

        # Initialize HF data manager
        self.hf_repo_id = env_config.get("hf_repo_id", "dynamicslab/HydroGym-environments")
        self.local_fallback_dir = env_config.get("local_fallback_dir", None)
        self.use_clean_cache = env_config.get("use_clean_cache", True)
        self.hf_token = env_config.get("hf_token", None)
        self.hf_revision = env_config.get("hf_revision", None)

        self.data_manager = HFDataManager(
            repo_id=self.hf_repo_id,
            local_fallback_dir=self.local_fallback_dir,
            use_clean_cache=self.use_clean_cache,
            fallback_profile=self.SOLVER_TYPE,
            token=self.hf_token,
            revision=self.hf_revision,
        )

        # Environment identification
        self.environment_name = env_config.get("environment_name")

        if not self.environment_name:
            raise ConfigError("'environment_name' must be specified in env_config")

        # Download/get environment configuration
        self.env_data_path = self._setup_environment_data()

        # Handle configuration file - support multiple ways of specifying it
        self.configuration_file = self._resolve_configuration_file(env_config.get("configuration_file"))

        if not self.configuration_file:
            raise ConfigError(
                f"No configuration file found. Please either:\n"
                f"1. Provide configuration_file='/path/to/config.yaml' in env_config\n"
                f"2. Add a config.yaml file to the HF environment: {self.env_data_path}\n"
                f"Available files: {os.listdir(self.env_data_path) if os.path.exists(self.env_data_path) else 'N/A'}"
            )

        # Load configuration
        self.cfg = omegaconf.OmegaConf.load(self.configuration_file)

        # Update paths in configuration to use downloaded data
        self._update_configuration_paths()

        self.runtime_property_file = os.path.join(self.env_data_path, "properties_run.toml")

        self.num_substeps_per_iteration = self._resolve_num_substeps(self.cfg.jax)
        self.observation_type = self.cfg.jax.observation_type
        self.max_episode_steps = self.cfg.env.max_episode_steps
        self.num_inputs = self.cfg.jax.num_action_inputs * self.cfg.env.n_agents
        self.MAX_CONTROL = self.cfg.jax.max_control
        self.render = self.cfg.jax.render
        self.compute_grad = self.cfg.compute_grad

        # Read property file and extract parameters
        self.runtime_property_file_data = self._read_property_file(self.runtime_property_file)
        self.Retau = self._get_property(self.runtime_property_file_data, "Retau")
        self.xLength = self._get_property(self.runtime_property_file_data, "xLength")
        self.yLength = self._get_property(self.runtime_property_file_data, "yLength")
        self.Nx = self._get_property(self.runtime_property_file_data, "Nx")
        self.Ny = self._get_property(self.runtime_property_file_data, "Ny")
        self.nDim = self._get_property(self.runtime_property_file_data, "nDim")
        self.zLength = self._get_property(self.runtime_property_file_data, "zLength") if self.nDim == 3 else self.dX
        self.Nz = self._get_property(self.runtime_property_file_data, "Nz") if self.nDim == 3 else self.dX

    def _update_configuration_paths(self) -> None:
        """
        Update file paths in the configuration to point to downloaded environment data.

        This tells the CFD solver where to find the actual simulation files.
        """
        print(f"Environment data located at: {self.env_data_path}")

        # Update paths that the CFD solver needs
        path_mappings = {
            "maia.runtime_property_file": "properties_run.toml",
        }

        for config_key, filename in path_mappings.items():
            file_path = os.path.join(self.env_data_path, filename)
            if os.path.exists(file_path):
                # Update the configuration with absolute path
                keys = config_key.split(".")
                config_section = self.cfg
                for key in keys[:-1]:
                    if key not in config_section:
                        config_section[key] = {}
                    config_section = config_section[key]
                config_section[keys[-1]] = file_path
                print(f"CFD solver will use: {config_key} = {file_path}")
            else:
                print(f"Warning: Required file not found: {file_path}")

    def get_environment_files_info(self) -> Dict:
        """
        Get information about where environment files are stored.

        Useful for debugging and understanding file locations.

        Returns:
            Dictionary containing environment name, paths, and file information.
        """
        info = {
            "environment_name": self.environment_name,
            "local_cache_path": self.env_data_path,
            "configuration_file": self.configuration_file,
            "files": {},
        }

        # List all files in the environment
        if os.path.exists(self.env_data_path):
            for root, dirs, files in os.walk(self.env_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.env_data_path)
                    info["files"][rel_path] = {
                        "absolute_path": file_path,
                        "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    }

        return info

    @classmethod
    def create_from_hf_env(
        cls,
        environment_name: str,
        hf_repo_id: str = "your-username/maiagym-envs",
        local_fallback_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Create environment directly from HF environment name.

        Args:
            environment_name: Name of environment (e.g., 'Cylinder_2D_Re200').
            hf_repo_id: Hugging Face repository ID.
            local_fallback_dir: Local fallback directory.
            **kwargs: Additional environment configuration parameters.

        Returns:
            Configured MaiaFlowEnv instance.
        """
        env_config = {
            "environment_name": environment_name,
            "hf_repo_id": hf_repo_id,
            "local_fallback_dir": local_fallback_dir,
            **kwargs,
        }

        return cls(env_config)

    def get_available_environments(self) -> List[str]:
        """
        Get list of all available environments from HF Hub.

        Returns:
            List of environment names.
        """
        return self.data_manager.get_available_environments()

    def update_environment_data(self, force_download: bool = False) -> None:
        """
        Update environment data from HF Hub.

        Args:
            force_download: Force re-download even if cached.
        """
        self.env_data_path = self.data_manager.download_environment(
            self.environment_name, force_download=force_download
        )
        self._update_configuration_paths()

        # Reload configuration
        self.cfg = omegaconf.OmegaConf.load(self.configuration_file)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset the environment to its initial condition.

        Args:
            key: PRNG key for stochastic resets.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, state)`` of the initial observation and state.

        Raises:
            NotImplementedError: Always; subclasses define the actual reset.
        """
        raise NotImplementedError

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Compute the observation from the current environment state.

        Args:
            state: Current environment state.
            params: Environment parameters.
            key: Optional PRNG key for stochastic observations.

        Returns:
            Observation array.

        Raises:
            NotImplementedError: Always; subclasses define the observation model.
        """
        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Whether the episode has ended.

        An episode terminates when the state flags ``terminal`` or the elapsed
        time reaches ``params.max_episode_steps``.

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Boolean array indicating termination.
        """
        return jnp.logical_or(state.terminal, state.time >= params.max_episode_steps)

    def step_env(self, key: chex.PRNGKey, state: EnvState, action: jnp.array, params: EnvParams):
        """Advance the environment by one step under the given action.

        Args:
            key: PRNG key for stochastic transitions.
            state: Current environment state.
            action: Control (actuation) input.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, next_state, reward, done, info)``.

        Raises:
            NotImplementedError: Always; subclasses define the dynamics.
        """
        raise NotImplementedError


### NOTE: Issues with HF manager resulted in below minimal, HF-free class implementation for now.
### Will re-integrate HF functionality.


class JAXFlowEnvBase(environment.Environment[EnvState, EnvParams]):
    """
    Base JAXFlowEnv without Hugging Face Hub integration.
    Contains core environment interface methods required by Gymnax.

    """

    def __init__(self, env_config: Optional[dict] = None):
        """Initialize the environment with an optional configuration dictionary.

        Args:
            env_config: Environment configuration options (e.g.
                ``max_episode_steps``); defaults to an empty dict.
        """
        self.env_config = env_config or {}

    def default_params(self) -> EnvParams:
        """Return the environment's default parameters.

        Sets ``self.max_episode_steps`` from ``env_config`` (default 1000) as a
        side effect.

        Returns:
            Default environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must provide the parameters.
        """
        self.max_episode_steps = self.env_config.get("max_episode_steps", 1000)
        raise NotImplementedError

    def name(self) -> str:
        """Return the environment's name.

        Raises:
            NotImplementedError: Always; subclasses must provide a name.
        """
        raise NotImplementedError

    def action_space(self, params: Optional[EnvParams] = None):
        """Return the (Box) action space bounded by the parameters' action limits.

        Args:
            params: Environment parameters; defaults to ``self.default_params``.

        Returns:
            Gymnax Box space of shape ``(params.action_dim,)`` with bounds
            ``[params.min_action, params.max_action]``.
        """
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: EnvParams):
        """Return the (Box) observation space bounded by the parameters' observation limits.

        Args:
            params: Environment parameters.

        Returns:
            Gymnax Box space of shape ``(params.obs_dim,)`` with bounds
            ``[params.min_obs, params.max_obs]``.
        """
        return spaces.Box(
            low=params.min_obs,
            high=params.max_obs,
            shape=(params.obs_dim,),
        )

    def _clip_action(self, action: chex.Array, params: EnvParams) -> chex.Array:
        return jnp.clip(action, params.min_action, params.max_action)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Whether the episode has ended (state flags ``terminal`` or time limit reached).

        Args:
            state: Current environment state.
            params: Environment parameters.

        Returns:
            Boolean array indicating termination.
        """
        return jnp.logical_or(state.terminal, state.time >= params.max_episode_steps)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset the environment to its initial condition.

        Args:
            key: PRNG key for stochastic resets.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, state)`` of the initial observation and state.

        Raises:
            NotImplementedError: Always; subclasses define the actual reset.
        """
        raise NotImplementedError

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Compute the observation from the current environment state.

        Args:
            state: Current environment state.
            params: Environment parameters.
            key: Optional PRNG key for stochastic observations.

        Returns:
            Observation array.

        Raises:
            NotImplementedError: Always; subclasses define the observation model.
        """
        raise NotImplementedError

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ):
        """Advance the environment by one step under the given action.

        Args:
            key: PRNG key for stochastic transitions.
            state: Current environment state.
            action: Control (actuation) input.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, next_state, reward, done, info)``.

        Raises:
            NotImplementedError: Always; subclasses define the dynamics.
        """
        raise NotImplementedError


###############################################################

# BELOW CODE FROM PUREJAXRL REPO [1] WITH SLIGHT MODIFICATIONS
# [1] https://github.com/luchris429/purejaxrl/

###############################################################


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        """Wrap the given environment.

        Args:
            env: The Gymnax environment (or wrapper) being wrapped.
        """
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        """Wrap the given environment (see :class:`GymnaxWrapper`)."""
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        """Return the wrapped environment's observation space, flattened to 1D.

        Args:
            params: Environment parameters.

        Returns:
            Gymnax Box space whose shape is the product of the underlying
            space's shape, with the same bounds and dtype.

        Raises:
            AssertionError: If the wrapped space is not a Box.
        """
        assert isinstance(self._env.observation_space(params), spaces.Box), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the wrapped environment and flatten the initial observation.

        Args:
            key: PRNG key for the reset.
            params: Environment parameters (None uses the environment default).

        Returns:
            Tuple ``(obs, state)`` with the flattened observation.
        """
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the wrapped environment and flatten the returned observation.

        Args:
            key: PRNG key for the transition.
            state: Current environment state.
            action: Action to apply.
            params: Environment parameters (None uses the environment default).

        Returns:
            Tuple ``(obs, state, reward, done, info)`` with the flattened
            observation; all other values are passed through unchanged.
        """
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    """State bookkeeping for :class:`LogWrapper`.

    Attributes:
        env_state: The wrapped environment's own state.
        episode_returns: Return accumulated so far in the current episode.
        episode_lengths: Number of steps taken so far in the current episode.
        returned_episode_returns: Return of the most recently completed episode.
        returned_episode_lengths: Length of the most recently completed episode.
        timestep: Total number of steps taken across all episodes.
    """

    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        """Wrap the given environment (see :class:`GymnaxWrapper`)."""
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the wrapped environment and zero the episode bookkeeping.

        Args:
            key: PRNG key for the reset.
            params: Environment parameters (None uses the environment default).

        Returns:
            Tuple ``(obs, LogEnvState)`` with all counters initialized to zero.
        """
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the wrapped environment, updating and exporting episode statistics.

        Accumulates the running episode return/length and, on episode end,
        freezes them into ``returned_episode_returns``/``returned_episode_lengths``
        before resetting the running counters. These values (plus the global
        timestep and an ``returned_episode`` done flag) are added to ``info``.

        Args:
            key: PRNG key for the transition.
            state: Current :class:`LogEnvState`.
            action: Action to apply to the wrapped environment.
            params: Environment parameters (None uses the environment default).

        Returns:
            Tuple ``(obs, new LogEnvState, reward, done, info)`` with the
            logging fields added to ``info``.
        """
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class NavixGymnaxWrapper:
    """Adapter exposing a Navix environment through the Gymnax-style API.

    Wraps a Navix environment created from ``env_name`` and translates its
    reset/step/space API into the ``(obs, state, reward, done, info)`` tuple
    convention used by the other wrappers here.
    """

    def __init__(self, env_name):
        """Create the underlying Navix environment.

        Args:
            env_name: Name of the Navix environment (passed to ``navix.make``).
        """
        self._env = nx.make(env_name)

    def reset(self, key, params=None):
        """Reset the Navix environment.

        Args:
            key: PRNG key for the reset.
            params: Unused; accepted for Gymnax API compatibility.

        Returns:
            Tuple ``(observation, timestep)`` from the Navix reset.
        """
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        """Step the Navix environment.

        Args:
            key: Unused; accepted for Gymnax API compatibility.
            state: Current Navix timestep (used as the environment state).
            action: Action to apply.
            params: Unused; accepted for Gymnax API compatibility.

        Returns:
            Tuple ``(observation, timestep, reward, done, info)`` where ``info``
            is always an empty dict.
        """
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        """Return the flattened Navix observation space as a Gymnax Box.

        Args:
            params: Unused; accepted for Gymnax API compatibility.

        Returns:
            Gymnax Box with the Navix space's bounds, flattened shape, and dtype.
        """
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        """Return the Navix action space as a Gymnax Discrete space.

        Args:
            params: Unused; accepted for Gymnax API compatibility.

        Returns:
            Gymnax Discrete space with the number of Navix action categories.
        """
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


class ClipAction(GymnaxWrapper):
    """Wrapper that clips actions to a fixed range before stepping the environment."""

    def __init__(self, env, low=-1.0, high=1.0):
        """Wrap the environment and store the clipping bounds.

        Args:
            env: The environment being wrapped.
            low: Lower bound for clipping actions.
            high: Upper bound for clipping actions.
        """
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """Clip the action to ``[low, high]`` and step the wrapped environment.

        Args:
            key: PRNG key for the transition.
            state: Current environment state.
            action: Action to clip and apply.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, state, reward, done, info)`` from the wrapped environment.

        Note:
            TODO: In theory the clip bounds should come from the action space.
        """
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    """Wrapper that applies a function to the observation after reset and step."""

    def __init__(self, env, transform_obs):
        """Wrap the environment and store the observation transform.

        Args:
            env: The environment being wrapped.
            transform_obs: Callable applied to every observation.
        """
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        """Reset the wrapped environment and transform the initial observation.

        Args:
            key: PRNG key for the reset.
            params: Environment parameters.

        Returns:
            Tuple ``(transformed obs, state)``.
        """
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        """Step the wrapped environment and transform the returned observation.

        Args:
            key: PRNG key for the transition.
            state: Current environment state.
            action: Action to apply.
            params: Environment parameters.

        Returns:
            Tuple ``(transformed obs, state, reward, done, info)``.
        """
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    """Wrapper that applies a function to the reward after each step."""

    def __init__(self, env, transform_reward):
        """Wrap the environment and store the reward transform.

        Args:
            env: The environment being wrapped.
            transform_reward: Callable applied to every reward.
        """
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        """Step the wrapped environment and transform the returned reward.

        Args:
            key: PRNG key for the transition.
            state: Current environment state.
            action: Action to apply.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, state, transformed reward, done, info)``.
        """
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    """Wrapper that vectorizes the wrapped environment's reset and step over batched keys/states/actions."""

    def __init__(self, env):
        """Wrap the environment and install vmapped reset/step methods.

        Args:
            env: The environment being wrapped.
        """
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    """Running-normalization bookkeeping for :class:`NormalizeVecObservation`.

    Attributes:
        mean: Running per-dimension mean of the observations.
        var: Running per-dimension variance of the observations.
        count: Running count of observations seen so far.
        env_state: The wrapped environment's own state.
    """

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    """Wrapper that normalizes vectorized observations with a running mean/variance.

    Statistics are updated from the batch of observations at every reset/step
    (Welford-style parallel-variant merge) and observations are returned as
    ``(obs - mean) / sqrt(var + 1e-8)``.
    """

    def __init__(self, env):
        """Wrap the given environment (see :class:`GymnaxWrapper`)."""
        super().__init__(env)

    def reset(self, key, params=None):
        """Reset the wrapped environment and initialize/update the observation statistics.

        Args:
            key: PRNG key for the reset.
            params: Environment parameters.

        Returns:
            Tuple ``(normalized obs, NormalizeVecObsEnvState)``.
        """
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        """Step the wrapped environment and normalize the observation batch.

        The running mean/variance are first merged with the batch statistics of
        the new observations; the returned observation is the batch normalized
        with the updated statistics.

        Args:
            key: PRNG key for the transition.
            state: Current :class:`NormalizeVecObsEnvState`.
            action: Batched actions to apply.
            params: Environment parameters.

        Returns:
            Tuple ``(normalized obs, updated NormalizeVecObsEnvState, reward,
            done, info)``.
        """
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    """Running-normalization bookkeeping for :class:`NormalizeVecReward`.

    Attributes:
        mean: Running mean of the discounted episode returns.
        var: Running variance of the discounted episode returns.
        count: Running count of return samples seen so far.
        return_val: Current discounted episode return per environment.
        env_state: The wrapped environment's own state.
    """

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    """Wrapper that normalizes vectorized rewards by the running variance of the discounted returns.

    Rewards are accumulated per environment with discount factor ``gamma`` and
    each reward is divided by ``sqrt(var + 1e-8)`` of the running return
    statistics.
    """

    def __init__(self, env, gamma):
        """Wrap the environment and store the discount factor.

        Args:
            env: The environment being wrapped.
            gamma: Discount factor used to accumulate episode returns.
        """
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        """Reset the wrapped environment and zero the return statistics.

        Args:
            key: PRNG key for the reset.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, NormalizeVecRewEnvState)`` with mean 0, variance 1,
            and zeroed per-environment returns.
        """
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        """Step the wrapped environment and scale the reward by the running return std.

        The discounted episode return is updated per environment
        (``return_val * gamma * (1 - done) + reward``) and merged into the
        running mean/variance statistics; the reward is then divided by
        ``sqrt(var + 1e-8)`` before being returned.

        Args:
            key: PRNG key for the transition.
            state: Current :class:`NormalizeVecRewEnvState`.
            action: Batched actions to apply.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, updated NormalizeVecRewEnvState, scaled reward, done,
            info)``.
        """
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
