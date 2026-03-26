"""
Core Jax Gym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

"""

import glob
import os
import jax
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeVar

import omegaconf
import toml
import chex
import jax.numpy as jnp
import navix as nx
import numpy as np
from flax import struct
from functools import partial
from gymnax.environments import environment, spaces

from hydrogym.data_manager import HFDataManager, SOLVER_PROFILES  # noqa: F401

class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class EnvParams(environment.EnvParams):
    config: dict 

EnvState = TypeVar("EnvState", bound=environment.EnvState)
    
class JAXFlowEnv(environment.Environment[EnvState, EnvParams]):
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
        self.hf_repo_id = env_config.get('hf_repo_id', 'dynamicslab/HydroGym-environments')
        self.local_fallback_dir = env_config.get('local_fallback_dir', None)
        self.use_clean_cache = env_config.get('use_clean_cache', True)

        self.data_manager = HFDataManager(
            repo_id=self.hf_repo_id,
            local_fallback_dir=self.local_fallback_dir,
            use_clean_cache=self.use_clean_cache
        )

        # Environment identification
        self.environment_name = env_config.get('environment_name')

        if not self.environment_name:
            raise ConfigError("'environment_name' must be specified in env_config")

        # Download/get environment configuration
        self.env_data_path = self._setup_environment_data()

        # Handle configuration file - support multiple ways of specifying it
        self.configuration_file = self._resolve_configuration_file(
            env_config.get('configuration_file')
        )

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

        self.runtime_property_file = os.path.join(self.env_data_path, 'properties_run.toml')

        self.num_substeps_per_iteration = self.cfg.jax.num_sim_substeps_per_actuation
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
        self.zLength = (
            self._get_property(self.runtime_property_file_data, "zLength")
            if self.nDim == 3
            else self.dX
        )
        self.Nz = (
            self._get_property(self.runtime_property_file_data, "Nz")
            if self.nDim == 3
            else self.dX
        )

    def _setup_environment_data(self) -> str:
        """
        Download and setup environment data from HF Hub.

        First checks ~/.cache/maiagym/ for local data, otherwise falls back to data_manager.

        Returns:
            Path to the local environment data directory.

        Raises:
            ConfigError: If environment data cannot be retrieved.
        """
        # Check cache directory first
        cache_dir = Path.home() / ".cache" / "maiagym" / self.environment_name
        if cache_dir.exists() and cache_dir.is_dir():
            print(f"Using cached environment data from: {cache_dir}")
            return str(cache_dir)

        # Fall back to data_manager if cache doesn't exist
        try:
            env_path = self.data_manager.get_environment_path(self.environment_name)
            print(f"Using environment data from: {env_path}")
            return env_path
        except Exception as e:
            raise ConfigError(f"Failed to setup environment data for {self.environment_name}: {e}")

    def _resolve_configuration_file(self, config_file_input: Optional[str]) -> Optional[str]:
        """
        Resolve configuration file path from various input formats.

        Args:
            config_file_input: Can be:
                - None: Auto-detect in HF environment
                - Absolute path: Use directly
                - Relative path starting with . or /: Use as-is
                - Just filename: Look in HF environment directory

        Returns:
            Absolute path to configuration file, or None if not found.

        Raises:
            ConfigError: If specified configuration file is not found.
        """
        # Case 1: No config file specified - try to find one
        if config_file_input is None:
            print("No config file specified, searching in environment directory...")
            return self._find_configuration_file()

        # Case 2: Absolute path provided
        if os.path.isabs(config_file_input):
            if os.path.exists(config_file_input):
                print(f"Using absolute path config file: {config_file_input}")
                return config_file_input
            else:
                raise ConfigError(f"Configuration file not found: {config_file_input}")

        # Case 3: Relative path from current directory (starts with ./ or ../)
        if config_file_input.startswith('./') or config_file_input.startswith('../'):
            abs_path = os.path.abspath(config_file_input)
            if os.path.exists(abs_path):
                print(f"Using config file from current directory: {abs_path}")
                return abs_path
            else:
                raise ConfigError(f"Configuration file not found: {abs_path}")

        # Case 4: Just a filename - look in multiple places
        # First check current directory
        if os.path.exists(config_file_input):
            abs_path = os.path.abspath(config_file_input)
            print(f"Using config file from current directory: {abs_path}")
            return abs_path

        # Then check HF environment directory
        env_config_path = os.path.join(self.env_data_path, config_file_input)
        if os.path.exists(env_config_path):
            print(f"Using config file from environment: {env_config_path}")
            return env_config_path

        raise ConfigError(
            f"Configuration file '{config_file_input}' not found in:\n"
            f"  - Current directory: {os.getcwd()}\n"
            f"  - Environment directory: {self.env_data_path}"
        )

    def _find_configuration_file(self) -> Optional[str]:
        """
        Auto-detect configuration file in the environment data directory.

        Returns:
            Path to configuration file, or None if not found.
        """
        # Look for specific configuration file names (most specific first)
        config_names = [
            'config.yaml',
            'environment_config.yaml',
            'env_config.yaml',
            'environment.yaml',
            f'{self.environment_name}.yaml',
        ]

        # Check exact names first
        for name in config_names:
            file_path = os.path.join(self.env_data_path, name)
            if os.path.exists(file_path):
                print(f"Auto-detected configuration file: {name}")
                return file_path

        # Then try patterns (but be specific - avoid catching property files)
        config_patterns = ['config_*.yaml', 'config_*.yml']

        for pattern in config_patterns:
            matches = glob.glob(os.path.join(self.env_data_path, pattern))
            if matches:
                print(f"Auto-detected configuration file: {os.path.basename(matches[0])}")
                return matches[0]

        # Not found
        print(f"WARNING: No configuration file auto-detected in {self.env_data_path}")
        if os.path.exists(self.env_data_path):
            print(f"Available files: {os.listdir(self.env_data_path)}")

        return None

    def _update_configuration_paths(self) -> None:
        """
        Update file paths in the configuration to point to downloaded environment data.

        This tells the CFD solver where to find the actual simulation files.
        """
        print(f"Environment data located at: {self.env_data_path}")

        # Update paths that the CFD solver needs
        path_mappings = {
            'maia.runtime_property_file': 'properties_run.toml',
        }

        for config_key, filename in path_mappings.items():
            file_path = os.path.join(self.env_data_path, filename)
            if os.path.exists(file_path):
                # Update the configuration with absolute path
                keys = config_key.split('.')
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
            'environment_name': self.environment_name,
            'local_cache_path': self.env_data_path,
            'configuration_file': self.configuration_file,
            'files': {}
        }

        # List all files in the environment
        if os.path.exists(self.env_data_path):
            for root, dirs, files in os.walk(self.env_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.env_data_path)
                    info['files'][rel_path] = {
                        'absolute_path': file_path,
                        'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    }

        return info

    @classmethod
    def create_from_hf_env(
        cls,
        environment_name: str,
        hf_repo_id: str = 'your-username/maiagym-envs',
        local_fallback_dir: Optional[str] = None,
        **kwargs
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
            'environment_name': environment_name,
            'hf_repo_id': hf_repo_id,
            'local_fallback_dir': local_fallback_dir,
            **kwargs
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
            self.environment_name,
            force_download=force_download
        )
        self._update_configuration_paths()

        # Reload configuration
        self.cfg = omegaconf.OmegaConf.load(self.configuration_file)
        
        
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        raise NotImplementedError

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return jnp.logical_or(state.terminal, state.time >= params.max_episode_steps)
    
    def step_env(self, key: chex.PRNGKey, state: EnvState, action: jnp.array, params: EnvParams):
        raise NotImplementedError

### NOTE: Issues with HF manager resulted in below minimal, HF-free class implementation for now. 
### Will re-integrate HF functionality.

class JAXFlowEnvBase(environment.Environment[EnvState, EnvParams]):
    """
        Base JAXFlowEnv without Hugging Face Hub integration. 
        Contains core environment interface methods required by Gymnax.
        
    """
    def __init__(self, env_config: Optional[dict] = None):
        self.env_config = env_config or {}

    def default_params(self) -> EnvParams:
        self.max_episode_steps = self.env_config.get('max_episode_steps', 1000)
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def action_space(self, params: Optional[EnvParams] = None):
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: EnvParams):
        return spaces.Box(
            low=params.min_obs,
            high=params.max_obs,
            shape=(params.obs_dim,),
        )

    def _clip_action(self, action: chex.Array, params: EnvParams) -> chex.Array:
        return jnp.clip(action, params.min_action, params.max_action)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return jnp.logical_or(state.terminal, state.time >= params.max_episode_steps)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        raise NotImplementedError

    def get_obs(
        self, state: EnvState, params: EnvParams, key=None
    ) -> chex.Array:
        raise NotImplementedError

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ):
        raise NotImplementedError


###############################################################

# BELOW CODE FROM PUREJAXRL REPO [1] WITH SLIGHT MODIFICATIONS 
# [1] https://github.com/luchris429/purejaxrl/ 

############################################################### 


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
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
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
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
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

class NavixGymnaxWrapper:
    def __init__(self, env_name):
        self._env = nx.make(env_name)

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        #action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
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
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
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
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
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
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
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