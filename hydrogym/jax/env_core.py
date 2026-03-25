"""
Core Jax Gym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeVar

import omegaconf
import toml
import chex
import jax.numpy as jnp
import numpy as np
from flax import struct
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
