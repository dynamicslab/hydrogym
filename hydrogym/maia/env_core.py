"""
Core MaiaGym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import omegaconf
import toml
from einops import rearrange
from mpi4py import MPI

from hydrogym.maia.hf_data_manager import HFDataManager
from hydrogym.maia.mpmd_interface import MaiaInterface


class ConfigError(Exception):
  """Exception raised for configuration-related errors."""

  pass


class MaiaFlowEnv(gym.Env):
  """
    Base MaiaFlowEnv with Hugging Face Hub integration for configuration management.

    This environment provides a Gymnasium-compatible interface for CFD simulations
    using the m-AIA solver. It handles:
    - Environment data management via Hugging Face Hub
    - Configuration file resolution and loading
    - MPI communication with the CFD solver
    - Observation normalization strategies
    - Action space configuration

    Attributes:
        environment_name: Name of the CFD environment configuration.
        env_data_path: Path to the local environment data directory.
        cfg: OmegaConf configuration object.
        maiaInterface: MPI interface for communication with m-AIA.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
    """

  SOLVER_TYPE: str = 'MAIA_LB'

  def __init__(self, env_config: Dict):
    """
        Initialize the MaiaFlowEnv environment.

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
    self.hf_repo_id = env_config.get('hf_repo_id',
                                     'dynamicslab/HydroGym-environments')
    self.local_fallback_dir = env_config.get('local_fallback_dir', None)
    self.use_clean_cache = env_config.get('use_clean_cache', True)

    self.data_manager = HFDataManager(
        repo_id=self.hf_repo_id,
        local_fallback_dir=self.local_fallback_dir,
        use_clean_cache=self.use_clean_cache,
        fallback_profile=self.SOLVER_TYPE,
    )

    # Environment identification
    self.environment_name = env_config.get('environment_name')

    if not self.environment_name:
      raise ConfigError("'environment_name' must be specified in env_config")

    # Download/get environment configuration
    self.env_data_path = self._setup_environment_data()

    # Handle configuration file - support multiple ways of specifying it
    self.configuration_file = self._resolve_configuration_file(
        env_config.get('configuration_file'))

    if not self.configuration_file:
      raise ConfigError(
          f"No configuration file found. Please either:\n"
          f"1. Provide configuration_file='/path/to/config.yaml' in env_config\n"
          f"2. Add a config.yaml file to the HF environment: {self.env_data_path}\n"
          f"Available files: {os.listdir(self.env_data_path) if os.path.exists(self.env_data_path) else 'N/A'}"
      )

    # Standard initialization
    self.is_testing = env_config.get('is_testing', False)
    self.probe_locations = env_config.get('probe_locations')
    self.obs_normalization_strategy = env_config.get(
        'obs_normalization_strategy', 'none')

    valid_strategies = ['U_inf', 'probewise_mean_std', 'none', 'customized']
    if self.obs_normalization_strategy not in valid_strategies:
      raise ConfigError(
          f"Invalid obs_normalization_strategy: '{self.obs_normalization_strategy}'. "
          f"Must be one of: {valid_strategies}")

    if self.obs_normalization_strategy == 'customized':
      self.obs_loc = env_config.get('obs_loc')
      self.obs_scale = env_config.get('obs_scale')
      if self.obs_loc is None or self.obs_scale is None:
        raise ConfigError(
            "obs_normalization_strategy='customized' requires both 'obs_loc' and 'obs_scale' "
            "to be provided in env_config")

    # Load configuration
    self.cfg = omegaconf.OmegaConf.load(self.configuration_file)

    # Update paths in configuration to use downloaded data
    self._update_configuration_paths()

    self.runtime_property_file = os.path.join(self.env_data_path,
                                              'properties_run.toml')

    self.num_substeps_per_iteration = self.cfg.maia.num_sim_substeps_per_actuation
    self.observation_type = self.cfg.maia.observation_type
    self.max_episode_steps = self.cfg.env.max_episode_steps
    self.num_inputs = self.cfg.maia.num_action_inputs * self.cfg.env.n_agents
    self.MAX_CONTROL = self.cfg.maia.max_control
    self.render = self.cfg.maia.render

    # Load reward scaling with fallback
    try:
      self.omega = self.cfg.maia.omega
    except omegaconf.errors.ConfigAttributeError:
      self.omega = 0.0

    # Read property file and extract parameters
    self.runtime_property_file_data = self._read_property_file(
        self.runtime_property_file)
    self.Ma = self._get_property(self.runtime_property_file_data, "Ma")
    self.maxRfnmntLvl = self._get_property(self.runtime_property_file_data,
                                           "maxRfnmntLvl")
    self.reductionFactor = self._get_property(self.runtime_property_file_data,
                                              "reductionFactor")
    self.domainLength = self._get_property(self.runtime_property_file_data,
                                           "domainLength")
    self.referenceLength = self._get_property(self.runtime_property_file_data,
                                              "referenceLength")
    self.nDim = self._get_property(self.runtime_property_file_data, "nDim")
    self.dX = self.reductionFactor * self.domainLength / (2**self.maxRfnmntLvl)
    self.bcId = self._get_property(self.runtime_property_file_data,
                                   "lbBndCndIdx")
    self.Re = self._get_property(self.runtime_property_file_data, "Re")
    self.zLength = (
        self._get_property(self.runtime_property_file_data, "zLength")
        if self.nDim == 3 else self.dX)

    # Initialize MPI communication
    self.comm_world = MPI.COMM_WORLD
    self.maiaInterface = MaiaInterface(self.nDim)
    self.maiaInterface.init_comm(self.comm_world)
    print('Python communicator initialized', flush=True)

    if self.Re != self.cfg.maia.Re:
      raise ConfigError(
          f"Re numbers of configuration file (Re={self.cfg.maia.Re}) and "
          f"property file (Re={self.Re}) do not match! Adjustment required!")

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
      raise ConfigError(
          f"Failed to setup environment data for {self.environment_name}: {e}")

  def _resolve_configuration_file(
      self, config_file_input: Optional[str]) -> Optional[str]:
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
    if config_file_input.startswith('./') or config_file_input.startswith(
        '../'):
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
        f"  - Environment directory: {self.env_data_path}")

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
        print(
            f"Auto-detected configuration file: {os.path.basename(matches[0])}")
        return matches[0]

    # Not found
    print(
        f"WARNING: No configuration file auto-detected in {self.env_data_path}")
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

  def configure_observations(self) -> None:
    """
        Configure the number of observations based on observation type and probes.

        Sets self.num_outputs based on which observation types are enabled
        (forces, u, v, w, rho, p) and the number of probes.
        """
    self.num_outputs = 0
    self.num_probes = int(len(self.probe_locations) / self.nDim)

    if 'forces' in self.observation_type:
      self.num_outputs += self.nDim * len(self.bcId)
    if 'u' in self.observation_type:
      self.num_outputs += self.num_probes
    if 'v' in self.observation_type:
      self.num_outputs += self.num_probes
    if 'w' in self.observation_type:
      self.num_outputs += self.num_probes
    if 'rho' in self.observation_type:
      self.num_outputs += self.num_probes
    if 'p' in self.observation_type:
      self.num_outputs += self.num_probes

  def setup_normalization(self) -> None:
    """
        Setup observation normalization factors based on the selected strategy.

        Strategies:
            - 'U_inf': Normalize by freestream velocity
            - 'none': No normalization (loc=0, scale=1)
            - 'probewise_mean_std': Compute mean/std from simulation episodes
            - 'customized': Use user-provided loc and scale values

        Raises:
            ConfigError: If normalization configuration is invalid.
        """
    if self.obs_normalization_strategy == 'U_inf':
      obs_loc, obs_scale = [], []

      if 'u' in self.observation_type:
        obs_loc.append([0.0] * self.noProbes)
        obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)
      if 'v' in self.observation_type:
        obs_loc.append([0.0] * self.noProbes)
        obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)

      if self.nDim == 2:
        if 'rho' in self.observation_type:
          obs_loc.append([0.0] * self.noProbes)
          obs_scale.append([self.cfg.maia.rho_inf] * self.noProbes)
        if 'p' in self.observation_type:
          dynamic_pressure = 0.5 * self.cfg.maia.rho_inf * self.cfg.maia.U_inf**2
          obs_loc.append([0.0] * self.noProbes)
          obs_scale.append([dynamic_pressure] * self.noProbes)
      elif self.nDim == 3:
        if 'w' in self.observation_type:
          obs_loc.append([0.0] * self.noProbes)
          obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)
        if 'rho' in self.observation_type:
          obs_loc.append([0.0] * self.noProbes)
          obs_scale.append([self.cfg.maia.rho_inf] * self.noProbes)
        if 'p' in self.observation_type:
          dynamic_pressure = 0.5 * self.cfg.maia.rho_inf * self.cfg.maia.U_inf**2
          obs_loc.append([0.0] * self.noProbes)
          obs_scale.append([dynamic_pressure] * self.noProbes)
      else:
        print(f'WARNING: nDim = {self.nDim} > 3. Something must be wrong!')

      if 'forces' in self.observation_type:
        for _ in range(len(self.bcId)):
          obs_loc.append([0.0] * self.nDim)
          obs_scale.append([1.0] * self.nDim)

      self.obs_loc = np.concatenate(obs_loc)
      self.obs_scale = np.concatenate(obs_scale)

    elif self.obs_normalization_strategy == 'none':
      self.obs_loc = [0.0] * self.num_outputs
      self.obs_scale = [1.0] * self.num_outputs

    elif self.obs_normalization_strategy == 'probewise_mean_std':
      print('WARNING: Selected Obs Normalization strategy: probewise_mean_std')
      print('Computing normalization factors now...')
      print('If pre-computed normalization factors exist, select:')
      print(
          "obs_normalization_strategy = 'customized' and provide normalization factors!"
      )
      self.compute_normalization_factors()
      print('Computed loc values:', self.obs_loc.tolist(), flush=True)
      print('Computed scale values:', self.obs_scale.tolist(), flush=True)

    elif self.obs_normalization_strategy == 'customized':
      if self.obs_scale is not None and self.obs_loc is not None:
        if len(self.obs_loc) == self.num_outputs and len(
            self.obs_scale) == self.num_outputs:
          print('Using customized normalization values', flush=True)
        else:
          raise ConfigError(f"Customized normalization dimensions mismatch: "
                            f"obs_loc has {len(self.obs_loc)} elements, "
                            f"obs_scale has {len(self.obs_scale)} elements, "
                            f"but num_outputs is {self.num_outputs}")
    else:
      raise ConfigError(
          f"Invalid obs_normalization_strategy: '{self.obs_normalization_strategy}'. "
          f"This should have been caught earlier.")

  def compute_normalization_factors(self, zero_actuation: bool = False) -> None:
    """
        Compute normalization factors for observations by running simulation episodes.

        Runs several episodes with random (or zero) actuation to collect probe data,
        then computes mean and standard deviation for normalization.

        Args:
            zero_actuation: If True, use zero actuation. Otherwise, use random actuation.
        """
    obs_norm_episodes = 5
    probes = np.zeros(
        shape=(obs_norm_episodes * self.max_episode_steps,
               int(self.noProbes * self.noProbeVars)))

    for i in range(obs_norm_episodes * self.max_episode_steps):
      if i % self.max_episode_steps == 0 and i > 0:
        # Reset environment
        self.maiaInterface.runTimeSteps(1)
        self.maiaInterface.reinit()
        self.maiaInterface.setControlProperties(
            self.convert_action(action=np.zeros(shape=self.num_inputs)))
        self.maiaInterface.continueRun()

      self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)

      if zero_actuation:
        self.maiaInterface.setControlProperties(
            self.convert_action(action=np.zeros(shape=self.num_inputs)))
      else:
        self.maiaInterface.setControlProperties(
            self.convert_action(
                action=np.random.uniform(
                    -self.MAX_CONTROL, self.MAX_CONTROL, size=self.num_inputs)))

      probes[i, :] = self.maiaInterface.getProbeData(self.probe_locations)
      self.maiaInterface.continueRun()

    loc = np.mean(probes, axis=0)
    scale = np.std(probes, axis=0)

    loc = rearrange(loc, '(n p) -> n p', n=self.noProbes)
    scale = rearrange(scale, '(n p) -> n p', n=self.noProbes)

    self.obs_loc, self.obs_scale = [], []

    if 'u' in self.observation_type:
      self.obs_loc.append(loc[:, 0])
      self.obs_scale.append(scale[:, 0])
    if 'v' in self.observation_type:
      self.obs_loc.append(loc[:, 1])
      self.obs_scale.append(scale[:, 1])

    if self.nDim == 2:
      if 'rho' in self.observation_type:
        self.obs_loc.append(loc[:, 2])
        self.obs_scale.append(scale[:, 2])
      if 'p' in self.observation_type:
        self.obs_loc.append(loc[:, 3])
        self.obs_scale.append(scale[:, 3])
    elif self.nDim == 3:
      if 'w' in self.observation_type:
        self.obs_loc.append(loc[:, 2])
        self.obs_scale.append(scale[:, 2])
      if 'rho' in self.observation_type:
        self.obs_loc.append(loc[:, 3])
        self.obs_scale.append(scale[:, 3])
      if 'p' in self.observation_type:
        self.obs_loc.append(loc[:, 4])
        self.obs_scale.append(scale[:, 4])

    self.obs_loc = np.concatenate(self.obs_loc)
    self.obs_scale = np.concatenate(self.obs_scale)

  @classmethod
  def create_from_hf_env(cls,
                         environment_name: str,
                         hf_repo_id: str = 'your-username/maiagym-envs',
                         local_fallback_dir: Optional[str] = None,
                         **kwargs):
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
        self.environment_name, force_download=force_download)
    self._update_configuration_paths()

    # Reload configuration
    self.cfg = omegaconf.OmegaConf.load(self.configuration_file)

  def step(
      self,
      action: Optional[np.ndarray] = None
  ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
        Advance the state of the environment by one step.

        Args:
            action: Action array with values in [-1, 1], scaled by MAX_CONTROL.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
    action = [a * self.MAX_CONTROL for a in action]

    self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)
    self.maiaInterface.setControlProperties(self.convert_action(action=action))

    self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
    self.probeData = rearrange(self.probeData, '(n p) -> n p', n=self.noProbes)

    self.obs = []
    if 'u' in self.observation_type:
      self.obs.append(self.probeData[:, 0])
    if 'v' in self.observation_type:
      self.obs.append(self.probeData[:, 1])

    if self.nDim == 2:
      if 'rho' in self.observation_type:
        self.obs.append(self.probeData[:, 2])
      if 'p' in self.observation_type:
        self.obs.append(self.probeData[:, 3])
    elif self.nDim == 3:
      if 'w' in self.observation_type:
        self.obs.append(self.probeData[:, 2])
      if 'rho' in self.observation_type:
        self.obs.append(self.probeData[:, 3])
      if 'p' in self.observation_type:
        self.obs.append(self.probeData[:, 4])
    else:
      print(f'WARNING: nDim = {self.nDim} > 3. Something must be wrong!')

    if 'forces' in self.observation_type:
      forces = []
      for bc_id in self.bcId:
        forces.append(self.maiaInterface.getForce(bc_id))
      self.obs.append(np.stack(forces).flatten())

    self.obs = np.concatenate(self.obs)
    reward, obj_dict = self.get_reward()
    self.obs = (self.obs - self.obs_loc) / self.obs_scale

    self.iter += 1
    done = self.check_complete()
    info = {}

    self.maiaInterface.continueRun()

    return self.obs, reward, bool(done), bool(done), info

  def reset(self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options.

        Returns:
            Tuple of (initial_observation, info).
        """
    print('Resetting environment', flush=True)

    self.maiaInterface.runTimeSteps(1)
    self.maiaInterface.reinit()

    self.maiaInterface.setControlProperties(
        self.convert_action(action=np.zeros(shape=self.num_inputs)))
    self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
    self.probeData = rearrange(self.probeData, '(n p) -> n p', n=self.noProbes)

    self.obs = []
    if 'u' in self.observation_type:
      self.obs.append(self.probeData[:, 0])
    if 'v' in self.observation_type:
      self.obs.append(self.probeData[:, 1])

    if self.nDim == 2:
      if 'rho' in self.observation_type:
        self.obs.append(self.probeData[:, 2])
      if 'p' in self.observation_type:
        self.obs.append(self.probeData[:, 3])
    elif self.nDim == 3:
      if 'w' in self.observation_type:
        self.obs.append(self.probeData[:, 2])
      if 'rho' in self.observation_type:
        self.obs.append(self.probeData[:, 3])
      if 'p' in self.observation_type:
        self.obs.append(self.probeData[:, 4])
    else:
      print(f'WARNING: nDim = {self.nDim} > 3. Something must be wrong!')

    if 'forces' in self.observation_type:
      forces = []
      for bc_id in self.bcId:
        forces.append(self.maiaInterface.getForce(bc_id))
      self.obs.append(np.stack(forces).flatten())

    self.obs = np.concatenate(self.obs)
    self.obs = (self.obs - self.obs_loc) / self.obs_scale

    self.iter = 0
    info = {}

    self.maiaInterface.continueRun()

    return self.obs, info

  def compute_nondim_coefficients(self,
                                  forces: np.ndarray,
                                  density: float = 1.0,
                                  referenceVelocity: float = 0.1 / np.sqrt(3),
                                  projectionLength: float = 20.0) -> np.ndarray:
    """
        Compute non-dimensionalized force coefficients.

        Args:
            forces: Force array from the CFD solver.
            density: Reference density.
            referenceVelocity: Reference velocity.
            projectionLength: Reference projection length.

        Returns:
            Non-dimensional force coefficients.
        """
    force_coefficients = (2 * forces) / (
        density * referenceVelocity**2 * projectionLength)
    return force_coefficients

  def convert_action(self, action: np.ndarray) -> List:
    """
        Convert RL action to CFD actuation format.

        This method should be overridden by subclasses to implement
        environment-specific action conversion.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
    pass

  def get_reward(self) -> Tuple[float, Dict]:
    """
        Compute the reward for the current state.

        This method should be overridden by subclasses to implement
        environment-specific reward computation.

        Returns:
            Tuple of (reward, objective_dict).
        """
    pass

  def check_complete(self) -> bool:
    """
        Check if the episode is complete.

        Returns:
            True if the episode has reached max_episode_steps.
        """
    return self.iter > self.max_episode_steps

  def close(self) -> None:
    """
        Close the environment and signal maia to finish.

        This sends a finish signal to the maia solver to properly
        terminate the MPMD coupling.
        """
    print(
        'ENV: Closing environment and signaling maia to finish...', flush=True)
    try:
      self.maiaInterface.finishRun()
      print('ENV: Finish signal sent to maia', flush=True)
    except Exception as e:
      print(f'ENV: Error during close: {e}', flush=True)

  def set_observation_action_spaces(self) -> None:
    """
        Set up the observation and action spaces for Gymnasium compatibility.
        """
    self.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.num_outputs,),
        dtype=float,
    )

    self.action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(self.num_inputs,),
        dtype=float,
    )

  def configure_probe_dimensions(self) -> None:
    """
        Configure probe dimensions based on simulation dimensionality.

        Sets noProbeVars (number of variables per probe) and noProbes (total probes).
        """
    if self.nDim == 2:
      self.noProbeVars = 4  # u, v, rho, p
    elif self.nDim == 3:
      self.noProbeVars = 5  # u, v, w, rho, p

    self.noProbes = len(self.probe_locations) // self.nDim

  def _read_property_file(self, property_file_path: str) -> Dict:
    """
        Read a TOML property file.

        Args:
            property_file_path: Path to the TOML file.

        Returns:
            Dictionary containing the parsed TOML data.
        """
    with open(property_file_path, 'r') as f:
      property_file_data = toml.load(f)
    return property_file_data

  def _get_property(self, property_file_data: Dict, key: Union[str, List[str]]):
    """
        Get a property value from the property file data.

        Args:
            property_file_data: Dictionary from the property file.
            key: Property key, either a string or list of strings for nested access.

        Returns:
            The property value.
        """
    if isinstance(key, list):
      return property_file_data[key[0]][key[1]]
    else:
      return property_file_data[key]

  def _update_property(self, property_file_data: Dict,
                       key: Union[str, List[str]], value) -> None:
    """
        Update a property value in the property file data.

        Args:
            property_file_data: Dictionary from the property file.
            key: Property key, either a string or list of strings for nested access.
            value: New value for the property.
        """
    if isinstance(key, list):
      property_file_data[key[0]][key[1]] = value
    else:
      property_file_data[key] = value

  def _write_property_file(self, property_file_data: Dict,
                           property_file_path: str) -> None:
    """
        Write property file data back to a TOML file.

        Args:
            property_file_data: Dictionary to write.
            property_file_path: Path to the output file.
        """
    with open(property_file_path, 'w') as f:
      toml.dump(property_file_data, f)


# Environment Type Registry
_ENVIRONMENT_REGISTRY: Dict[str, type] = {}


def register_environment(env_prefix: str, env_class: type) -> None:
  """
    Register an environment class for automatic detection.

    Args:
        env_prefix: Environment prefix (e.g., 'Cylinder').
        env_class: Environment class to register.
    """
  _ENVIRONMENT_REGISTRY[env_prefix] = env_class


def from_hf(environment_name: str,
            hf_repo_id: str = 'dynamicslab/HydroGym-environments',
            **kwargs) -> MaiaFlowEnv:
  """
    Factory function to create any MaiaGym environment from Hugging Face Hub.

    Args:
        environment_name: Name of environment (e.g., 'Cylinder_2D_Re200').
        hf_repo_id: Hugging Face repository ID.
        **kwargs: Additional environment configuration parameters.

    Returns:
        Configured environment instance.

    Raises:
        ValueError: If environment name format is invalid or type is unknown.
        ConfigError: If environment creation fails.

    Examples:
        >>> env = maiaGym.from_hf('Cylinder_2D_Re200')
        >>> env = maiaGym.from_hf('RotaryCylinder_2D_Re1000', obs_loc=custom_loc)
        >>> env = maiaGym.from_hf('Cavity_2D_Re4140')
    """
  # Parse environment type from name
  if '_' not in environment_name:
    available_types = list(_ENVIRONMENT_REGISTRY.keys())
    raise ValueError(
        f"Invalid environment name '{environment_name}'. "
        f"Expected format: 'TYPE_PARAMETERS' (e.g., 'Cylinder_2D_Re200'). "
        f"Available types: {available_types}")

  env_type = environment_name.split('_', 1)[0]

  # Look up environment class
  if env_type not in _ENVIRONMENT_REGISTRY:
    available_types = list(_ENVIRONMENT_REGISTRY.keys())
    raise ValueError(
        f"Unknown environment type '{env_type}' from name '{environment_name}'. "
        f"Available types: {available_types}. "
        f"If this is a new environment type, make sure the corresponding "
        f"HF environment class is imported and registered.")

  env_class = _ENVIRONMENT_REGISTRY[env_type]

  # Create environment configuration
  env_config = {
      'environment_name': environment_name,
      'hf_repo_id': hf_repo_id,
      **kwargs
  }

  try:
    return env_class(env_config)
  except Exception as e:
    raise ConfigError(
        f"Failed to create environment '{environment_name}' of type '{env_type}': {e}"
    ) from e


def list_available_environments(
    hf_repo_id: str = 'dynamicslab/HydroGym-environments') -> List[str]:
  """
    List all available environments from HF Hub.

    Args:
        hf_repo_id: Hugging Face repository ID.

    Returns:
        List of environment names.
    """
  data_manager = HFDataManager(repo_id=hf_repo_id)
  return data_manager.get_available_environments()


def list_registered_types() -> Dict[str, type]:
  """
    List all registered environment types.

    Returns:
        Dictionary mapping environment types to their classes.
    """
  return dict(_ENVIRONMENT_REGISTRY)
