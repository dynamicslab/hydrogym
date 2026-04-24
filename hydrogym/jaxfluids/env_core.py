import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from jaxfluids_rl.jxf_env import JAXFluidsEnv, RenderMode
from omegaconf import OmegaConf

from hydrogym.data_manager import HFDataManager


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class JAXFluidsFlowEnv(JAXFluidsEnv):
    """
    Base JAXFluidsFlowEnv with Hugging Face Hub integration for configuration management.

    Arguments:
        - environment_name: Required. Name of the environment.
        - hf_repo_id: Hugging Face repository (default: 'dynamicslab/HydroGym-environments')

        - use_clean_cache: Use clean cache directory (default: True)
            * True - Creates fresh workspace copy (recommended for production)
            * False - Uses cached workspace (faster for development/testing)
        - local_fallback_dir: Local directory for offline usage
        - configuration_file: Custom path to MAIA config.yaml (optional)

    :param JAXFluidsEnv: _description_
    :type JAXFluidsEnv: _type_
    """

    def _init_from_hf(self, env_config: dict) -> None:
        # Initialize HF data manager
        self.hf_repo_id = env_config.get("hf_repo_id", "dynamicslab/HydroGym-environments")
        self.local_fallback_dir = env_config.get("local_fallback_dir", None)
        self.use_clean_cache = env_config.get("use_clean_cache", True)

        self.data_manager = HFDataManager(
            repo_id=self.hf_repo_id,
            local_fallback_dir=self.local_fallback_dir,
            use_clean_cache=self.use_clean_cache,
            fallback_profile="JAXFLUIDS",
        )

        # Environment identification
        self.environment_name = env_config.get("environment_name")

        if not self.environment_name:
            raise ConfigError("'environment_name' must be specified in env_config")

        # Download/get environment configuration
        self.env_data_path = self._setup_environment_data()

        # Resolve and load configuration file
        self.configuration_file = self._resolve_configuration_file(env_config.get("configuration_file"))

        if not self.configuration_file:
            raise ConfigError(
                f"No configuration file found for environment '{self.environment_name}'. "
                f"Expected config.yaml in: {self.env_data_path}"
            )

        # Load configuration from HF
        self.conf = OmegaConf.load(self.configuration_file)

    def _setup_environment_data(self) -> str:
        """
        Download and setup environment data from HF Hub.

        First checks ~/.cache/jaxfluidsgym/ for local data, otherwise falls back to data_manager.

        Returns:
            Path to the local environment data directory.

        Raises:
            ConfigError: If environment data cannot be retrieved.
        """
        # Check cache directory first
        cache_dir = Path.home() / ".cache" / "jaxfluidsgym" / self.environment_name
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
        if config_file_input.startswith("./") or config_file_input.startswith("../"):
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
            "config.yaml",
            "environment_config.yaml",
            "env_config.yaml",
            "environment.yaml",
            f"{self.environment_name}.yaml",
        ]

        # Check exact names first
        for name in config_names:
            file_path = os.path.join(self.env_data_path, name)
            if os.path.exists(file_path):
                print(f"Auto-detected configuration file: {name}")
                return file_path

        # Then try patterns (but be specific - avoid catching property files)
        config_patterns = ["config_*.yaml", "config_*.yml"]

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
