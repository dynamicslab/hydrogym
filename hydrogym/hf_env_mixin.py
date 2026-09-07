"""Shared Hugging-Face environment-config resolution logic (audit Task 3.1).

Four backends (jaxfluids, jax, maia, nek) carried near-identical private
copies of the same three methods:

    _setup_environment_data()      -- local ~/.cache/<namespace>/<env> lookup,
                                      falling back to HFDataManager
    _resolve_configuration_file()  -- None / abs path / ./relative / filename
    _find_configuration_file()     -- auto-detect config.yaml in the env dir

with only the cache-namespace string differing per backend. This module
extracts them into `HFEnvConfigMixin`, parameterized by the class-level
`HF_CACHE_NAMESPACE` (and, for data-manager construction, `SOLVER_TYPE`).
Backends migrate to the mixin one at a time to keep blast radius small;
the method bodies here are copied verbatim from the JAX-Fluids backend,
which is migrated first (Task 3.1).
"""

import glob
import os
from pathlib import Path
from typing import Optional

from hydrogym.data_manager import HFDataManager


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""


class HFEnvConfigMixin:
    """Environment-data download / config-file resolution shared by the
    HF-backed backends.

    Requires on the consuming class:
      - ``HF_CACHE_NAMESPACE``: per-backend cache directory name under
        ``~/.cache`` (e.g. "jaxfluidsgym", "jaxgym", "maiagym", "nekgym").
      - ``self.environment_name``, ``self.env_data_path``, and a
        ``self.data_manager`` (HFDataManager) instance before the
        resolution methods are called (i.e. ``_init_from_hf`` order:
        data_manager -> environment_name -> _setup_environment_data ->
        _resolve_configuration_file).
    """

    # Solver profile used by HFDataManager when no sentinel file is found
    # (offline / legacy data). Must be a key of data_manager.SOLVER_PROFILES.
    SOLVER_TYPE: Optional[str] = None

    # Optional prefix for the resolution-method log lines (e.g. "[NEK] " on
    # the Nek backend, which tags all its prints that way). Empty for the
    # backends whose copies printed bare messages.
    LOG_PREFIX: str = ""

    HF_CACHE_NAMESPACE: str = None

    def _make_data_manager(
        self,
        hf_repo_id: str,
        local_fallback_dir: Optional[str],
        use_clean_cache: bool,
        token: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> HFDataManager:
        """Build the backend's HFDataManager from the standard env_config keys."""
        return HFDataManager(
            repo_id=hf_repo_id,
            local_fallback_dir=local_fallback_dir,
            use_clean_cache=use_clean_cache,
            fallback_profile=self.SOLVER_TYPE,
            token=token,
            revision=revision,
        )

    def _setup_environment_data(self) -> str:
        """
        Download and setup environment data from HF Hub.

        First checks ~/.cache/<HF_CACHE_NAMESPACE>/ for local data, otherwise
        falls back to data_manager.

        Returns:
            Path to the local environment data directory.

        Raises:
            ConfigError: If environment data cannot be retrieved.
        """
        # Check cache directory first
        cache_dir = Path.home() / ".cache" / self.HF_CACHE_NAMESPACE / self.environment_name
        if cache_dir.exists() and cache_dir.is_dir():
            print(f"{self.LOG_PREFIX}Using cached environment data from: {cache_dir}")
            return str(cache_dir)

        # Fall back to data_manager if cache doesn't exist
        try:
            env_path = self.data_manager.get_environment_path(self.environment_name)
            print(f"{self.LOG_PREFIX}Using environment data from: {env_path}")
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
