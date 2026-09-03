import os
from typing import Dict, Optional, Tuple, Union

from jaxfluids_rl.jxf_env import JAXFluidsEnv, RenderMode
from omegaconf import OmegaConf

from hydrogym.data_manager import HFDataManager
from hydrogym.hf_env_mixin import ConfigError, HFEnvConfigMixin


class JAXFluidsFlowEnv(HFEnvConfigMixin, JAXFluidsEnv):
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

    # Solver profile used by HFDataManager when no sentinel file is found
    # (offline / legacy data). Must be a key of data_manager.SOLVER_PROFILES.
    SOLVER_TYPE = "JAXFLUIDS"

    # Per-backend cache directory under ~/.cache
    HF_CACHE_NAMESPACE = "jaxfluidsgym"

    def _init_from_hf(self, env_config: dict) -> None:
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

        # Resolve and load configuration file
        self.configuration_file = self._resolve_configuration_file(env_config.get("configuration_file"))

        if not self.configuration_file:
            raise ConfigError(
                f"No configuration file found for environment '{self.environment_name}'. "
                f"Expected config.yaml in: {self.env_data_path}"
            )

        # Load configuration from HF
        self.conf = OmegaConf.load(self.configuration_file)
