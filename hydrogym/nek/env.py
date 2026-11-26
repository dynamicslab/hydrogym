"""
Core Nek5000 environment with Gymnasium interface.
Single-agent with array-based observations/actions.

Supports two initialization patterns:
1. Hugging Face pattern (recommended): env = NekEnv.from_hf('EnvName', nproc=10)
2. Legacy pattern: env = NekEnv(conf=config_obj)
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from mpi4py import MPI
from omegaconf import OmegaConf

from hydrogym.core_external import ExternalProcessEnvMixin, mpi_split  # noqa: F401  (mpi_split re-exported)
from hydrogym.data_manager import HFDataManager
from hydrogym.hf_env_mixin import ConfigError, HFEnvConfigMixin
from hydrogym.power import POWER_COSTS, PowerConfig, power_info

from .configs import Config
from .nek_lib.lglnodes import lglnodes
from .nek_lib.nek_utils import POWER_MODES, rea_mismatches, remove_sch
from .nek_lib.reward_logger import RewardLogger


class NekDivergenceError(RuntimeError):
    """Raised when the Nek simulation's CFL number blows up during evolve.

    The failure is catchable by RL loops and gym wrappers. By the time this is raised the solver side has
    already been shut down cleanly (TERMN sent, intercomm freed, MPI
    finalized) -- the environment is not usable afterwards.
    """


class NekEnv(HFEnvConfigMixin, ExternalProcessEnvMixin, gym.Env):
    """
    Core Nek5000 environment with Gymnasium interface.

    This is a single-agent environment where the agent controls multiple
    actuators (control points) on the mesh. Observations and actions are
    flat arrays representing all actuators.

    Supports two initialization patterns:

    1. Hugging Face pattern (recommended):
        env = NekEnv.from_hf(
            'TCFmini_3D_Re180',
            nproc=10,
            hostfile='',
        )

    2. Legacy pattern (deprecated):
        conf = OmegaConf.load('config.yaml')
        env = NekEnv(conf=conf)

    Args (Hugging Face pattern via env_config):
      environment_name: Name of environment on HuggingFace
      nproc: Number of MPI workers for Nek (required)
      hostfile: MPI hostfile path (default: '')
      mpi_bind_to: MPI bind policy passed to mpirun via MPI.Info
          (default: 'none' = unbound; e.g. 'core' binds ranks to cores)
      hf_repo_id: HuggingFace repository (default: 'dynamicslab/HydroGym-environments')
      use_clean_cache: Use fresh workspace (default: True)
      local_fallback_dir: Local directory for offline usage
      configuration_file: Override config file path
      run_root: Root directory for outputs (default: '.')
      run_name: Name for this run (default: '' = no subdirectory, use run_root directly)
      reward_agg: Reward aggregation method ("mean", "sum", or "median") - legacy
          name, kept working
      reward_aggregation: Same as reward_agg, primary name (matches core.py /
          Firedrake); takes precedence when both are given
      ... (runtime overrides for config parameters)

    Args (Legacy pattern):
      conf: Configuration object (OmegaConf)
      run_root: Root directory for run outputs (default: '.')
      run_name: Name for this run (default: no subdirectory, use run_root directly)
      reward_agg: How to aggregate per-actuator rewards ("mean", "sum", or "median")
    """

    metadata = {"render_modes": ["human"]}
    SOLVER_TYPE = "NEK5000"

    # Per-backend cache directory under ~/.cache, and log prefix for the
    # HFEnvConfigMixin resolution messages (Nek tags all its prints).
    HF_CACHE_NAMESPACE = "nekgym"
    LOG_PREFIX = "[NEK] "
    # MPI rank-binding policy passed to mpirun via MPI.Info ("bind_to").
    # Overridable via the `mpi_bind_to` env_config key (Hugging Face pattern).
    DEFAULT_MPI_BIND_TO = "none"

    def __init__(
        self,
        conf: Optional[Config] = None,
        env_config: Optional[Dict] = None,
        run_root: str = ".",
        run_name: Optional[str] = None,
        reward_agg: str = "mean",
        reward_aggregation: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize NekEnv with either legacy conf or MAIA env_config pattern.

        Args:
          conf: Legacy OmegaConf object (deprecated)
          env_config: Configuration dict of the Hugging Face pattern (recommended)
          run_root: Output directory root (default: '.')
          run_name: Run name; None or '' = no subdirectory, use run_root directly
          reward_agg: Reward aggregation method ('mean', 'sum' or 'median'; legacy name)
          reward_aggregation: Same as reward_agg (takes precedence when both are given)
          **kwargs: Additional parameters (for backward compatibility)

        Raises:
          ValueError: If both 'conf' and 'env_config' are given, or an invalid
            reward aggregation is requested.
          UserWarning: If **kwargs is non-empty -- these are accepted for
            backward compatibility but have NO effect; runtime overrides
            belong in env_config (see NekEnv.from_hf).
        """
        # Determine which API is being used
        if conf is not None and env_config is not None:
            raise ValueError("Cannot provide both 'conf' and 'env_config'. Use one or the other.")

        if kwargs:
            warnings.warn(
                f"NekEnv.__init__ got unsupported keyword argument(s) {sorted(kwargs)}; "
                "they have no effect. Runtime configuration overrides belong in "
                "env_config (see NekEnv.from_hf).",
                stacklevel=2,
            )

        # ``reward_aggregation`` is the primary name (matching core.py's
        # actuation_config / Firedrake); ``reward_agg`` is the legacy Nek
        # name and keeps working unchanged.
        if reward_aggregation is not None:
            if reward_agg != "mean" and reward_agg != reward_aggregation:
                warnings.warn(
                    "Both reward_agg and reward_aggregation were provided with different values; "
                    "using reward_aggregation",
                    stacklevel=2,
                )
            reward_agg = reward_aggregation
        if reward_agg not in ("mean", "sum", "median"):
            raise ValueError(f"reward aggregation must be 'mean', 'sum', or 'median', got {reward_agg!r}")
        self.reward_agg = reward_agg

        if conf is not None:
            # Legacy API
            warnings.warn(
                "Passing 'conf' is deprecated. Use the Hugging Face pattern: env = NekEnv.from_hf('EnvName', nproc=10)",
                DeprecationWarning,
                stacklevel=2,
            )
            self._init_from_legacy(conf, run_root, run_name, reward_agg)

        elif env_config is not None:
            # MAIA API
            self._init_from_hf(env_config, run_root, run_name, reward_agg)

        else:
            raise ValueError(
                "Must provide either 'conf' (legacy) or 'env_config' (Hugging Face pattern). "
                "Recommended: use NekEnv.from_hf('EnvName', nproc=10)"
            )

    @classmethod
    def from_hf(cls, environment_name: str, nproc: int, hostfile: str = "", **kwargs):
        """
        Create environment from HuggingFace Hub (Hugging Face pattern).

        Args:
          environment_name: Name of the environment (e.g., 'TCFmini_3D_Re180')
          nproc: Number of MPI workers for Nek (required)
          hostfile: MPI hostfile path (default: '')
          **kwargs: Additional env_config parameters:
            - hf_repo_id: HF repository (default: 'dynamicslab/HydroGym-environments')
            - use_clean_cache: Fresh workspace (default: True)
            - local_fallback_dir: Local directory
            - configuration_file: Override config path
            - run_root: Output directory (default: '.')
            - run_name: Run name (default: no subdirectory, use run_root directly)
            - reward_agg: 'mean', 'sum', or 'median' (default: 'mean'); legacy name
            - reward_aggregation: same as reward_agg, primary name (takes precedence)
            - normalize_input: Override normalization strategy (shorthand)
            - nb_interactions: Override episode length (shorthand)
            - random_init: Override IC randomization (shorthand)
            - rescale_actions: Override action rescaling (shorthand)
            - rew_mode: Override reward mode (shorthand)
            - '<section>.<param>': Override ANY existing config-tree entry by
              dotted path (e.g. 'simulation.walltime'); unknown paths raise
              ConfigError

        Returns:
          NekEnv instance

        Example:
          env = NekEnv.from_hf('TCFmini_3D_Re180', nproc=10)

          env = NekEnv.from_hf(
              'TCFmini_3D_Re180',
              nproc=10,
              hostfile='hosts.txt',
              use_clean_cache=True,
              normalize_input='utau',
          )
        """
        env_config = {
            "environment_name": environment_name,
            "nproc": nproc,
            "hostfile": hostfile,
            **kwargs,
        }
        return cls(env_config=env_config)

    def _init_from_legacy(self, conf: Config, run_root: str, run_name: Optional[str], reward_agg: str):
        """Initialize using legacy conf object."""
        self.conf = conf
        self.reward_agg = reward_agg
        self.nproc = conf.simulation.nproc
        self.hostfile = conf.simulation.hostfile

        # Create run folder
        if run_name is None:
            run_name = ""

        # If run_name is empty string, use run_root directly (no subdirectory)
        if run_name:
            self.run_folder = os.path.join(run_root, run_name)
        else:
            self.run_folder = run_root

        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder, exist_ok=True)

        # MPI communicator required by Nek
        comm_world = MPI.COMM_WORLD
        self.sub_comm = self._split_mpmd_comm(comm_world, nproc=self.nproc)

        # Initialize the environment (this sets n_actuators, obs_per_actuator, etc.)
        self._initialize()

    def _init_from_hf(self, env_config: Dict, run_root: str, run_name: Optional[str], reward_agg: str):
        """Initialize using Hugging Face pattern from HuggingFace."""
        # Validate required parameters
        if "environment_name" not in env_config:
            raise ConfigError("'environment_name' must be specified in env_config")

        if "nproc" not in env_config:
            raise ConfigError("'nproc' must be specified in env_config")

        self.environment_name = env_config["environment_name"]
        self.nproc = env_config["nproc"]
        self.hostfile = env_config.get("hostfile", "")

        # reward_agg/reward_aggregation may also arrive via env_config (e.g.
        # through from_hf(**kwargs)). The `reward_agg` argument here is the value already resolved in
        # __init__ from the reward_agg/reward_aggregation kwargs, so it wins
        # whenever it is not at its "mean" default; env_config overrides
        # (from_hf kwargs) win over the bare default:
        #   resolved kwarg (incl. reward_aggregation) > env_config
        #   ["reward_aggregation"] > env_config["reward_agg"] > "mean".
        if "reward_aggregation" in env_config:
            cfg_reward = env_config["reward_aggregation"]
        elif reward_agg != "mean":
            cfg_reward = reward_agg
        elif "reward_agg" in env_config:
            cfg_reward = env_config["reward_agg"]
        else:
            cfg_reward = reward_agg
        if cfg_reward not in ("mean", "sum", "median"):
            raise ValueError(f"reward aggregation must be 'mean', 'sum', or 'median', got {cfg_reward!r}")
        self.reward_agg = cfg_reward

        # Initialize HF data manager
        self.hf_repo_id = env_config.get("hf_repo_id", "dynamicslab/HydroGym-environments")
        self.local_fallback_dir = env_config.get("local_fallback_dir", None)
        self.use_clean_cache = env_config.get("use_clean_cache", True)
        self.hf_token = env_config.get("hf_token", None)
        self.hf_revision = env_config.get("hf_revision", None)
        self.mpi_bind_to = env_config.get("mpi_bind_to", self.DEFAULT_MPI_BIND_TO)

        self.data_manager = HFDataManager(
            repo_id=self.hf_repo_id,
            local_fallback_dir=self.local_fallback_dir,
            use_clean_cache=self.use_clean_cache,
            fallback_profile=self.SOLVER_TYPE,
            token=self.hf_token,
            revision=self.hf_revision,
        )

        # Download/get environment data
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

        # Update paths in configuration to use downloaded data
        self._update_configuration_paths()

        # Apply runtime overrides from env_config
        self._apply_runtime_overrides(env_config)

        # Create run folder
        run_root = env_config.get("run_root", run_root)
        if run_name is None:
            run_name = env_config.get("run_name", "")

        # If run_name is empty string, use run_root directly (no subdirectory)
        if run_name:
            self.run_folder = os.path.join(run_root, run_name)
        else:
            self.run_folder = run_root

        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder, exist_ok=True)

        # Create SESSION.NAME file before MPI split (Nek needs this to start)
        self._create_session_file_early()

        # MPI communicator required by Nek
        comm_world = MPI.COMM_WORLD
        self.sub_comm = self._split_mpmd_comm(comm_world, nproc=self.nproc)

        # Initialize the environment
        self._initialize()

    def _create_session_file_early(self):
        """
        Create SESSION.NAME file required by Nek5000.
        This must be called before MPI split so Nek workers can read it on startup.
        The file must be in the current working directory where mpirun is executed,
        but it contains the path to the run_folder where Nek will execute.
        """
        # Get case name from config (or use environment name as fallback)
        if hasattr(self.conf, "simulation") and hasattr(self.conf.simulation, "CASENAME"):
            casename = self.conf.simulation.CASENAME
        elif hasattr(self.conf, "mesh") and hasattr(self.conf.mesh, "CASENAME"):
            casename = self.conf.mesh.CASENAME
        else:
            # Fallback: use environment name or default
            casename = getattr(self, "environment_name", "phill")

        # Construct absolute path to run folder
        run_folder_abs = os.path.abspath(self.run_folder)

        # Create SESSION.NAME in current working directory (where Nek workers start)
        # The file content points to the run_folder where Nek will actually execute
        cwd = os.getcwd()
        session_file = os.path.join(cwd, "SESSION.NAME")
        with open(session_file, "w") as f:
            f.write(f"{casename}\n")
            f.write(f"{run_folder_abs}\n")

        print(f"[NEK] Created SESSION.NAME: {session_file}")
        print(f"  Case: {casename}")
        print(f"  Work directory: {run_folder_abs}")

    def _resolve_configuration_file(self, config_file_input: Optional[str]) -> Optional[str]:
        """Resolve configuration file path.

        NekEnv uses its own, simpler resolution instead of the
        HFEnvConfigMixin one. Differences from the mixin default:
          - None auto-detects only environment_config.yaml then config.yaml
            (the mixin's _find_configuration_file also tries env_config.yaml,
            environment.yaml, <env>.yaml and config_*.y*ml globs).
          - A missing absolute path or unknown filename returns None (the
            caller's "no configuration file" ConfigError then fires) where
            the mixin raises a more specific ConfigError immediately.
          - A relative path is used when it exists from the working directory,
            else it is looked up in the environment directory.
        """
        # If None, look for config files in environment directory (try multiple names)
        if config_file_input is None:
            # Try both environment_config.yaml (NEK standard) and config.yaml (MAIA standard)
            for config_name in ["environment_config.yaml", "config.yaml"]:
                config_path = os.path.join(self.env_data_path, config_name)
                if os.path.exists(config_path):
                    print(f"[NEK] Using config file: {config_path}")
                    return config_path
            return None

        # If absolute path, use directly
        if os.path.isabs(config_file_input):
            if os.path.exists(config_file_input):
                print(f"[NEK] Using config file: {config_file_input}")
                return config_file_input
            return None

        # A relative path that exists from the working directory (e.g. --config-file ../../../configs/x.yml)
        if os.path.exists(config_file_input):
            config_path = os.path.abspath(config_file_input)
            print(f"[NEK] Using config file: {config_path}")
            return config_path

        # Otherwise, look in environment directory
        config_path = os.path.join(self.env_data_path, config_file_input)
        if os.path.exists(config_path):
            print(f"[NEK] Using config file: {config_path}")
            return config_path
        return None

    def _update_configuration_paths(self):
        """Update paths in configuration to use downloaded data."""
        # Update restart folder path to point to HF data
        if hasattr(self.conf, "initial_conditions"):
            restart_folder = self.conf.initial_conditions.restart_folder
            # Handle both relative and absolute paths
            if not os.path.isabs(restart_folder):
                self.conf.initial_conditions.restart_folder = os.path.join(self.env_data_path, restart_folder)
        elif hasattr(self.conf, "simulation") and hasattr(self.conf.simulation, "restart_folder"):
            # Support old format (both 'restarts' and 'restart_files')
            restart_folder = self.conf.simulation.restart_folder
            if not os.path.isabs(restart_folder):
                # Check if restart folder exists in env_data_path
                # Try both the configured name and common alternatives
                for alt_name in [restart_folder, "restart_files", "restarts"]:
                    candidate_path = os.path.join(self.env_data_path, alt_name)
                    if os.path.exists(candidate_path):
                        self.conf.simulation.restart_folder = candidate_path
                        print(f"[NEK] Found restart files at: {candidate_path}")
                        break
                else:
                    # Fallback: use configured path even if it doesn't exist yet
                    self.conf.simulation.restart_folder = os.path.join(self.env_data_path, restart_folder)

    # env_config keys that are consumed structurally by __init__/_init_from_hf
    # (environment selection, HF/cache options, run layout, reward aggregation)
    # and must never be interpreted as config-tree overrides.
    RESERVED_ENV_CONFIG_KEYS = frozenset(
        {
            "environment_name",
            "nproc",
            "hostfile",
            "hf_repo_id",
            "use_clean_cache",
            "local_fallback_dir",
            "configuration_file",
            "run_root",
            "run_name",
            "reward_agg",
            "reward_aggregation",
            "hf_token",
            "hf_revision",
            "mpi_bind_to",
        }
    )

    # Legacy shorthand keys, mapped to the config-tree path they set. Kept
    # working unchanged; the dotted path is the general form. The second path
    # is the same entry in the older configuration layout (section "runner"),
    # read by the environment as a fallback; a shorthand sets whichever of the
    # two the loaded configuration has.
    SHORTHAND_OVERRIDES = {
        "normalize_input": (("normalization", "normalize_input"), ("runner", "normalize_input")),
        "nb_interactions": (("episode", "max_interactions"), ("runner", "nb_interactions")),
        "random_init": (("initial_conditions", "random_init"), ("runner", "random_init")),
        "rescale_actions": (("rl_interface", "rescale_actions"), ("runner", "rescale_actions")),
        "rew_mode": (("episode", "reward_mode"), ("runner", "rew_mode")),
    }

    def _config_has(self, path: str) -> bool:
        """Whether the dotted path ``section.param`` exists in the loaded configuration."""
        section_path, _, param = path.rpartition(".")
        parent = OmegaConf.select(self.conf, section_path) if section_path else self.conf
        return parent is not None and param in parent

    def _apply_runtime_overrides(self, env_config: Dict):
        """Apply runtime overrides from env_config to the loaded config.

        Two key forms are supported (anything else is a ConfigError, not a
        silent drop):

        - Dotted config-tree paths, e.g. ``{"episode.max_interactions": 5}``
          or ``{"simulation.walltime": 100}``. The path must already exist
          in the loaded configuration.
        - The five legacy shorthand keys in SHORTHAND_OVERRIDES, which map
          to the same tree paths (in a configuration of the older layout, to
          the same entry under ``runner``) and keep working unchanged.

        Reserved structural keys (RESERVED_ENV_CONFIG_KEYS) are ignored here;
        they configure the environment machinery itself, not the solver
        config.
        """
        for key, value in env_config.items():
            if key in self.RESERVED_ENV_CONFIG_KEYS:
                continue
            if key in self.SHORTHAND_OVERRIDES:
                candidates = [f"{section}.{param}" for section, param in self.SHORTHAND_OVERRIDES[key]]
                path = next((c for c in candidates if self._config_has(c)), candidates[0])
            elif "." in key:
                path = key
            else:
                raise ConfigError(
                    f"Unknown env_config override key {key!r}. Use a dotted "
                    "config path (e.g. 'episode.max_interactions') or one of "
                    f"the shorthand keys {sorted(self.SHORTHAND_OVERRIDES)}."
                )

            section_path, _, param = path.rpartition(".")
            parent = OmegaConf.select(self.conf, section_path) if section_path else self.conf
            if parent is None or param not in parent:
                raise ConfigError(
                    f"Runtime override path {path!r} does not exist in the "
                    f"configuration for '{self.environment_name}'. Check the "
                    "environment's config file for the correct section.param."
                )
            parent[param] = value
            print(f"[NEK] Override: {path} = {value}")

    def _get_config_value(self, *paths, default=None):
        """
        Get config value supporting both old and new config formats.

        Args:
          *paths: Tuple of (section, param) pairs to try in order
          default: Default value if not found

        Returns:
          Config value or default

        Example:
          # Try new format first, then old format
          value = self._get_config_value(
              ('mesh', 'lx1'),          # New format
              ('simulation', 'lx1'),     # Old format
          )
        """
        for path in paths:
            if len(path) == 2:
                section, param = path
                if hasattr(self.conf, section):
                    cfg_section = getattr(self.conf, section)
                    if hasattr(cfg_section, param):
                        return getattr(cfg_section, param)
        return default

    def _initialize(self):
        """Initialize the Nek environment."""
        print("------------ INITIALIZATION -------------", flush=True)

        # I/O setup
        self.history_path = Path(f"{self.run_folder}/history")
        self.history_path.mkdir(exist_ok=True)

        # Get restart folder (try new format first, then old)
        restart_folder = self._get_config_value(
            ("initial_conditions", "restart_folder"), ("simulation", "restart_folder")
        )
        # Handle both absolute and relative paths
        if os.path.isabs(restart_folder):
            self.rstart_folder = Path(restart_folder)
        else:
            self.rstart_folder = Path(f"{os.getcwd()}/{restart_folder}")

        # Remove .sch file if it exists
        remove_sch(self.run_folder)

        # Save config
        OmegaConf.save(self.conf, os.path.join(self.history_path, "current_conf.yml"))
        OmegaConf.save(self.conf, os.path.join(self.run_folder, "current_conf.yml"))

        # MPI info for Nek
        mpi_info = MPI.Info.Create()
        mpi_info.Set("wdir", f"{os.getcwd()}/{self.run_folder}")
        mpi_info.Set("bind_to", getattr(self, "mpi_bind_to", self.DEFAULT_MPI_BIND_TO))
        if self.hostfile and self.hostfile != "":
            mpi_info.Set("hostfile", self.hostfile)
            print("[NEK] LOAD HOSTFILE!")
        self.mpi_info = mpi_info

        # Get TOTCTRL before initializing actuators (needed by _init_actuators)
        self.TOTCTRL = self._get_config_value(("mesh", "TOTCTRL"), ("simulation", "TOTCTRL"))

        # Initialize actuators (communicate with Nek to get control point info)
        self._init_actuators()

        # State/observation setup
        self.obs_per_actuator = self._get_config_value(("rl_interface", "npl_state"), ("runner", "npl_state"))
        self.utau = self._get_config_value(("normalization", "u_tau"), ("runner", "u_tau"))

        # Action setup
        lx1 = self._get_config_value(("mesh", "lx1"), ("simulation", "lx1"))
        _, self.gll_weight, _ = lglnodes(N=lx1 - 1)
        print(f"[NEK] GLL WEIGHT={self.gll_weight}", flush=True)

        # Action rescaling
        self.rescale_actions = self._get_config_value(
            ("rl_interface", "rescale_actions"),
            ("runner", "rescale_actions"),
            default=False,
        )
        ctrl_max_amp = self._get_config_value(("rl_interface", "ctrl_max_amp"), ("runner", "ctrl_max_amp"), default=1.0)
        ctrl_min_amp = self._get_config_value(
            ("rl_interface", "ctrl_min_amp"), ("runner", "ctrl_min_amp"), default=-1.0
        )

        if self.rescale_actions:
            self.rescale_factors = [[ctrl_max_amp, ctrl_max_amp]]
            self.ctrl_min_amp = -1.0
            self.ctrl_max_amp = 1.0
        else:
            self.ctrl_min_amp = ctrl_min_amp
            self.ctrl_max_amp = ctrl_max_amp

        # Reward setup
        self.baseline_dudy = self._get_config_value(("normalization", "dUdy"), ("runner", "dUdy"))
        self.restart_index = 0
        self.act_index = 0
        self.reward_log = []

        # Initialize reward logger
        self.reward_logger = RewardLogger(
            log_dir=self.history_path,
            log_name="rewards",
            log_per_agent=False,
            log_aggregated=True,
            flush_frequency=10,
        )

        print(f"SCALE: dUdy={self.baseline_dudy}\n Utau={self.utau}", flush=True)

        # Reward history (if using moving average)
        rew_mode = self._get_config_value(("episode", "reward_mode"), ("runner", "rew_mode"), default="Homo")
        if rew_mode == "MovingAverage":
            size_history = self._get_config_value(("episode", "size_history"), ("runner", "size_history"), default=10)
            self.reward_history = RingBuffer(length=size_history, dim=(self.n_actuators,))
            for i_h in range(size_history):
                self.reward_history.data[i_h] = self.baseline_dudy * np.ones((self.n_actuators,))

        # Define Gym spaces (after all attributes are set)
        # Observation: flattened array of all actuator observations
        obs_size = self.n_actuators * self.obs_per_actuator
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action: flattened array of all actuator actions
        act_low = np.full(self.n_actuators, self.ctrl_min_amp, dtype=np.float32)
        act_high = np.full(self.n_actuators, self.ctrl_max_amp, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            dtype=np.float32,
        )

        # Store config values for later use
        # (TOTCTRL already set before _init_actuators)
        self.tmax = self._get_config_value(
            ("simulation", "tmax"),
            default=1e10,  # Large default means time-based termination is disabled
        )
        self.nb_interactions = self._get_config_value(
            ("episode", "max_interactions"),
            ("runner", "nb_interactions"),
            default=10000,  # Default max steps per episode (old configs don't have this)
        )
        self.ndrl = self._get_config_value(
            ("simulation", "ndrl"),
            default=1,  # Default: 1 Nek step per RL action
        )
        self.target_cfl = self._get_config_value(
            ("simulation", "target_cfl"),
            default=2.0,  # Default CFL threshold
        )
        self.znmf_avg = self._get_config_value(("physics", "znmf_avg"), ("simulation", "znmf_avg"), default=0)
        self.normalize_input = self._get_config_value(
            ("normalization", "normalize_input"),
            ("runner", "normalize_input"),
            default="None",
        )
        self.random_init = self._get_config_value(
            ("initial_conditions", "random_init"), ("runner", "random_init"), default=1
        )
        self.rank = self._get_config_value(("runner", "rank"), default=0)

        self._setup_power()
        self._check_rea()

        print("------------ FINISH INITIALIZATION -------------", flush=True)

    # Solver codes of the power modes (PARAM(96) of the v17 wing case, see drl/drl_power.f)
    POWER_MODES = POWER_MODES

    def _setup_power(self):
        """
        Actuation power (hydrogym.power interface) from the optional ``power`` section of the configuration:

          power:
            enabled: true            # default: true if the section exists, false without it
            power_sampling: mean     # mean (every solver step of an action, dt-weighted) | end
            power_cost: abs          # form charged in the reward (signed | positive | abs)
            power_weight: 0.0        # 0: reward unchanged
            reference: {velocity: 1.0, length: 1.0, span: 0.1}   # C_P = P / (0.5 rho U^3 A); default A = power area
            reply_timeout: 600       # s to wait for the first POWER reply (an executable without it never answers)

        The solver computes the power of the imposed wall velocity at every node of the actuated wall
        (drl/drl_power.f) and returns it on the request POWER. ``simulation.power_mode`` is set from this section
        (0 off, 1 mean, 2 end); the case's parameter writer maps it to p096 of the .rea file.
        """
        spec = self.conf.get("power", None) if hasattr(self.conf, "get") else None
        spec = {} if spec is None else dict(OmegaConf.to_container(spec, resolve=True))
        enabled = bool(spec.get("enabled", bool(spec)))
        self.power_config = PowerConfig.from_config(spec)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self.power_enabled = enabled
        self._power_reference = dict(spec.get("reference") or {})
        self._power_timeout = float(spec.get("reply_timeout", 600.0))
        self._power_checked = False
        self._power_warned = False
        if not enabled:
            if self.power_config.weight != 0.0:
                raise ConfigError("power_weight > 0 needs the actuation power (power.enabled)")
            mode = 0
        else:
            mode = self.POWER_MODES[self.power_sampling]
        sim = self.conf.get("simulation", None) if hasattr(self.conf, "get") else None
        if sim is not None:
            sim["power_mode"] = mode
        if enabled:
            print(
                f"[NEK] actuation power: sampling {self.power_sampling}, weight {self.power_config.weight} "
                f"({self.power_config.cost})",
                flush=True,
            )

    def _check_rea(self):
        """
        v17 cases: Nek5000 reads the .rea file at start-up, before this environment exists, so a configuration value
        that the .rea of the run folder does not have does not reach the solver. Warn about every such entry;
        prepare_workspace.py writes the configuration into the .rea before the launch.
        """
        sim = self.conf.get("simulation", None) if hasattr(self.conf, "get") else None
        if sim is None or not self._is_v17_config(sim):
            return
        path = os.path.join(self.run_folder, f"{sim.CASENAME}.rea")
        if not os.path.exists(path):
            return
        bad = rea_mismatches(sim, path)
        if bad:
            warnings.warn(
                f"{path} differs from the configuration in "
                + ", ".join(f"{pk} ({attr}: file {f:g}, configuration {c:g})" for pk, attr, f, c in bad)
                + "; Nek5000 has read the file at start-up, so these configuration values do not apply. Write the "
                "configuration into the .rea before launching (examples/nek/getting_started/prepare_workspace.py "
                "--config).",
                stacklevel=2,
            )

    @staticmethod
    def _is_v17_config(sim) -> bool:
        return "v17" in str(sim.get("solver_version", "v19")).lower()

    def _reference_power(self, area: float) -> float:
        """0.5 rho U^3 A from power.reference ({velocity, length, span} or {velocity, area}); default A = area."""
        ref = self._power_reference
        rho = float(self._get_config_value(("simulation", "density"), default=1.0))
        u = float(ref.get("velocity", 1.0))
        if ref.get("area") not in (None, "auto"):
            a = float(ref["area"])
        elif "length" in ref or "span" in ref:
            a = float(ref.get("length", 1.0)) * float(ref.get("span", 1.0))
        else:
            a = float(area)
        if not np.isfinite(a) or a <= 0:
            raise ValueError(f"power reference area must be positive, got {a!r}")
        return 0.5 * rho * u**3 * a

    def _recv_power(self) -> Dict:
        """Request POWER: C_P forms at the end of the action and, with sampling mean, their mean over its steps."""
        request = b"POWER"
        self.sub_comm.Send(
            [request, tag_dict["COMMAND"]["mpi_dtype"]],
            dest=0,
            tag=tag_dict["COMMAND"]["tag"],
        )
        hdr = np.empty(16, dtype=np.float64)
        req = self.sub_comm.Irecv([hdr, MPI.DOUBLE], 0, tag=tag_dict["POWER"]["tag"])
        if self._power_checked:
            req.Wait()
        else:
            # An executable built from a case without drl/drl_power.f ignores the request and would block here.
            t0 = time.monotonic()
            while not req.Test():
                if time.monotonic() - t0 > self._power_timeout:
                    req.Cancel()
                    req.Wait()
                    raise RuntimeError(
                        f"the Nek5000 solver did not answer the request POWER within {self._power_timeout:g} s: "
                        "its executable was probably built from a case without actuation power (drl/drl_power.f). "
                        "Rebuild the case executable, or remove the power section of the configuration."
                    )
                time.sleep(0.01)
            self._power_checked = True
        end = np.empty((self.nNID, self.TOTCTRL), dtype=np.float64)
        mean = np.empty((self.nNID, self.TOTCTRL), dtype=np.float64)
        requests = []
        for il, nid in enumerate(self.uniqID):
            requests.append(self.sub_comm.Irecv([end[il], MPI.DOUBLE], nid, tag=nid + tag_dict["POWER_END"]["tag"]))
            requests.append(self.sub_comm.Irecv([mean[il], MPI.DOUBLE], nid, tag=nid + tag_dict["POWER_MEAN"]["tag"]))
        MPI.Request.Waitall(requests)

        def per_actuator(buf):
            out = np.zeros(self.n_actuators, dtype=np.float64)
            icount = 0
            for il, nid in enumerate(self.uniqID):
                n = len(np.where(self.actuator_info["NID"] == nid)[0])
                out[icount : icount + n] = buf[il, :n]
                icount += n
            return out

        ref = self._reference_power(hdr[4])
        names = ("signed", "positive", "abs", "signed_full_pressure")
        out = {"end": {k: float(hdr[i]) / ref for i, k in enumerate(names)}}
        out["end"]["per_actuator"] = per_actuator(end) / ref
        if hdr[10] > 0:
            out["mean"] = {k: float(hdr[5 + i]) / ref for i, k in enumerate(names)}
            out["mean"]["per_actuator"] = per_actuator(mean) / ref
        out["end"]["wall_flux"] = float(hdr[14])
        if hdr[10] > 0:
            out["mean"]["wall_flux"] = float(hdr[15])
        out["area"] = float(hdr[4])
        out["samples"] = int(hdr[11])
        out["time"] = float(hdr[13])
        return out

    def _power_info(self, pw: Dict, sampling: str) -> Dict:
        """info entries of hydrogym.power for the sampling (end, or mean with fallback to end)."""
        if sampling == "mean" and "mean" not in pw:
            if not self._power_warned:
                warnings.warn(
                    "power_sampling mean, but the solver returned no per-step samples (p096 of the .rea file is "
                    "not 1); using the end of the action",
                    stacklevel=2,
                )
                self._power_warned = True
            sampling = "end"
        v = pw[sampling]
        info = power_info(v["per_actuator"], sampling=sampling, forms={k: v[k] for k in POWER_COSTS})
        info["actuation_power_full_pressure"] = float(v["signed_full_pressure"])
        return info

    def _init_actuators(self):
        """Initialize actuators by communicating with Nek to get control point info."""
        request = b"INTAL"
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=tag_dict["COMMAND"]["tag"])
        self.n_actuators = 0

        # Create actuator info storage
        self.actuator_info = {}
        for k in tag_dict.keys():
            if tag_dict[k]["cate"] == "info":
                self.actuator_info[k] = []

        # Handshake from Nek: get node list
        node_list = np.zeros((self.nproc,), dtype=np.int32)
        print(f"[NEK] REQUEST NODE LIST (expecting {self.nproc} ranks)", flush=True)
        self.sub_comm.Recv([node_list, MPI.INTEGER], 0, tag=tag_dict["NID"]["tag"])
        print(f"[NEK] RAW NODE LIST RECEIVED: {node_list}", flush=True)

        # Mask the nid_list
        nid_list = np.arange(self.nproc, dtype=tag_dict["NID"]["py_dtype"])
        print(f"[NEK] Before filter: {nid_list}", flush=True)

        # Keep the ranks that report between 1 and TOTCTRL actuators
        valid_mask = (node_list > 0) & (node_list <= self.TOTCTRL)
        nid_list = nid_list[valid_mask]

        print(
            f"[NEK] After filter (0 < node_list <= {self.TOTCTRL}): {nid_list}",
            flush=True,
        )
        print(f"[NEK] Control points per rank: {node_list[valid_mask]}", flush=True)
        print(f"[NEK] Will receive actuator info from {len(nid_list)} ranks", flush=True)

        # Receive actuator info from each rank
        for nid in nid_list:
            print(f"[NEK] Processing rank nid={nid}", flush=True)
            rank_data = {}
            rank_data["NID"] = np.array([nid], dtype=tag_dict["NID"]["py_dtype"])
            for k in self.actuator_info.keys():
                if "NID" not in k:
                    if "NUMCTRL" not in k:
                        rank_data[k] = np.empty((self.TOTCTRL,), dtype=tag_dict[k]["py_dtype"])
                    else:
                        rank_data[k] = np.empty((1,), dtype=tag_dict[k]["py_dtype"])

                    # MPI receive
                    print(
                        f"[NEK] Waiting to receive {k} from rank {nid}, tag={nid + tag_dict[k]['tag']}",
                        flush=True,
                    )
                    self.sub_comm.Recv(
                        [rank_data[k], tag_dict[k]["mpi_dtype"]],
                        nid,
                        tag=nid + tag_dict[k]["tag"],
                    )
                    print(f"[NEK] Received {k} from rank {nid}", flush=True)

            # Resize to match actual number of control points
            numctrl = rank_data["NUMCTRL"][0]
            for k in rank_data.keys():
                if ("NUMCTRL" in k) or ("NID" in k):
                    rank_data[k] = rank_data[k][0] * np.ones(shape=(numctrl,), dtype=tag_dict[k]["py_dtype"])
                else:
                    rank_data[k] = rank_data[k][:numctrl]
                self.actuator_info[k].append(rank_data[k])

            self.n_actuators += numctrl

        # Concatenate all actuator info
        for k in self.actuator_info.keys():
            self.actuator_info[k] = np.concatenate(self.actuator_info[k])
        print(f"[NEK] INIT END, NUM ACTUATORS={self.n_actuators}")

        # Get unique node IDs
        self.uniqID = np.unique(self.actuator_info["NID"])
        self.nNID = len(self.uniqID)

        # Save actuator info
        df = pd.DataFrame(self.actuator_info)
        fname = os.path.join(self.history_path, "ACTUATOR_INFO.csv")
        df.to_csv(fname)
        print(f"[NEK] DUMP ACTUATOR INFO : {fname}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        print("[NEK] RESET!", flush=True)

        # Update episode counter
        self.restart_index += 1
        print(f"[NEK] EPISODE={self.restart_index}", flush=True)

        # Reset action index
        self.act_index = 0

        # Save and reset reward log
        np.savez(
            os.path.join(self.history_path, f"rewlog_{self.restart_index:05d}.npz"),
            rew=np.array(self.reward_log),
        )

        # Log episode summary
        self.reward_logger.log_episode_summary(episode=self.restart_index)

        self.reward_log = []
        print("[NEK] SAVE LOG", flush=True)

        # Remove .sch file
        remove_sch(self.run_folder)

        # Handle restart files
        self._restart_handle()

        # Start simulation
        self._start_simulation()
        print("[NEK] Start SIM", flush=True)

        # Get initial observation
        time, observation = self._get_state()

        info = {}
        if self.power_enabled:  # instantaneous power of the wall velocity now imposed
            info.update(self._power_info(self._recv_power(), "end"))
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.

        Args:
          action: Flat array of actions for all actuators, shape (n_actuators,)

        Returns:
          observation: Flat array of observations, shape (n_actuators * obs_per_actuator,)
          reward: Scalar reward
          terminated: Always False for Nek (physics failure raises
              NekDivergenceError from _evolve instead)
          truncated: Whether the episode budget was reached (simulation
              end time tmax or nb_interactions RL steps)
          info: Additional information
        """
        # Validate action shape
        action = np.asarray(action).reshape(-1)
        if action.size != self.n_actuators:
            raise ValueError(f"Action size {action.size} does not match expected {self.n_actuators}")

        # Rescale actions if needed
        if self.rescale_actions:
            action_rescaled = action.copy()
            action_rescaled[action < 0] *= self.rescale_factors[0][0]
            action_rescaled[action > 0] *= self.rescale_factors[0][1]
            action = action_rescaled

        # Send action to Nek
        self._send_action(action)

        # Evolve simulation and get rewards
        rewards_per_actuator = self._evolve()

        # Aggregate reward
        if self.reward_agg == "sum":
            reward = float(np.sum(rewards_per_actuator))
        elif self.reward_agg == "median":
            reward = float(np.median(rewards_per_actuator))
        else:  # mean
            reward = float(np.mean(rewards_per_actuator))

        pinfo = {}
        if self.power_enabled:
            pinfo = self._power_info(self._recv_power(), self.power_sampling)
            forms = {
                "signed": pinfo["actuation_power"],
                "positive": pinfo["actuation_power_positive"],
                "abs": pinfo["actuation_power_abs"],
            }
            reward = self.power_config.apply(reward, forms)

        # Get new observation
        flow_time, observation = self._get_state()

        # Check if done: reaching the episode's simulation end time or step
        # budget is a TRUNCATION, not a termination -- terminated is reserved
        # for physics failure, which surfaces as a raised NekDivergenceError
        # from _evolve, not a returned flag.
        truncated = False
        if flow_time > self.tmax:
            truncated = True

        self.act_index += 1
        if self.act_index >= self.nb_interactions:
            print(f"[STEP] ACT_INDEX={self.act_index}; TRUNCATED == TRUE", flush=True)
            truncated = True

        terminated = False

        info = {
            "time": flow_time,
            "reward_per_actuator": rewards_per_actuator,
            **pinfo,
        }

        return observation, reward, terminated, truncated, info

    def _get_state(self) -> Tuple[float, np.ndarray]:
        """Get current state from Nek."""
        request = b"STATE"
        self.sub_comm.Send(
            [request, tag_dict["COMMAND"]["mpi_dtype"]],
            dest=0,
            tag=tag_dict["COMMAND"]["tag"],
        )

        # Constants
        NFLDC = self.obs_per_actuator
        TOTCTRL = self.TOTCTRL

        # Receive current time
        current_time = np.ndarray((1,), dtype=np.float64)
        self.sub_comm.Recv([current_time, MPI.DOUBLE], 0, tag=1998)
        current_time = current_time[0]

        # Receive state from each node. Batched: post every per-node,
        # per-field Irecv up front and wait once. The solver sends one
        # point-to-point message per (node, field); MPI matching (by
        # source+tag, with non-overtaking order within a source+tag for
        # nid=0's shared tag) preserves the per-field ordering.
        state_buffer = np.ndarray(shape=(self.nNID, NFLDC, TOTCTRL), dtype=np.float64)
        requests = []
        for ni, nid in enumerate(self.uniqID):
            for t in range(NFLDC):
                requests.append(
                    self.sub_comm.Irecv(
                        [state_buffer[ni, t, :], tag_dict["STATE"]["mpi_dtype"]],
                        nid,
                        tag=nid * (t + 1) + tag_dict["STATE"]["tag"],
                    )
                )
        MPI.Request.Waitall(requests)

        # Normalize each node's block exactly as before (per node, over the
        # whole (NFLDC, TOTCTRL) block).
        for ni in range(self.nNID):
            state_buffer[ni, :, :] = self._normalize_state(state_buffer[ni, :, :])

        print("[NEK] STATE RECV", flush=True)

        # Flatten state into observation array
        # state_buffer shape: (nNID, NFLDC, TOTCTRL)
        # We need to create a flat array for all actuators
        observation = np.zeros(self.n_actuators * NFLDC, dtype=np.float32)

        icount = 0
        for il, nid in enumerate(self.uniqID):
            _index = np.where((self.actuator_info["NID"] == nid))[0]
            for jl in range(len(_index)):
                # Extract observation for this actuator
                obs = state_buffer[il, :, jl].flatten()  # Shape: (NFLDC,)
                observation[icount * NFLDC : (icount + 1) * NFLDC] = obs
                icount += 1

        assert icount == self.n_actuators, ValueError("[NEK] Actuator count mismatch!")

        return current_time, observation

    def _send_action(self, action: np.ndarray):
        """Send action to Nek."""
        request = b"CNTRL"
        self.sub_comm.Send(
            [request, tag_dict["COMMAND"]["mpi_dtype"]],
            dest=0,
            tag=tag_dict["COMMAND"]["tag"],
        )

        # Apply ZNMF condition
        action = self._apply_znmf(action)

        # Send actions to each node. Batched: build every per-node buffer
        # first, then post one Isend per node and wait once (one
        # point-to-point message per node).
        action_buffers = []
        requests = []
        icount = 0
        for il, nid in enumerate(self.uniqID):
            _index = np.where((self.actuator_info["NID"] == nid))[0]

            # Create action buffer for this node
            act_buffer = np.ndarray(shape=(self.TOTCTRL,), dtype=tag_dict["ACTION"]["py_dtype"])
            for jl in range(len(_index)):
                act_buffer[jl] = action[icount]
                icount += 1

            # Post the send; the buffer is kept alive until Waitall below
            action_buffers.append(act_buffer)
            requests.append(
                self.sub_comm.Isend(
                    [act_buffer, tag_dict["ACTION"]["mpi_dtype"]],
                    nid,
                    tag=nid + tag_dict["ACTION"]["tag"],
                )
            )
        MPI.Request.Waitall(requests)

        assert icount == self.n_actuators, ValueError("[NEK] Actuator count mismatch!")
        print(f"[NEK] ACTION for {icount} Actuators", flush=True)

    def _evolve(self) -> np.ndarray:
        """Evolve the simulation and return rewards."""
        request = b"EVOLV"
        self.sub_comm.Send(
            [request, tag_dict["COMMAND"]["mpi_dtype"]],
            dest=0,
            tag=tag_dict["COMMAND"]["tag"],
        )

        # Synchronize with Nek
        i_evolv = 1
        while i_evolv <= (self.ndrl):
            # Receive CFL data
            current_cfl = np.ndarray((1,), dtype=np.float64)
            self.sub_comm.Recv(
                [current_cfl, tag_dict["current_cfl"]["mpi_dtype"]],
                0,
                tag=tag_dict["current_cfl"]["tag"],
            )
            current_cfl = current_cfl[0]

            # Check for CFL explosion
            if current_cfl >= self.target_cfl:
                print(
                    f"[WARNING] {i_evolv}/{self.ndrl} Current CFL {current_cfl} >= {self.target_cfl}!",
                    flush=True,
                )
                # Tell the solver side to shut down cleanly (TERMN + comm free
                # + MPI finalize) BEFORE raising, so the MPMD job cannot hang
                # if the caller chooses not to catch. Raising lets gym wrappers /
                # RL loops observe the failure; the env is not usable afterwards.
                self._end_simulation(farewell=True)
                raise NekDivergenceError(
                    f"CFL blew up at evolve step {i_evolv}/{self.ndrl}: "
                    f"current_cfl={current_cfl} >= target_cfl={self.target_cfl}. "
                    "The Nek simulation has been terminated (TERMN sent)."
                )

            # Receive reward buffer at last step
            if i_evolv == self.ndrl:
                ws_stress_buffer = np.ndarray(shape=(self.nNID, self.TOTCTRL), dtype=tag_dict["REWRD"]["py_dtype"])
                for il, nid in enumerate(self.uniqID):
                    recv_buffer = np.ndarray(shape=(self.TOTCTRL,), dtype=np.float64)
                    self.sub_comm.Recv(
                        [recv_buffer, tag_dict["REWRD"]["mpi_dtype"]],
                        nid,
                        tag=nid + tag_dict["REWRD"]["tag"],
                    )
                    ws_stress_buffer[il, :] = recv_buffer

            i_evolv += 1

        # Extract rewards for each actuator
        rewards = np.zeros(self.n_actuators, dtype=np.float32)
        icount = 0
        for il, nid in enumerate(self.uniqID):
            _index = np.where((self.actuator_info["NID"] == nid))[0]
            for jl in range(len(_index)):
                r_reward = ws_stress_buffer[il, jl]
                i_reward = self._normalize_reward(r_reward)
                rewards[icount] = i_reward
                icount += 1

        # Log rewards
        self.reward_log.append(i_reward)

        # Real-time reward logging
        self.reward_logger.log_rewards(
            rewards={"aggregated": float(np.mean(rewards))},
            dUdy_raw={"aggregated": float(np.mean(ws_stress_buffer))},
            episode=self.restart_index,
            step=self.act_index,
        )

        print(
            f"[LOGGER] act_index={self.act_index} dUdy={np.mean(ws_stress_buffer):.5f} R={np.mean(rewards):.5f}",
            flush=True,
        )

        return rewards

    def _apply_znmf(self, action: np.ndarray) -> np.ndarray:
        """Apply zero-net-mass-flux condition."""
        znmf_avg = self.znmf_avg

        if znmf_avg == -1:  # Naive average
            mean_action = np.mean(action)
            return action - mean_action

        elif znmf_avg == -2:  # Weighted average
            mean_action = 0.0
            wxz = 0.0
            for il in range(self.n_actuators):
                ix = self.actuator_info["ix"][il]
                iz = self.actuator_info["iz"][il]
                wx = self.gll_weight[ix - 1]
                wz = self.gll_weight[iz - 1]
                wxz += wx * wz
                mean_action += action[il] * wx * wz
            mean_action /= wxz
            return action - mean_action

        elif znmf_avg in [0, 1]:  # Done by Nek
            print("[ZNMF] DONE BY NEK5000", flush=True)
            return action

        else:
            raise NotImplementedError("Please ensure the ZNMF condition!")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state observations."""
        if self.normalize_input != "None":
            if self.normalize_input == "utau":
                state /= self.utau
            elif self.normalize_input == "std":
                state /= np.std(state)
            elif self.normalize_input == "minmax":
                state = 2 * (state - np.min(state)) / (np.max(state) - np.min(state)) - 1
        return state

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward."""
        if self.baseline_dudy is None:
            raise ValueError(
                "the reward needs the uncontrolled mean wall gradient: set normalization.dUdy (or runner.dUdy) in the "
                "environment configuration, or env.baseline_dudy"
            )
        return 1 - (reward / self.baseline_dudy)

    def _start_simulation(self):
        """Start the Nek simulation."""
        request = b"RSETS"
        self.sub_comm.Send([request, tag_dict["COMMAND"]["mpi_dtype"]], dest=0, tag=22)

    def _end_simulation(self, farewell=False):
        """End the current simulation."""
        if farewell:
            request = b"TERMN"
            self.sub_comm.Send(
                [request, tag_dict["COMMAND"]["mpi_dtype"]],
                dest=0,
                tag=tag_dict["COMMAND"]["tag"],
            )
            self.sub_comm.Free()
            sleep_time = 0.01 * self.nproc
            print(f"[NEK] TERMN ENV, Sleep: {sleep_time}SEC")
            time.sleep(sleep_time)
            MPI.Finalize()
        else:
            print("[NEK] Restart the simulation!", flush=True)
            self._start_simulation()

    def _restart_handle(self):
        """Manage restart files for each episode."""
        import shutil

        if self.random_init > 0:
            n_init = np.random.randint(low=1, high=self.random_init + 1)
            target_folder = os.path.join(self.rstart_folder, f"init_{n_init}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if "rs" in f]
            for rsfile in rs_list:
                rsfile = os.path.join(target_folder, rsfile)
                # copy the contents only (the cached restarts are read-only) and replace an earlier copy
                dst = os.path.join(self.mpi_info["wdir"], os.path.basename(rsfile))
                if os.path.lexists(dst):
                    os.remove(dst)
                shutil.copyfile(rsfile, dst)
                print(f"[RSTART] RESET: {rsfile}", flush=True)

        elif self.random_init == -1:
            print(f"[RSTART] NOT SHUFFLE; RANK={self.rank}", flush=True)
            target_folder = os.path.join(self.rstart_folder, f"init_{self.rank}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if "rs" in f]
            for rsfile in rs_list:
                rsfile = os.path.join(target_folder, rsfile)
                # copy the contents only (the cached restarts are read-only) and replace an earlier copy
                dst = os.path.join(self.mpi_info["wdir"], os.path.basename(rsfile))
                if os.path.lexists(dst):
                    os.remove(dst)
                shutil.copyfile(rsfile, dst)
                print(f"[RSTART] RESET: {rsfile}", flush=True)

        elif (self.random_init < -1) and (self.restart_index == 1):
            print(f"[RSTART] NOT OVERWRITE; RANK {self.rank}", flush=True)
            target_folder = os.path.join(self.rstart_folder, f"init_{self.rank}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if "rs" in f]

            if len(rs_list) < 3:
                raise ValueError("[RSTART] NOT ENOUGH FILE TO RESTART")
            else:
                for rsfile in rs_list:
                    print(f"[RSTART] EXIST: {rsfile}", flush=True)

    @staticmethod
    def _name_agent(nid, gllid, iface, ix, iy, iz):
        """Create agent name from grid information."""
        return f"jet_np{nid:08d}_gid{gllid:08d}_iface{iface}_ix{ix:08d}_iy{iy:08d}_iz{iz:08d}"

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment."""
        print("[NEK] CLOSE ENV", flush=True)

        # Close reward logger
        if hasattr(self, "reward_logger"):
            self.reward_logger.close()
            print("[NEK] REWARD LOGGER CLOSED", flush=True)

        self._end_simulation(farewell=True)
        time.sleep(1)


class RingBuffer:
    """N-dimensional ring buffer using numpy arrays."""

    def __init__(self, length, dim=1):
        """Allocate the backing array.

        Args:
            length: Number of entries the buffer holds before wrapping.
            dim: Shape of each entry (an int, or a tuple of axis sizes);
                defaults to scalar entries.
        """
        if type(dim) is int:
            dim = (dim,)
        self.data = np.zeros((length,) + dim, dtype="f")
        self.index = 0

    def extend(self, x):
        """Add array x to ring buffer."""
        assert x.shape == self.data.shape[1:], "Input array does not match the ring buffer size"
        x_index = self.index % self.data.shape[0]
        self.data[x_index] = x
        self.index = x_index + 1

    def get(self):
        """Return first-in-first-out data in the ring buffer."""
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

    def average(self):
        """Return average of entries in the ring buffer."""
        return np.mean(self.data, axis=0)


# MPI tag dictionary
tag_dict = {
    "NID": {
        "tag": 1996,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "NUMCTRL": {
        "tag": 10000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "GLLID": {
        "tag": 20000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "FACEID": {
        "tag": 30000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "ix": {
        "tag": 40000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "iy": {
        "tag": 50000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "iz": {
        "tag": 60000,
        "mpi_dtype": MPI.INTEGER,
        "py_dtype": np.int32,
        "cate": "info",
    },
    "x": {
        "tag": 100000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "info",
    },
    "y": {
        "tag": 200000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "info",
    },
    "z": {
        "tag": 300000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "info",
    },
    "current_cfl": {
        "tag": 1999,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "request",
    },
    "current_time": {
        "tag": 1998,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "request",
    },
    "STATE": {
        "tag": 70000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "request",
    },
    "REWRD": {
        "tag": 80000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "request",
    },
    "ACTION": {
        "tag": 90000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        "cate": "send",
    },
    "COMMAND": {"tag": 22, "mpi_dtype": MPI.CHARACTER, "py_dtype": str, "cate": "send"},
    # actuation power (request POWER, drl/drl_power.f of the wing case)
    "POWER": {"tag": 1995, "mpi_dtype": MPI.DOUBLE, "py_dtype": np.float64, "cate": "request"},
    "POWER_END": {"tag": 110000, "mpi_dtype": MPI.DOUBLE, "py_dtype": np.float64, "cate": "request"},
    "POWER_MEAN": {"tag": 120000, "mpi_dtype": MPI.DOUBLE, "py_dtype": np.float64, "cate": "request"},
}
