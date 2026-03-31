"""
Core Nek5000 environment with Gymnasium interface.
Single-agent with array-based observations/actions.

Supports two initialization patterns:
1. MAIA pattern (recommended): env = NekEnv.from_hf('EnvName', nproc=10)
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

from hydrogym.data_manager import HFDataManager

from .configs import Config
from .nek_lib.lglnodes import lglnodes
from .nek_lib.nek_utils import remove_sch
from .nek_lib.reward_logger import RewardLogger


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


def mpi_split(comm_world: MPI.Comm, nproc: Optional[int] = None) -> MPI.Comm:
    """
    Split MPI world into master/worker inter-communicator.

    Args:
      comm_world: MPI communicator
      nproc: Expected number of Nek workers (for validation)

    Returns:
      Inter-communicator between controller and workers
    """
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()

    if mpi_size < 2:
        raise RuntimeError(
            "MPI world size must be >= 2 to create the Nek inter-communicator. "
            "Launch with MPMD, e.g. `mpirun -n 1 python ... : -n N ./nek5000`, "
            "so rank 0 can connect to the Nek worker ranks."
        )

    # Validate MPI size matches nproc
    if nproc is not None:
        expected_size = 1 + nproc  # 1 controller + N workers
        if mpi_size != expected_size:
            raise RuntimeError(
                f"MPI world size mismatch: expected {expected_size} "
                f"(1 controller + {nproc} workers), got {mpi_size}. "
                f"Launch with: mpirun -n 1 python ... : -n {nproc} ./nek5000"
            )

    if mpi_rank == 0:
        color = 0
    else:
        color = 1

    local_comm = comm_world.Split(color, mpi_rank)
    print(
        f"[MPI_SPLIT] World rank {mpi_rank}, color {color}, "
        f"local_comm size: {local_comm.Get_size()}, "
        f"local rank: {local_comm.Get_rank()}",
        flush=True,
    )

    sub_comm = local_comm.Create_intercomm(
        local_leader=0, peer_comm=MPI.COMM_WORLD, remote_leader=1, tag=99
    )
    print(
        f"[MPI_SPLIT] Inter-comm created: local_size={sub_comm.Get_size()}, "
        f"remote_size={sub_comm.Get_remote_size()}",
        flush=True,
    )
    return sub_comm


class NekEnv(gym.Env):
    """
    Core Nek5000 environment with Gymnasium interface.

    This is a single-agent environment where the agent controls multiple
    actuators (control points) on the mesh. Observations and actions are
    flat arrays representing all actuators.

    Supports two initialization patterns:

    1. MAIA pattern (recommended):
        env = NekEnv.from_hf(
            'MiniChannel_Re180',
            nproc=10,
            hostfile='',
        )

    2. Legacy pattern (deprecated):
        conf = OmegaConf.load('config.yaml')
        env = NekEnv(conf=conf)

    Args (MAIA pattern via env_config):
      environment_name: Name of environment on HuggingFace
      nproc: Number of MPI workers for Nek (required)
      hostfile: MPI hostfile path (default: '')
      hf_repo_id: HuggingFace repository (default: 'dynamicslab/HydroGym-environments')
      use_clean_cache: Use fresh workspace (default: True)
      local_fallback_dir: Local directory for offline usage
      configuration_file: Override config file path
      run_root: Root directory for outputs (default: 'runs')
      run_name: Name for this run (default: '' = no subdirectory, use run_root directly)
      reward_agg: Reward aggregation method ("mean" or "sum")
      ... (runtime overrides for config parameters)

    Args (Legacy pattern):
      conf: Configuration object (OmegaConf)
      run_root: Root directory for run outputs
      run_name: Name for this run (defaults to MPI rank)
      reward_agg: How to aggregate per-actuator rewards ("mean" or "sum")
    """

    metadata = {"render_modes": ["human"]}
    SOLVER_TYPE = "NEK5000"

    def __init__(
        self,
        conf: Optional[Config] = None,
        env_config: Optional[Dict] = None,
        run_root: str = ".",
        run_name: Optional[str] = None,
        reward_agg: str = "mean",
        **kwargs,
    ):
        """
        Initialize NekEnv with either legacy conf or MAIA env_config pattern.

        Args:
          conf: Legacy OmegaConf object (deprecated)
          env_config: MAIA-style configuration dict (recommended)
          run_root: Output directory root
          run_name: Run name (auto-generate if None)
          reward_agg: Reward aggregation method
          **kwargs: Additional parameters (for backward compatibility)
        """
        # Determine which API is being used
        if conf is not None and env_config is not None:
            raise ValueError(
                "Cannot provide both 'conf' and 'env_config'. Use one or the other."
            )

        if conf is not None:
            # Legacy API
            warnings.warn(
                "Passing 'conf' is deprecated. Use the MAIA pattern: "
                "env = NekEnv.from_hf('EnvName', nproc=10)",
                DeprecationWarning,
                stacklevel=2,
            )
            self._init_from_legacy(conf, run_root, run_name, reward_agg)

        elif env_config is not None:
            # MAIA API
            self._init_from_hf(env_config, run_root, run_name, reward_agg)

        else:
            raise ValueError(
                "Must provide either 'conf' (legacy) or 'env_config' (MAIA pattern). "
                "Recommended: use NekEnv.from_hf('EnvName', nproc=10)"
            )

    @classmethod
    def from_hf(cls, environment_name: str, nproc: int, hostfile: str = "", **kwargs):
        """
        Create environment from HuggingFace Hub (MAIA pattern).

        Args:
          environment_name: Name of the environment (e.g., 'MiniChannel_Re180')
          nproc: Number of MPI workers for Nek (required)
          hostfile: MPI hostfile path (default: '')
          **kwargs: Additional env_config parameters:
            - hf_repo_id: HF repository (default: 'dynamicslab/HydroGym-environments')
            - use_clean_cache: Fresh workspace (default: True)
            - local_fallback_dir: Local directory
            - configuration_file: Override config path
            - run_root: Output directory (default: 'runs')
            - run_name: Run name (auto-generate if None)
            - reward_agg: 'mean' or 'sum' (default: 'mean')
            - normalize_input: Override normalization strategy
            - nb_interactions: Override episode length
            - random_init: Override IC randomization
            - rescale_actions: Override action rescaling
            - rew_mode: Override reward mode

        Returns:
          NekEnv instance

        Example:
          env = NekEnv.from_hf('MiniChannel_Re180', nproc=10)

          env = NekEnv.from_hf(
              'MiniChannel_Re180',
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

    def _init_from_legacy(
        self, conf: Config, run_root: str, run_name: Optional[str], reward_agg: str
    ):
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
        self.sub_comm = mpi_split(comm_world, nproc=self.nproc)

        # Initialize the environment (this sets n_actuators, obs_per_actuator, etc.)
        self._initialize()

    def _init_from_hf(
        self, env_config: Dict, run_root: str, run_name: Optional[str], reward_agg: str
    ):
        """Initialize using MAIA pattern from HuggingFace."""
        # Validate required parameters
        if "environment_name" not in env_config:
            raise ConfigError("'environment_name' must be specified in env_config")

        if "nproc" not in env_config:
            raise ConfigError("'nproc' must be specified in env_config")

        self.environment_name = env_config["environment_name"]
        self.nproc = env_config["nproc"]
        self.hostfile = env_config.get("hostfile", "")
        self.reward_agg = reward_agg

        # Initialize HF data manager
        self.hf_repo_id = env_config.get(
            "hf_repo_id", "dynamicslab/HydroGym-environments"
        )
        self.local_fallback_dir = env_config.get("local_fallback_dir", None)
        self.use_clean_cache = env_config.get("use_clean_cache", True)

        self.data_manager = HFDataManager(
            repo_id=self.hf_repo_id,
            local_fallback_dir=self.local_fallback_dir,
            use_clean_cache=self.use_clean_cache,
            fallback_profile=self.SOLVER_TYPE,
        )

        # Download/get environment data
        self.env_data_path = self._setup_environment_data()

        # Resolve and load configuration file
        self.configuration_file = self._resolve_configuration_file(
            env_config.get("configuration_file")
        )

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
        self.sub_comm = mpi_split(comm_world, nproc=self.nproc)

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
        if hasattr(self.conf, "simulation") and hasattr(
            self.conf.simulation, "CASENAME"
        ):
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

    def _setup_environment_data(self):
        """
        Download and setup environment data from HF Hub.

        First checks ~/.cache/nekgym/ for local data, otherwise falls back to data_manager.

        Returns:
            Path to the local environment data directory.
        """
        from pathlib import Path

        # Check cache directory first (like MAIA does)
        cache_dir = Path.home() / ".cache" / "nekgym" / self.environment_name
        if cache_dir.exists() and cache_dir.is_dir():
            print(f"[NEK] Using cached environment data from: {cache_dir}")
            return str(cache_dir)

        # Fall back to data_manager if cache doesn't exist
        try:
            env_path = self.data_manager.get_environment_path(self.environment_name)
            print(f"[NEK] Using environment data from: {env_path}")
            return env_path
        except Exception as e:
            raise ConfigError(
                f"Failed to setup environment data for {self.environment_name}: {e}"
            )

    def _resolve_configuration_file(
        self, config_file_input: Optional[str]
    ) -> Optional[str]:
        """
        Resolve configuration file path.

        Args:
          config_file_input: Can be None (auto-detect), absolute path, or filename

        Returns:
          Absolute path to config file, or None if not found
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
                self.conf.initial_conditions.restart_folder = os.path.join(
                    self.env_data_path, restart_folder
                )
        elif hasattr(self.conf, "simulation") and hasattr(
            self.conf.simulation, "restart_folder"
        ):
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
                    self.conf.simulation.restart_folder = os.path.join(
                        self.env_data_path, restart_folder
                    )

    def _apply_runtime_overrides(self, env_config: Dict):
        """Apply runtime overrides from env_config to loaded config."""
        # Allow runtime override of certain parameters
        override_map = {
            "normalize_input": ("normalization", "normalize_input"),
            "nb_interactions": ("episode", "max_interactions"),
            "random_init": ("initial_conditions", "random_init"),
            "rescale_actions": ("rl_interface", "rescale_actions"),
            "rew_mode": ("episode", "reward_mode"),
        }

        for key, (section, param) in override_map.items():
            if key in env_config:
                # Check if section exists in config
                if hasattr(self.conf, section):
                    cfg_section = getattr(self.conf, section)
                    if hasattr(cfg_section, param):
                        setattr(cfg_section, param, env_config[key])
                        print(f"[NEK] Override: {section}.{param} = {env_config[key]}")

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
        mpi_info.Set("bind_to", "none")
        if self.hostfile and self.hostfile != "":
            mpi_info.Set("hostfile", self.hostfile)
            print("[NEK] LOAD HOSTFILE!")
        self.mpi_info = mpi_info

        # Get TOTCTRL before initializing actuators (needed by _init_actuators)
        self.TOTCTRL = self._get_config_value(
            ("mesh", "TOTCTRL"), ("simulation", "TOTCTRL")
        )

        # Initialize actuators (communicate with Nek to get control point info)
        self._init_actuators()

        # State/observation setup
        self.obs_per_actuator = self._get_config_value(
            ("rl_interface", "npl_state"), ("runner", "npl_state")
        )
        self.utau = self._get_config_value(
            ("normalization", "u_tau"), ("runner", "u_tau")
        )

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
        ctrl_max_amp = self._get_config_value(
            ("rl_interface", "ctrl_max_amp"), ("runner", "ctrl_max_amp"), default=1.0
        )
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
        self.baseline_dudy = self._get_config_value(
            ("normalization", "dUdy"), ("runner", "dUdy")
        )
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
        rew_mode = self._get_config_value(
            ("episode", "reward_mode"), ("runner", "rew_mode"), default="Homo"
        )
        if rew_mode == "MovingAverage":
            size_history = self._get_config_value(
                ("episode", "size_history"), ("runner", "size_history"), default=10
            )
            self.reward_history = RingBuffer(
                length=size_history, dim=(self.n_actuators,)
            )
            for i_h in range(size_history):
                self.reward_history.data[i_h] = self.baseline_dudy * np.ones(
                    (self.n_actuators,)
                )

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
            ("simulation", "ndrl"), default=1  # Default: 1 Nek step per RL action
        )
        self.target_cfl = self._get_config_value(
            ("simulation", "target_cfl"), default=2.0  # Default CFL threshold
        )
        self.znmf_avg = self._get_config_value(
            ("physics", "znmf_avg"), ("simulation", "znmf_avg"), default=0
        )
        self.normalize_input = self._get_config_value(
            ("normalization", "normalize_input"),
            ("runner", "normalize_input"),
            default="None",
        )
        self.random_init = self._get_config_value(
            ("initial_conditions", "random_init"), ("runner", "random_init"), default=1
        )
        self.rank = self._get_config_value(("runner", "rank"), default=0)

        print("------------ FINISH INITIALIZATION -------------", flush=True)

    def _init_actuators(self):
        """Initialize actuators by communicating with Nek to get control point info."""
        request = b"INTAL"
        self.sub_comm.Send(
            [request, MPI.CHARACTER], dest=0, tag=tag_dict["COMMAND"]["tag"]
        )
        self.n_actuators = 0

        # Create actuator info storage
        self.actuator_info = {}
        for k in tag_dict.keys():
            if tag_dict[k]["cate"] == "info":
                self.actuator_info[k] = []

        # Handshake from Nek: get node list
        node_list = np.zeros(
            (self.nproc,), dtype=np.int32
        )  # Initialize to zero to avoid garbage values
        print(f"[NEK] REQUEST NODE LIST (expecting {self.nproc} ranks)", flush=True)
        self.sub_comm.Recv([node_list, MPI.INTEGER], 0, tag=tag_dict["NID"]["tag"])
        print(f"[NEK] RAW NODE LIST RECEIVED: {node_list}", flush=True)

        # Mask the nid_list
        nid_list = np.arange(self.nproc, dtype=tag_dict["NID"]["py_dtype"])
        print(f"[NEK] Before filter: {nid_list}", flush=True)

        # Filter: keep ranks where node_list has sensible positive values
        # (avoid garbage memory values that might be huge)
        valid_mask = (node_list > 0) & (node_list <= self.TOTCTRL)
        nid_list = nid_list[valid_mask]

        print(
            f"[NEK] After filter (0 < node_list <= {self.TOTCTRL}): {nid_list}",
            flush=True,
        )
        print(f"[NEK] Control points per rank: {node_list[valid_mask]}", flush=True)
        print(
            f"[NEK] Will receive actuator info from {len(nid_list)} ranks", flush=True
        )

        # Receive actuator info from each rank
        for nid in nid_list:
            print(f"[NEK] Processing rank nid={nid}", flush=True)
            rank_data = {}
            rank_data["NID"] = np.array([nid], dtype=tag_dict["NID"]["py_dtype"])
            for k in self.actuator_info.keys():
                if "NID" not in k:
                    if "NUMCTRL" not in k:
                        rank_data[k] = np.empty(
                            (self.TOTCTRL,), dtype=tag_dict[k]["py_dtype"]
                        )
                    else:
                        rank_data[k] = np.empty((1,), dtype=tag_dict[k]["py_dtype"])

                    # MPI receive
                    print(
                        f'[NEK] Waiting to receive {k} from rank {nid}, tag={nid + tag_dict[k]["tag"]}',
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
                    rank_data[k] = rank_data[k][0] * np.ones(
                        shape=(numctrl,), dtype=tag_dict[k]["py_dtype"]
                    )
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

        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.

        Args:
          action: Flat array of actions for all actuators, shape (n_actuators,)

        Returns:
          observation: Flat array of observations, shape (n_actuators * obs_per_actuator,)
          reward: Scalar reward
          terminated: Whether episode is done
          truncated: Whether episode was truncated (always False for Nek)
          info: Additional information
        """
        # Validate action shape
        action = np.asarray(action).reshape(-1)
        if action.size != self.n_actuators:
            raise ValueError(
                f"Action size {action.size} does not match expected {self.n_actuators}"
            )

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
        else:  # mean
            reward = float(np.mean(rewards_per_actuator))

        # Get new observation
        flow_time, observation = self._get_state()

        # Check if done
        terminated = False
        if flow_time > self.tmax:
            terminated = True

        self.act_index += 1
        if self.act_index >= self.nb_interactions:
            print(f"[STEP] ACT_INDEX={self.act_index}; TERMINATED == TRUE", flush=True)
            terminated = True

        truncated = False  # Nek doesn't use truncation

        info = {
            "time": flow_time,
            "reward_per_actuator": rewards_per_actuator,
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

        # Receive state from each node
        state_buffer = np.ndarray(shape=(self.nNID, NFLDC, TOTCTRL), dtype=np.float64)
        for ni, nid in enumerate(self.uniqID):
            node_buffer = np.ndarray(shape=(NFLDC, TOTCTRL), dtype=np.float64)
            for t in range(NFLDC):
                buffer = np.ndarray(shape=(TOTCTRL), dtype=np.float64)
                self.sub_comm.Recv(
                    [buffer, tag_dict["STATE"]["mpi_dtype"]],
                    nid,
                    tag=nid * (t + 1) + tag_dict["STATE"]["tag"],
                )
                node_buffer[t, :] = buffer[:]
            state_buffer[ni, :, :] = self._normalize_state(node_buffer)

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

        # Send actions to each node
        icount = 0
        for il, nid in enumerate(self.uniqID):
            _index = np.where((self.actuator_info["NID"] == nid))[0]

            # Create action buffer for this node
            act_buffer = np.ndarray(
                shape=(self.TOTCTRL,), dtype=tag_dict["ACTION"]["py_dtype"]
            )
            for jl in range(len(_index)):
                act_buffer[jl] = action[icount]
                icount += 1

            # Send buffer
            self.sub_comm.Send(
                [act_buffer, tag_dict["ACTION"]["mpi_dtype"]],
                nid,
                tag=nid + tag_dict["ACTION"]["tag"],
            )

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
                    f"[WARNING] {i_evolv}/{self.ndrl} Current CFL "
                    f"{current_cfl} >= {self.target_cfl}!",
                    flush=True,
                )
                self._end_simulation(farewell=True)
                exit()

            # Receive reward buffer at last step
            if i_evolv == self.ndrl:
                ws_stress_buffer = np.ndarray(
                    shape=(self.nNID, self.TOTCTRL), dtype=tag_dict["REWRD"]["py_dtype"]
                )
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
            f"[LOGGER] act_index={self.act_index} dUdy={np.mean(ws_stress_buffer):.5f} "
            f"R={np.mean(rewards):.5f}",
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
                state = (
                    2 * (state - np.min(state)) / (np.max(state) - np.min(state)) - 1
                )
        return state

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward."""
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
                shutil.copy(rsfile, dst=self.mpi_info["wdir"] + "/")
                print(f"[RSTART] RESET: {rsfile}", flush=True)

        elif self.random_init == -1:
            print(f"[RSTART] NOT SHUFFLE; RANK={self.rank}", flush=True)
            target_folder = os.path.join(self.rstart_folder, f"init_{self.rank}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if "rs" in f]
            for rsfile in rs_list:
                rsfile = os.path.join(target_folder, rsfile)
                shutil.copy(rsfile, dst=self.mpi_info["wdir"] + "/")
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
        return (
            f"jet_np{nid:08d}_"
            f"gid{gllid:08d}_"
            f"iface{iface}_"
            f"ix{ix:08d}_"
            f"iy{iy:08d}_"
            f"iz{iz:08d}"
        )

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
        if type(dim) is int:
            dim = (dim,)
        self.data = np.zeros((length,) + dim, dtype="f")
        self.index = 0

    def extend(self, x):
        """Add array x to ring buffer."""
        assert (
            x.shape == self.data.shape[1:]
        ), "Input array does not match the ring buffer size"
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
}
