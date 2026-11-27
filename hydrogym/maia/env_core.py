"""
Core MaiaGym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import omegaconf
import toml
from einops import rearrange
from mpi4py import MPI

from hydrogym.data_manager import HFDataManager
from hydrogym.hf_env_mixin import ConfigError, HFEnvConfigMixin
from hydrogym.maia.mpmd_interface import MaiaInterface
from hydrogym.power import PowerConfig, power_info


class MaiaFlowEnv(HFEnvConfigMixin, gym.Env):
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

    SOLVER_TYPE: str = "MAIA_LB"
    # Actuation power (set per instance in __init__ / setup_actuation_power; class defaults keep instances built
    # without __init__, e.g. in tests, valid: no power model, no power cost).
    power_cfg: PowerConfig = PowerConfig()
    _power_model = None
    _power_kind: Optional[str] = None
    _power_prev_commands = None
    _power_invalid_warned = False

    # Per-backend cache directory under ~/.cache
    HF_CACHE_NAMESPACE = "maiagym"

    @staticmethod
    def _resolve_num_substeps(cfg_section) -> int:
        """
        Resolve the per-actuation substep count from the environment's YAML
        config section (e.g. cfg.maia).

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
                - obs_normalization_seed (int): Optional. Seed of the random actions used to compute the
                    'probewise_mean_std' factors; set it (or store the printed factors and use 'customized')
                    to make the normalisation reproducible. Default: None (not reproducible).
                - obs_loc (list): Required if strategy is 'customized'.
                - obs_scale (list): Required if strategy is 'customized'.
                - nproc (int): Optional. Expected number of m-AIA solver
                    ranks in the MPMD launch. When given, validated against
                    the actual MPI world at construction time; a mismatch
                    raises a clear ``RuntimeError`` naming the fix, instead
                    of failing confusingly downstream. Omit to skip this
                    check (default).

        Raises:
            ConfigError: If required configuration is missing or invalid.
            RuntimeError: If ``nproc`` is given and does not match the
                actual number of m-AIA solver ranks in the MPMD job.
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

        # Standard initialization
        self.is_testing = env_config.get("is_testing", False)
        self.probe_locations = env_config.get("probe_locations")
        if self.probe_locations is None:
            raise ConfigError(
                "'probe_locations' must be specified in env_config -- there is no "
                "universal default (the sensible probe grid is a per-flow-geometry "
                "choice). Example for a 2D flow: probe_locations=[x0, y0, x1, y1, ...] "
                "(flattened coordinate pairs); see MaiaFlowEnv's docstring / "
                "examples/maia/getting_started/test_maia_env.py's create_probe_locations "
                "for a worked example."
            )
        self.obs_normalization_strategy = env_config.get("obs_normalization_strategy", "none")
        # seed of the random actions of the 'probewise_mean_std' normalisation run (None: not reproducible)
        self.obs_normalization_seed = env_config.get("obs_normalization_seed")

        valid_strategies = ["U_inf", "probewise_mean_std", "none", "customized"]
        if self.obs_normalization_strategy not in valid_strategies:
            raise ConfigError(
                f"Invalid obs_normalization_strategy: '{self.obs_normalization_strategy}'. "
                f"Must be one of: {valid_strategies}"
            )

        if self.obs_normalization_strategy == "customized":
            self.obs_loc = env_config.get("obs_loc")
            self.obs_scale = env_config.get("obs_scale")
            if self.obs_loc is None or self.obs_scale is None:
                raise ConfigError(
                    "obs_normalization_strategy='customized' requires both 'obs_loc' and 'obs_scale' "
                    "to be provided in env_config"
                )

        # Load configuration
        self.cfg = omegaconf.OmegaConf.load(self.configuration_file)

        # Update paths in configuration to use downloaded data
        self._update_configuration_paths()

        self.runtime_property_file = os.path.join(self.env_data_path, "properties_run.toml")

        self.num_substeps_per_iteration = self._resolve_num_substeps(self.cfg.maia)
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

        # Actuation power (hydrogym.power): reported in info; charged in the reward only if power_weight > 0.
        self.power_cfg = PowerConfig.from_config(
            omegaconf.OmegaConf.to_container(self.cfg.maia, resolve=True),
            {k: env_config[k] for k in ("power_weight", "power_cost", "power_sampling") if k in env_config},
        )
        if self.power_cfg.resolve_sampling("end") != "end":
            raise ConfigError(
                "m-AIA environments sample the actuation power at the end of each action interval only "
                "(power_sampling: end)"
            )
        self._power_model = None
        self._power_kind = None

        # Read property file and extract parameters
        self.runtime_property_file_data = self._read_property_file(self.runtime_property_file)
        self.Ma = self._get_property(self.runtime_property_file_data, "Ma")
        self.maxRfnmntLvl = self._get_property(self.runtime_property_file_data, "maxRfnmntLvl")
        self.reductionFactor = self._get_property(self.runtime_property_file_data, "reductionFactor")
        self.domainLength = self._get_property(self.runtime_property_file_data, "domainLength")
        self.referenceLength = self._get_property(self.runtime_property_file_data, "referenceLength")
        self.nDim = self._get_property(self.runtime_property_file_data, "nDim")
        self.dX = self.reductionFactor * self.domainLength / (2**self.maxRfnmntLvl)
        self.bcId = self._get_property(self.runtime_property_file_data, "lbBndCndIdx")
        self.Re = self._get_property(self.runtime_property_file_data, "Re")
        self.zLength = self._get_property(self.runtime_property_file_data, "zLength") if self.nDim == 3 else self.dX
        self._check_cell_size_against_grid()

        # Initialize MPI communication
        self.comm_world = MPI.COMM_WORLD
        self.maiaInterface = MaiaInterface(self.nDim)
        # Optional launch-config validation: when the caller states how many MAIA solver ranks it expects,
        # catch a wrong `-np` in the MPMD launch command here instead of
        # failing confusingly (or silently misbehaving) downstream.
        self.maiaInterface.init_comm(self.comm_world, nproc=env_config.get("nproc"))
        print("Python communicator initialized", flush=True)

        if self.Re != self.cfg.maia.Re:
            raise ConfigError(
                f"Re numbers of configuration file (Re={self.cfg.maia.Re}) and "
                f"property file (Re={self.Re}) do not match! Adjustment required!"
            )

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

    def configure_observations(self) -> None:
        """
        Configure the number of observations based on observation type and probes.

        Sets self.num_outputs based on which observation types are enabled
        (forces, u, v, w, rho, p) and the number of probes.
        """
        self.num_outputs = 0
        self.num_probes = int(len(self.probe_locations) / self.nDim)

        if "forces" in self.observation_type:
            self.num_outputs += self.nDim * len(self.bcId)
        if "u" in self.observation_type:
            self.num_outputs += self.num_probes
        if "v" in self.observation_type:
            self.num_outputs += self.num_probes
        if "w" in self.observation_type:
            self.num_outputs += self.num_probes
        if "rho" in self.observation_type:
            self.num_outputs += self.num_probes
        if "p" in self.observation_type:
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
        if self.obs_normalization_strategy == "U_inf":
            obs_loc, obs_scale = [], []

            if "u" in self.observation_type:
                obs_loc.append([0.0] * self.noProbes)
                obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)
            if "v" in self.observation_type:
                obs_loc.append([0.0] * self.noProbes)
                obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)

            if self.nDim == 2:
                if "rho" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.cfg.maia.rho_inf] * self.noProbes)
                if "p" in self.observation_type:
                    dynamic_pressure = 0.5 * self.cfg.maia.rho_inf * self.cfg.maia.U_inf**2
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([dynamic_pressure] * self.noProbes)
            elif self.nDim == 3:
                if "w" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.cfg.maia.U_inf] * self.noProbes)
                if "rho" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.cfg.maia.rho_inf] * self.noProbes)
                if "p" in self.observation_type:
                    dynamic_pressure = 0.5 * self.cfg.maia.rho_inf * self.cfg.maia.U_inf**2
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([dynamic_pressure] * self.noProbes)
            else:
                print(f"WARNING: nDim = {self.nDim} > 3. Something must be wrong!")

            if "forces" in self.observation_type:
                for _ in range(len(self.bcId)):
                    obs_loc.append([0.0] * self.nDim)
                    obs_scale.append([1.0] * self.nDim)

            self.obs_loc = np.concatenate(obs_loc)
            self.obs_scale = np.concatenate(obs_scale)

        elif self.obs_normalization_strategy == "none":
            self.obs_loc = [0.0] * self.num_outputs
            self.obs_scale = [1.0] * self.num_outputs

        elif self.obs_normalization_strategy == "probewise_mean_std":
            print("WARNING: Selected Obs Normalization strategy: probewise_mean_std")
            print("Computing normalization factors now...")
            print("If pre-computed normalization factors exist, select:")
            print("obs_normalization_strategy = 'customized' and provide normalization factors!")
            self.compute_normalization_factors()
            print("Computed loc values:", self.obs_loc.tolist(), flush=True)
            print("Computed scale values:", self.obs_scale.tolist(), flush=True)

        elif self.obs_normalization_strategy == "customized":
            if self.obs_scale is not None and self.obs_loc is not None:
                if len(self.obs_loc) == self.num_outputs and len(self.obs_scale) == self.num_outputs:
                    print("Using customized normalization values", flush=True)
                else:
                    raise ConfigError(
                        f"Customized normalization dimensions mismatch: "
                        f"obs_loc has {len(self.obs_loc)} elements, "
                        f"obs_scale has {len(self.obs_scale)} elements, "
                        f"but num_outputs is {self.num_outputs}"
                    )
        else:
            raise ConfigError(
                f"Invalid obs_normalization_strategy: '{self.obs_normalization_strategy}'. "
                f"This should have been caught earlier."
            )

    def compute_normalization_factors(self, zero_actuation: bool = False) -> None:
        """
        Compute normalization factors for observations by running simulation episodes.

        Runs several episodes with random (or zero) actuation to collect probe data,
        then computes mean and standard deviation for normalization.

        Args:
            zero_actuation: If True, use zero actuation. Otherwise, use random actuation.
        """
        obs_norm_episodes = 5
        rng = np.random.default_rng(self.obs_normalization_seed)
        probes = np.zeros(shape=(obs_norm_episodes * self.max_episode_steps, int(self.noProbes * self.noProbeVars)))

        for i in range(obs_norm_episodes * self.max_episode_steps):
            if i % self.max_episode_steps == 0 and i > 0:
                # Reset environment
                self.maiaInterface.runTimeSteps(1)
                self.maiaInterface.reinit()
                self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))
                self.maiaInterface.continueRun()

            self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)

            if zero_actuation:
                self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))
            else:
                self.maiaInterface.setControlProperties(
                    self.convert_action(
                        action=rng.uniform(-self.MAX_CONTROL, self.MAX_CONTROL, size=self.num_inputs)
                    )
                )

            probes[i, :] = self._read_probes()
            self._power_probe_data = None
            self.maiaInterface.continueRun()

        loc = np.mean(probes, axis=0)
        scale = np.std(probes, axis=0)
        # a probe that stays constant (std 0) would give inf/NaN observations: leave it unscaled
        scale = np.where(scale < 1e-12, 1.0, scale)

        loc = rearrange(loc, "(n p) -> n p", n=self.noProbes)
        scale = rearrange(scale, "(n p) -> n p", n=self.noProbes)

        self.obs_loc, self.obs_scale = [], []

        if "u" in self.observation_type:
            self.obs_loc.append(loc[:, 0])
            self.obs_scale.append(scale[:, 0])
        if "v" in self.observation_type:
            self.obs_loc.append(loc[:, 1])
            self.obs_scale.append(scale[:, 1])

        if self.nDim == 2:
            if "rho" in self.observation_type:
                self.obs_loc.append(loc[:, 2])
                self.obs_scale.append(scale[:, 2])
            if "p" in self.observation_type:
                self.obs_loc.append(loc[:, 3])
                self.obs_scale.append(scale[:, 3])
        elif self.nDim == 3:
            if "w" in self.observation_type:
                self.obs_loc.append(loc[:, 2])
                self.obs_scale.append(scale[:, 2])
            if "rho" in self.observation_type:
                self.obs_loc.append(loc[:, 3])
                self.obs_scale.append(scale[:, 3])
            if "p" in self.observation_type:
                self.obs_loc.append(loc[:, 4])
                self.obs_scale.append(scale[:, 4])

        self.obs_loc = np.concatenate(self.obs_loc)
        self.obs_scale = np.concatenate(self.obs_scale)

    @classmethod
    def create_from_hf_env(
        cls,
        environment_name: str,
        hf_repo_id: str = "dynamicslab/HydroGym-environments",
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

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance the state of the environment by one step.

        Args:
            action: Action array with values in [-1, 1], scaled by MAX_CONTROL.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            terminated is always False (no in-band physics-failure signal);
            truncated becomes True once more than max_episode_steps steps have been
            taken, so an episode has max_episode_steps + 1 steps.
        """
        action = [a * self.MAX_CONTROL for a in action]

        # m-AIA answers control / probe requests only at the start of a substep segment (LbSolver::callMPMDRoutines).
        # The action is set while m-AIA waits there, the segment then runs N substeps with it, and the probes are read
        # at the start of the next segment: the observation is the state at the END of the interval the action acted
        # on (gym convention).
        commands = self.convert_action(action=action)
        self.maiaInterface.setControlProperties(commands)
        self.maiaInterface.continueRun()
        self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)

        self.probeData = self._read_probes()
        self.probeData = rearrange(self.probeData, "(n p) -> n p", n=self.noProbes)

        self.obs = []
        if "u" in self.observation_type:
            self.obs.append(self.probeData[:, 0])
        if "v" in self.observation_type:
            self.obs.append(self.probeData[:, 1])

        if self.nDim == 2:
            if "rho" in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if "p" in self.observation_type:
                self.obs.append(self.probeData[:, 3])
        elif self.nDim == 3:
            if "w" in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if "rho" in self.observation_type:
                self.obs.append(self.probeData[:, 3])
            if "p" in self.observation_type:
                self.obs.append(self.probeData[:, 4])
        else:
            print(f"WARNING: nDim = {self.nDim} > 3. Something must be wrong!")

        if "forces" in self.observation_type:
            forces = []
            for bc_id in self.bcId:
                forces.append(self.maiaInterface.getForce(bc_id))
            self.obs.append(np.stack(forces).flatten())

        self.obs = np.concatenate(self.obs)
        reward, obj_dict = self.get_reward()
        info = self._power_info(commands, obj_dict)
        reward = self._charge_power(reward, info)
        self.obs = (self.obs - self.obs_loc) / self.obs_scale

        self.iter += 1
        # Reaching the step budget is a TRUNCATION, not a
        # termination -- terminated is reserved for physics failure. (The
        # MAIA solver exposes no in-band divergence signal today; a solver-
        # side crash surfaces as an MPI communication error, not a flag.)
        done = self.check_complete()

        return self.obs, reward, False, bool(done), info

    def _seed_reset(self, seed: Optional[int]) -> None:
        """Seed the gymnasium RNG and the action space (call at the start of every ``reset``).

        The solver itself is deterministic, so the seed only affects what is drawn on the Python side (for example
        ``action_space.sample()``). Without a seed both generators keep their state, so repeated resets continue the
        same random stream. ``gym.Env.reset`` is called directly because ``reset`` is overridden by the subclasses.
        """
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options.

        Returns:
            Tuple of (initial_observation, info).
        """
        print("Resetting environment", flush=True)
        self._seed_reset(seed)

        # Between calls m-AIA waits in its request loop at the start of the next segment (see step()).  Leave it first
        # (after a step), open an N-substep segment, reinitialise, and stay in the request loop (no continueRun).
        if getattr(self, "_in_request_loop", False):
            self.maiaInterface.continueRun()
        self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)
        self.maiaInterface.reinit()

        self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))
        self.probeData = self._read_probes()
        self._power_probe_data = None  # the power is sampled in step() only
        self.probeData = rearrange(self.probeData, "(n p) -> n p", n=self.noProbes)

        self.obs = []
        if "u" in self.observation_type:
            self.obs.append(self.probeData[:, 0])
        if "v" in self.observation_type:
            self.obs.append(self.probeData[:, 1])

        if self.nDim == 2:
            if "rho" in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if "p" in self.observation_type:
                self.obs.append(self.probeData[:, 3])
        elif self.nDim == 3:
            if "w" in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if "rho" in self.observation_type:
                self.obs.append(self.probeData[:, 3])
            if "p" in self.observation_type:
                self.obs.append(self.probeData[:, 4])
        else:
            print(f"WARNING: nDim = {self.nDim} > 3. Something must be wrong!")

        if "forces" in self.observation_type:
            forces = []
            for bc_id in self.bcId:
                forces.append(self.maiaInterface.getForce(bc_id))
            self.obs.append(np.stack(forces).flatten())

        self.obs = np.concatenate(self.obs)
        self.obs = (self.obs - self.obs_loc) / self.obs_scale
        self._in_request_loop = True

        self.iter = 0
        info = self._power_info(None, None)

        return self.obs, info

    # ------------------------------------------------------------------ actuation power
    def setup_actuation_power(
        self, kind: Optional[str], geometry: Optional[str] = None, model_options: Optional[Dict] = None
    ) -> None:
        """Build the actuation-power model of this environment (call at the end of the subclass constructor).

        Args:
            kind: ``"rotation"`` (rotating circular bodies, BC 2008), ``"slot_jets"`` (slot jets on circular bodies,
                BC 2007), ``"wall_jets"`` (BC 2007 jets on other walls, ``hydrogym.maia.wall_jets``), ``"solver"``
                (power coefficient ``C_P`` returned by the solver in the objective dict, e.g. the structured
                finite-volume surface-wave cases), ``"solver_jets"`` (jets of the structured finite-volume solver,
                BC 1008: power per jet and cost forms formed per wall face by the solver,
                ``MaiaInterface.getActuationPower``) or ``None`` (no power model; the info keys are NaN).
            geometry: for ``"wall_jets"``: ``"flat"`` (jets on flat walls along the lattice axes), ``"curve"`` (2D
                wall from the case's BC-2007 geometry file, extruded in 3D) or ``"sphere"`` (object class 3).
            model_options: keyword arguments of ``hydrogym.maia.wall_jets.WallJetPower`` (e.g.
                ``wall_distances``).
        """
        from hydrogym.maia import power as mpower

        props = self.runtime_property_file_data
        common = dict(
            n_dim=self.nDim,
            dX=self.dX,
            Ma=self.Ma,
            Re=self.Re,
            reference_length=self.referenceLength,
            z_length=(self.zLength if self.nDim == 3 else self.dX),
        )
        self._power_kind = kind
        self._power_model = None
        if kind == "rotation":
            geo = mpower.rotating_bodies(props, self.num_inputs)
            if geo is not None:
                centers, radii, zc, signs = geo
                self._power_model = mpower.RingTorquePower(
                    centers,
                    radii,
                    lattice_origin=self._lattice_origin(),
                    axis_signs=signs,
                    z_center=float(np.mean(zc)),
                    **common,
                )
        elif kind == "slot_jets":
            n_jets = int(self._get_property(props, "lbNoJets"))
            geo = mpower.jets_on_bodies(props, n_jets)
            if geo is not None:
                centers, radii, angles, widths = geo
                zc = (
                    float(np.asarray(props["lbJetCenter"], dtype=float).reshape(n_jets, self.nDim)[:, -1].mean())
                    if self.nDim == 3
                    else 0.0
                )
                self._power_model = mpower.SlotJetPower(
                    centers, radii, angles, widths, lattice_origin=self._lattice_origin(), z_center=zc, **common
                )
        elif kind == "wall_jets":
            self._power_model = self._wall_jet_model(geometry, **(model_options or {}))
        elif kind == "solver":
            self._power_model = "solver"
        elif kind == "solver_jets":
            self._power_model = "solver_jets"
        elif kind is not None:
            raise ValueError(f"unknown actuation power model {kind!r}")
        if self._power_model is None and self.power_cfg.weight > 0:
            raise ConfigError(
                f"power_weight > 0 but {type(self).__name__} has no actuation-power model for this configuration"
            )

    def _check_cell_size_against_grid(self) -> None:
        """Stop with a clear error if the property file's dX does not match the grid file (if the grid is present)."""
        from hydrogym.maia.power import check_cell_size

        grid = os.path.join(self.env_data_path, "grid.Netcdf")
        if not os.path.exists(grid):
            return  # grids generated at run time (out_lb/) are written by m-AIA from these same properties
        try:
            check_cell_size(grid, self.dX, source=self.runtime_property_file)
        except ValueError as err:
            raise ConfigError(str(err)) from None

    def _grid_file(self) -> str:
        """Path of the m-AIA grid file of the case (environment data, or ``out_lb`` of the run directory)."""
        for cand in (
            os.path.join(self.env_data_path, "grid.Netcdf"),
            os.path.join(self.env_data_path, "out_lb", "grid.Netcdf"),
            os.path.join(os.getcwd(), "out_lb", "grid.Netcdf"),
        ):
            if os.path.exists(cand):
                return cand
        raise ConfigError("actuation power: grid.Netcdf of the environment not found (needed for the cell lattice)")

    def _lattice_origin(self) -> np.ndarray:
        """Origin of the finest-level cell-centre lattice, from the grid file of the environment."""
        from hydrogym.maia.power import lattice_origin_from_grid

        return lattice_origin_from_grid(self._grid_file(), self.dX, self.nDim)

    def power_projection(self) -> float:
        """Projection length (2D, cells) or area (3D, cells^2) of the drag coefficient, used for ``C_P``.

        Default: ``referenceLength / dX * zLength / dX`` (``zLength = dX`` in 2D), as in the force coefficients.
        """
        return self.referenceLength / self.dX * self.zLength / self.dX

    def _wall_jet_model(self, geometry: Optional[str], **options):
        """``hydrogym.maia.wall_jets.WallJetPower`` of the BC-2007 jets of this case."""
        from hydrogym.maia import wall_jets as wj
        from hydrogym.maia.power import read_cdf_global_attributes

        props = self.runtime_property_file_data
        jets = wj.Bc2007Jets.from_properties(props)
        # finest cell size and lattice origin from the grid itself (the power model uses the solver's lattice, defined
        # by the grid file)
        att = read_cdf_global_attributes(self._grid_file())
        dX = float(att["lengthLevel0"]) / 2 ** int(att["maxLevel"])
        origin = np.asarray(att["centerOfGravity"], dtype=float).reshape(-1)[: self.nDim] - 0.5 * float(
            att["lengthLevel0"]
        )
        l_proj = self.power_projection() * (self.dX / dX) ** (self.nDim - 1)
        if abs(dX / self.dX - 1.0) > 1e-6:
            import warnings

            warnings.warn(
                f"{type(self).__name__}: the grid's finest cell size {dX:.6g} differs from the property file's "
                f"{self.dX:.6g} (domainLength / 2^maxRfnmntLvl); the actuation power uses the grid's",
                stacklevel=2,
            )
        n_dist = int(props.get("noDistributions", 9 if self.nDim == 2 else 27))
        span = z_planes = None
        span_weight = 1.0
        if self.nDim == 3:
            zc = float(np.asarray(read_cdf_global_attributes(self._grid_file())["centerOfGravity"])[2])
            span = (zc - 0.5 * self.zLength, zc + 0.5 * self.zLength)
            n_planes = int(options.pop("n_planes", 16))
            k = np.arange(n_planes)
            raw = span[0] + (k + 0.5) / n_planes * self.zLength
            z_planes = origin[2] + (np.floor((raw - origin[2]) / dX) + 0.5) * dX
            span_weight = self.zLength / dX / n_planes
        else:
            options.pop("n_planes", None)
        if geometry == "flat":
            el = wj.flat_wall_jets(jets, dX, origin, span=span)
        elif geometry == "curve":
            files = wj.geometry_files(os.path.join(self.env_data_path, "geometry.toml"))
            if not files:
                raise ConfigError("actuation power: no BC-2007 geometry file in geometry.toml")
            if self.nDim == 2:
                segs = np.concatenate([wj.read_polyline_stl(f) for f in files])
            else:
                # section at the cell-centre plane nearest mid-span (not a vertex plane of the surface mesh)
                zs = origin[2] + (np.floor((0.5 * (span[0] + span[1]) - origin[2]) / dX) + 0.5) * dX
                segs = np.concatenate([wj.read_stl_section(f, zs) for f in files])
            el = wj.curve_link_jets(jets, segs, dX, origin, n_dist=n_dist, z_planes=z_planes, span_weight=span_weight)
        elif geometry == "sphere":
            el = wj.sphere_link_jets(jets, dX, origin, n_dist=n_dist)
            options.setdefault("wall_distances", (2.0, 3.0, 4.0, 5.0))
        else:
            raise ValueError(f"unknown wall-jet geometry {geometry!r}")
        return wj.WallJetPower(el, jets.n_jets, origin, dX, self.Ma, l_proj, **options)

    def drag_coefficient_from_objective(self, obj_dict: Optional[Dict]) -> Optional[float]:
        """Total drag coefficient of the controlled bodies from the objective dict of ``get_reward``.

        Subclasses override this; the default returns None (no drag reported).
        """
        return None

    def _power_info(self, commands, obj_dict: Optional[Dict]) -> Dict:
        """Actuation-power entries of ``info`` (hydrogym.power); ``commands`` None means no actuation (reset)."""
        if commands is not None:
            n = len(commands)
        elif self._power_kind == "slot_jets" and self._power_model is not None:
            n = len(self._power_model.body_centers)
        elif self._power_kind == "wall_jets" and self._power_model is not None:
            n = self._power_model.n_jets
        else:
            n = self.num_inputs
        if self._power_model is None:
            per = np.full(n, np.nan)
            forms = {"signed": np.nan, "positive": np.nan, "abs": np.nan}
        elif commands is None:
            per = np.zeros(1 if self._power_kind == "solver" else n)
            forms = None
        elif self._power_kind == "solver":
            per = np.array([float(obj_dict["C_P"])])
            forms = None
        elif self._power_kind == "solver_jets":
            # [P, P_pos, P_abs, P_0..P_{N-1}, Q_0..Q_{N-1}] (m-AIA StructuredBndryCnd3D::computeJetPower1008)
            data = np.asarray(self.maiaInterface.getActuationPower(), dtype=float)
            n_jets = (len(data) - 3) // 2
            if n_jets != n or len(data) != 3 + 2 * n_jets:
                raise RuntimeError(
                    f"actuation power: m-AIA returned {len(data)} values for {n} jets (expected {3 + 2 * n}); "
                    "the m-AIA build predates the jet power of boundary condition 1008 or fvNoJets does not match "
                    "the number of actions"
                )
            per = data[3 : 3 + n_jets]
            forms = {"signed": data[0], "positive": data[1], "abs": data[2]}
        else:
            data = getattr(self, "_power_probe_data", None)
            if data is None:  # not sampled with the observation probes (e.g. a subclass with its own step)
                data = self.maiaInterface.getProbeData(list(self._power_model.probe_coordinates))
            self._power_probe_data = None  # one sample per step
            ev = self._power_model.evaluate(data, commands)
            per, forms = ev["per_actuator"], ev["forms"]
        info = power_info(
            per,
            sampling="end",
            forms=forms,
            drag_coefficient=self.drag_coefficient_from_objective(obj_dict) if obj_dict is not None else None,
            objective=obj_dict,
        )
        info["power_model"] = self._power_kind
        # self-check of the probe-based models: spread over the circles (rotation) or between linear and quadratic
        # wall extrapolation (jets), relative to the value; NaN where not applicable
        spread = np.nan
        if commands is not None and self._power_kind == "rotation" and self._power_model is not None:
            spread = float(np.max(self._power_model.last_spread))
        elif commands is not None and self._power_kind in ("slot_jets", "wall_jets") and self._power_model is not None:
            spread = float(self._power_model.last_extrapolation_spread)
        info["actuation_power_model_spread"] = spread
        # validity: the ring torque needs a (nearly) steady annulus -> flag actions whose rotation command changed by
        # more than ROTATION_STEADY_TOL of the bound (the jet and solver models have no such restriction)
        valid = self._power_model is not None
        if commands is None:
            self._power_prev_commands = None
        elif self._power_kind == "rotation" and self._power_model is not None:
            from hydrogym.maia.power import ROTATION_STEADY_TOL, rotation_change

            change = rotation_change(self._power_prev_commands, commands, self.MAX_CONTROL)
            info["actuation_power_command_change"] = change
            valid = change <= ROTATION_STEADY_TOL
            self._power_prev_commands = list(commands)
        info["actuation_power_valid"] = bool(valid)
        return info

    def _charge_power(self, reward, info: Dict):
        """Reward minus power_weight * C_P (form power_cost); unchanged for power_weight 0."""
        if self.power_cfg.weight == 0.0:
            return reward
        if not np.isscalar(reward):
            raise ConfigError("power_weight > 0 needs a scalar reward (one boundary segment or a summed reward)")
        key = {"signed": "actuation_power", "positive": "actuation_power_positive", "abs": "actuation_power_abs"}
        if not info.get("actuation_power_valid", True) and not self._power_invalid_warned:
            import warnings

            warnings.warn(
                "actuation power charged in the reward although the sample is flagged as not valid (the rotation "
                "changed by more than 10 % of the bound within the action; the ring torque assumes a steady "
                "annulus). Use smoother actuation (smaller action changes) or treat the power cost with care.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._power_invalid_warned = True
        return reward - self.power_cfg.weight * info[key[self.power_cfg.cost]]

    def compute_nondim_coefficients(
        self,
        forces: np.ndarray,
        density: float = 1.0,
        referenceVelocity: float = 0.1 / np.sqrt(3),
        projectionLength: float = 20.0,
    ) -> np.ndarray:
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
        force_coefficients = (2 * forces) / (density * referenceVelocity**2 * projectionLength)
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
            True once ``iter > max_episode_steps`` (an episode has ``max_episode_steps + 1`` steps).
        """
        return self.iter > self.max_episode_steps

    def close(self) -> None:
        """
        Close the environment and signal maia to finish.

        This sends a finish signal to the maia solver to properly
        terminate the MPMD coupling.
        """
        print("ENV: Closing environment and signaling maia to finish...", flush=True)
        try:
            if getattr(self, "_in_request_loop", False):  # leave the request loop before the finish signal
                self.maiaInterface.continueRun()
                self._in_request_loop = False
            self.maiaInterface.finishRun()
            print("ENV: Finish signal sent to maia", flush=True)
        except Exception as e:
            print(f"ENV: Error during close: {e}", flush=True)

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

    def _probe_power_points(self) -> Optional[np.ndarray]:
        """Flat coordinates of the probe-based power model (rotation / slot jets / wall jets), else None."""
        if self._power_kind in ("rotation", "slot_jets", "wall_jets") and self._power_model is not None:
            return np.asarray(self._power_model.probe_coordinates, dtype=float).ravel()
        return None

    def _read_probes(self):
        """Request the observation probes and, if present, the power-model points in ONE list; return the
        observation-probe values and keep the power-model values for ``_power_info``.

        One fixed list (same points, same order every request) lets m-AIA reuse its cached probe cell ids; requesting
        the two lists separately would replace the cache at every request. Each point is sampled independently, so the
        values are the same as for separate requests.
        """
        self._power_probe_data = None
        power_points = self._probe_power_points()
        if power_points is None:
            return self.maiaInterface.getProbeData(self.probe_locations)
        coords = list(self.probe_locations) + power_points.tolist()
        data = self.maiaInterface.getProbeData(coords)
        if data is None:  # non-root controller rank: m-AIA answers the root only
            return data
        data = np.asarray(data)
        n_obs = len(data) * len(self.probe_locations) // len(coords)  # values per point x observation points
        self._power_probe_data = data[n_obs:]
        return data[:n_obs]

    def _read_property_file(self, property_file_path: str) -> Dict:
        """
        Read a TOML property file.

        Args:
            property_file_path: Path to the TOML file.

        Returns:
            Dictionary containing the parsed TOML data.
        """
        with open(property_file_path, "r") as f:
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

    def _update_property(self, property_file_data: Dict, key: Union[str, List[str]], value) -> None:
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

    def _write_property_file(self, property_file_data: Dict, property_file_path: str) -> None:
        """
        Write property file data back to a TOML file.

        Args:
            property_file_data: Dictionary to write.
            property_file_path: Path to the output file.
        """
        with open(property_file_path, "w") as f:
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


def from_hf(environment_name: str, hf_repo_id: str = "dynamicslab/HydroGym-environments", **kwargs) -> MaiaFlowEnv:
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
        >>> env = maiaGym.from_hf('Cavity_2D_Re4200')
    """
    # Parse environment type from name
    if "_" not in environment_name:
        available_types = list(_ENVIRONMENT_REGISTRY.keys())
        raise ValueError(
            f"Invalid environment name '{environment_name}'. "
            f"Expected format: 'TYPE_PARAMETERS' (e.g., 'Cylinder_2D_Re200'). "
            f"Available types: {available_types}"
        )

    env_type = environment_name.split("_", 1)[0]

    # Look up environment class
    if env_type not in _ENVIRONMENT_REGISTRY:
        available_types = list(_ENVIRONMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment type '{env_type}' from name '{environment_name}'. "
            f"Available types: {available_types}. "
            f"If this is a new environment type, make sure the corresponding "
            f"HF environment class is imported and registered."
        )

    env_class = _ENVIRONMENT_REGISTRY[env_type]

    # Create environment configuration
    env_config = {"environment_name": environment_name, "hf_repo_id": hf_repo_id, **kwargs}

    try:
        return env_class(env_config)
    except Exception as e:
        raise ConfigError(f"Failed to create environment '{environment_name}' of type '{env_type}': {e}") from e


def list_available_environments(
    hf_repo_id: str = "dynamicslab/HydroGym-environments",
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
) -> List[str]:
    """
    List all available environments from HF Hub.

    Args:
        hf_repo_id: Hugging Face repository ID.
        hf_token: Hugging Face access token (private/gated repos; ``None`` = ambient auth).
        hf_revision: Git revision to pin the file listing to.

    Returns:
        List of environment names.
    """
    data_manager = HFDataManager(repo_id=hf_repo_id, token=hf_token, revision=hf_revision)
    return data_manager.get_available_environments()


def list_registered_types() -> Dict[str, type]:
    """
    List all registered environment types.

    Returns:
        Dictionary mapping environment types to their classes.
    """
    return dict(_ENVIRONMENT_REGISTRY)
