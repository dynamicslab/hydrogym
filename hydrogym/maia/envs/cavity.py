"""
Cavity Flow Environment
========================

This module provides cavity flow CFD environments for reinforcement learning,
designed for flow control applications in cavity configurations.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class CavityBase(MaiaFlowEnv):
    """
    Base class for cavity flow environments with Hugging Face integration.

    This environment implements two reward strategies:
    - 'baseline_mean': Penalize deviation from a pre-computed baseline state
    - 'running_mean': Penalize deviation from a running average

    Attributes:
        reward_strategy: Strategy for reward computation ('baseline_mean' or 'running_mean').
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the cavity base environment.

        Args:
            env_config: Environment configuration dictionary containing:
                - reward_strategy (str): 'baseline_mean' or 'running_mean'. Default: 'baseline_mean'
        """
        super().__init__(env_config)

        self.reward_strategy = env_config.get("reward_strategy", "baseline_mean")
        self._env_config = env_config
        self._baseline_loaded = False

    def _find_baseline_file(self) -> Optional[str]:
        """
        Auto-detect baseline file in the environment data directory.

        Returns:
            Path to baseline file, or None if not found.
        """
        file_names = [
            "baseline_state.feather",
            f"{self.environment_name}.feather",
        ]

        for name in file_names:
            file_path = os.path.join(self.env_data_path, name)
            if os.path.exists(file_path):
                print(f"Auto-detected baseline file: {name}")
                return file_path

        config_patterns = ["baseline_state.feather", "baseline_*.feather"]

        for pattern in config_patterns:
            matches = glob.glob(os.path.join(self.env_data_path, pattern))
            if matches:
                print(f"Auto-detected baseline file: {os.path.basename(matches[0])}")
                return matches[0]

        print(f"WARNING: No baseline state file auto-detected in {self.env_data_path}")
        if os.path.exists(self.env_data_path):
            print(f"Available files: {os.listdir(self.env_data_path)}")

        return None

    def _load_baseline_data(self, env_config: Dict) -> None:
        """
        Load baseline data from file and extract values for probe locations.

        Uses KD-tree for efficient nearest neighbor search to find baseline
        values at probe locations.

        Args:
            env_config: Environment configuration dictionary.

        Raises:
            ValueError: If baseline file is not found or has incompatible format.
        """
        baseline_file = self._find_baseline_file()

        if baseline_file is None:
            raise ValueError(
                "No baseline file found. Please provide 'baseline_file' in env_config "
                "or place a baseline file in the environment data directory"
            )

        if baseline_file.endswith(".feather"):
            print(f"Loading baseline data from Feather file: {baseline_file}", flush=True)
            baseline_df = pd.read_feather(baseline_file)
        elif baseline_file.endswith(".csv"):
            print(f"Loading baseline data from CSV file: {baseline_file}", flush=True)
            baseline_df = pd.read_csv(baseline_file)
        else:
            raise ValueError(f"Unsupported file format: {baseline_file}. Use .feather or .csv")

        coord_cols = [col for col in baseline_df.columns if "Points" in col or col in ["x", "y", "z"]]

        if len(coord_cols) == 0:
            if all(col in baseline_df.columns for col in ["Points:0", "Points:1", "Points:2"]):
                coord_cols = ["Points:0", "Points:1", "Points:2"]
            else:
                raise ValueError(
                    f"Could not find coordinate columns in baseline file. "
                    f"Available columns: {baseline_df.columns.tolist()}"
                )

        baseline_coords = baseline_df[coord_cols[: self.nDim]].values

        print(f"Building KD-Tree for {len(baseline_coords)} baseline points", flush=True)
        kdtree = cKDTree(baseline_coords)

        probe_coords = np.array(self.probe_locations).reshape(-1, self.nDim)
        print(f"Finding baseline values for {len(probe_coords)} probe locations", flush=True)

        distances, indices = kdtree.query(probe_coords)

        self.baseline_values = []

        if "forces" in self.observation_type:
            print("WARNING: Forces are in observation_type but not extracted from baseline file")
            self.baseline_values.extend([0.0] * self.nDim)

        probe_obs_types = ["u", "v", "w", "rho", "p"]

        for obs_var in probe_obs_types:
            if obs_var in self.observation_type:
                if obs_var not in baseline_df.columns:
                    raise ValueError(
                        f"Observation variable '{obs_var}' not found in baseline file. "
                        f"Available: {baseline_df.columns.tolist()}"
                    )
                var_values = baseline_df[obs_var].values[indices]
                self.baseline_values.extend(var_values)

        self.baseline_values = np.array(self.baseline_values)

        print(f"Baseline values extracted. Shape: {self.baseline_values.shape}", flush=True)
        print(
            f"Expected observation shape: {self.num_outputs if hasattr(self, 'num_outputs') else 'not yet configured'}",
            flush=True,
        )
        print(f"Max distance to nearest baseline point: {distances.max():.6f}", flush=True)

        if distances.max() > 0.1:
            print(
                f"WARNING: Some probes are far from baseline data points (max distance: {distances.max():.6f})",
                flush=True,
            )

    def _ensure_baseline_loaded(self) -> None:
        """
        Load baseline data if not already loaded (lazy loading).

        Called on first reward calculation when using baseline_mean strategy.

        Raises:
            ValueError: If baseline values don't match observation dimensions.
        """
        if self.reward_strategy == "baseline_mean" and not self._baseline_loaded:
            print("Loading baseline data for reward calculation...", flush=True)
            self._load_baseline_data(self._env_config)
            self._baseline_loaded = True

            if len(self.baseline_values) != self.num_outputs:
                raise ValueError(
                    f"Baseline values shape mismatch! "
                    f"Expected {self.num_outputs} values (matching num_outputs), "
                    f"but got {len(self.baseline_values)} from baseline file. "
                    f"Check that observation_type matches baseline file contents."
                )

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on the selected strategy.

        For 'running_mean': Penalizes deviation from exponential moving average.
        For 'baseline_mean': Penalizes deviation from pre-computed baseline.

        Returns:
            Tuple containing:
                - reward: Negative sum of absolute deviations
                - obj_dict: Empty dictionary (for compatibility)

        Raises:
            ValueError: If unknown reward strategy is specified.
        """
        obj_dict = {}

        if self.reward_strategy == "running_mean":
            if not hasattr(self, "running_mean"):
                self.running_mean = self.obs.copy()
                self.alpha = 0.025

            self.running_mean = self.alpha * self.obs + (1 - self.alpha) * self.running_mean
            print("running mean:", self.running_mean, flush=True)

            deviation = (self.obs - self.running_mean) / self.obs_scale
            reward = np.sum(np.abs(deviation))

        elif self.reward_strategy == "baseline_mean":
            self._ensure_baseline_loaded()

            deviation = (self.obs - self.baseline_values) / self.obs_scale
            reward = np.sum(np.abs(deviation))
            print("baseline deviation:", deviation, flush=True)

        else:
            raise ValueError(f"Unknown reward_strategy: {self.reward_strategy}. Use 'baseline_mean' or 'running_mean'")

        return -reward, obj_dict


class Cavity(CavityBase):
    """
    Single-jet cavity environment.

    This environment simulates cavity flow with a single jet actuator
    for flow control.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the cavity environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)
        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert RL action to CFD actuation format.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
        return action


class Cavity3Jet(CavityBase):
    """
    Three-jet cavity environment.

    This environment simulates cavity flow with three independent
    jet actuators for flow control.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the 3-jet cavity environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)
        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert RL action to CFD actuation format.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
        return action


# Register environment types with the factory
register_environment("Cavity", Cavity)
register_environment("Cavity3Jet", Cavity3Jet)
