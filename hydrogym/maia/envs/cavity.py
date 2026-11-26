"""
Cavity Flow Environment
========================

This module provides cavity flow CFD environments for reinforcement learning,
designed for flow control applications in cavity configurations.

Hugging Face cases ``Cavity_2D_Re{4200,7500}`` (one jet) and ``Cavity3Jet_2D_Re{4200,7500}`` (three jets), on the
literature configuration of the open cavity (Barbagallo et al. 2009 / Sipp & Lebedev 2007), validated against the
finite-volume reference:

* domain x in [-1, 4.7] x [-1, 0.5] in case coordinates (= literature coordinates + 1.2; cavity [1.2, 2.2] x [-1, 0]),
  1 D sponge layers outside the literature domain, slip top, slip floor upstream of x = 0.8 and downstream of x = 2.95,
  velocity inlet, non-reflecting (plane-wave characteristic) outlet;
* D2Q9 cumulant collision, Ma 0.0125 (U_inf = 0.0072169 lattice units), uniform grid D/240;
* ``Re4200`` is run at solver Re 4070, the same distance above this grid's critical Reynolds number as Re 4200 is
  above the reference Re_c = 4140 (weakly unstable); ``Re7500`` is the limit-cycle case;
* jets (equilibrium jet boundary condition, parabolic profile over 0.02): leading edge x = 1.175 on y = 0
  (wall-normal), and for ``Cavity3Jet`` also the trailing cavity wall (x = 2.2, y = -0.25) and the leading cavity
  wall (x = 1.2, y = -0.5); actions in [-1, 1] times ``max_control`` = centre velocity 0.35 U (one jet) /
  0.69 U (three jets); one action = 633 LB steps = 0.019 D/U;
* restart = a developed uncontrolled state; ``baseline_state.feather`` = the uncontrolled time mean (40 D/U) at
  every cell centre.

Actuation power (``info['actuation_power']`` etc., ``hydrogym.maia.wall_jets``): one actuator per jet,
``C_P = 2 P / (U^3 D)`` per unit span (``D`` = cavity depth, ``referenceLength``; there is no drag coefficient).
The jet cells are the fluid cells next to the slot (exactly the cells of BC 2007); the jets are not
zero-net-mass-flux, ``p_ref`` is the mean pressure over the jets (one group), so the signed power is gauge-free but
leaves out the work of supplying the net flow. In this configuration (walls on cell faces, equilibrium jet BC) the
mass flux of the model agrees with a control-volume balance of the solver's flow within 2 %. The slot is 4.8 cells
wide, so the exit pressure is not resolved: the signed power (a small residual for a single jet, about 4e-4 of it
kinetic) changes by a factor 2-3 with the wall stencil; only its order of magnitude is meaningful. Cases whose walls
are off the cell faces are modelled link by link.

Timing of the MPMD loop: ``step(a_k)`` sets ``a_k``, advances one control interval with it and returns the probe state
at the END of that interval (gym convention; ``hydrogym.maia.env_core.MaiaFlowEnv.step``). ``reset()`` returns the
state of the restart after the reinitialisation.

3D cases ``Cavity_3D_Re{4200,7500}_lz024`` (one jet) and ``Cavity3Jet_3D_Re{4200,7500}_lz024`` (three jets)
-------------------------------------------------------------------------------------------------------------------
The same classes load the 3D cavities (``nDim = 3`` in the property file):

* The ``_lz024`` cases are the literature cavity above extruded over a spanwise-periodic width L_z = 0.24 D
  (48 planes), uniform grid D/200 (7.39 M cells), D3Q27 cumulant collision, Ma 0.0125, the plane-wave
  characteristic outlet, the same sponge, jets and Reynolds numbers as 2D (``Re4200`` at solver Re 4070). They are
  the environments of the 3D cavity policies in ``policies/``.
* **Literature coordinates**, unlike the 2D cases: cavity [0, 1] x [-1, 0], z in [-0.12, 0.12] (the 2D cases use
  literature + 1.2 in x). Jets: leading edge x = -0.025 on y = 0; ``Cavity3Jet`` also trailing wall (x = 1,
  y = -0.25) and leading wall (x = 0, y = -0.5); the slots span the full width. ``max_control`` as 2D (0.35 U /
  0.69 U); one action = 527 LB steps = 0.019 D/U.
* **Probes are passed by the user** (``probe_locations`` is required, flattened ``[x0, y0, z0, x1, ...]``).
  ``probes_8planes.txt`` in each HF folder holds the 1760 cells (8 spanwise planes x 220 cells) that the released 3D
  policies observe; ``observation_weights.json`` gives their observation operator (50 shear-layer probes and the
  sensor per plane, averaged over the planes).  ``baseline_state.feather`` (uncontrolled time mean over t = 20-60 D/U,
  spanwise-averaged) is given at all cells of the 48 planes; the env takes the nearest
  point.
* **Launch:** MPMD, 1 Python rank + N m-AIA ranks (4 GPUs: about 2.5 s per action). With ``srun --multi-prog`` set
  ``MAIA_MPMD_APPNUM`` (0 for Python, 1 for m-AIA); ``mpirun -np 1 python ... : -np 4 maia properties_run.toml``
  sets ``MPI_APPNUM`` itself.  Needs an m-AIA version with the D3Q27 characteristic outflow (bc 4046); without it
  bc 4046 silently falls back to its LODI default, i.e. a different case.
* **Episode length:** ``max_episode_steps`` is 200 (3.8 D/U), as in 2D. The controlled 3D flow needs far longer: a
  typical evaluation is a roll-out of 60 D/U = 3156 actions with the fluctuation averaged over t = 40-60 D/U (the
  transient lasts 16-40 D/U depending on the start). Pass
  ``max_episode_steps=3156`` (or larger), or chain episodes without ``reset()``.
* **Start state:** ``restart_.Netcdf`` is a developed uncontrolled m-AIA state (t = 160 D/U).
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

            deviation = (self.obs - self.running_mean) / self.obs_scale
            reward = np.sum(np.abs(deviation))

        elif self.reward_strategy == "baseline_mean":
            self._ensure_baseline_loaded()

            deviation = (self.obs - self.baseline_values) / self.obs_scale
            reward = np.sum(np.abs(deviation))

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
        self.setup_actuation_power("wall_jets", geometry="flat")

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
        self.setup_actuation_power("wall_jets", geometry="flat")

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
