"""
NACA 0012 Airfoil Environment
==============================

This module provides NACA 0012 airfoil flow CFD environments for
reinforcement learning, including configurations for steady flow
and gust response control.
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class NACA0012Base(MaiaFlowEnv):
    """
    Base class for NACA 0012 airfoil environments with Hugging Face integration.

    Provides common functionality for NACA airfoil configurations.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)


class NACA0012(NACA0012Base):
    """
    NACA 0012 airfoil environment optimizing lift-to-drag ratio.

    This environment controls flow around a NACA 0012 airfoil using
    synthetic jets. The reward maximizes L/D ratio.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 environment.

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

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on lift-to-drag ratio.

        Maximizes L/D ratio. Penalizes heavily for negative lift
        or near-zero drag.

        Returns:
            Tuple containing:
                - reward: L/D ratio (or -100 for invalid conditions)
                - obj_dict: Dictionary with force information
        """
        forces = self.maiaInterface.getForce(self.bcId[0])

        nondim_forces = self.compute_nondim_coefficients(
            forces=forces,
            density=1.0,
            referenceVelocity=self.Ma / np.sqrt(3),
            projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
        )

        lift = nondim_forces[1]
        drag = nondim_forces[0]

        # Penalize negative lift or near-zero drag
        if drag < 1e-6 or lift < 0:
            reward = -100.0
        else:
            reward = lift / drag

        obj_dict = {"forces": forces}

        return reward, obj_dict


class NACA0012Gust(NACA0012Base):
    """
    NACA 0012 airfoil environment for gust response control.

    This environment controls flow around a NACA 0012 airfoil
    during gust encounters. The reward minimizes deviation from
    unperturbed baseline forces.

    Gust: if the case's ``properties_run.toml`` sets ``lbGustFieldFile`` (m-AIA LB solver, 2026-11), m-AIA
    superposes the gust field of that file onto the (no-gust) restart state every time it loads the restart, i.e.
    at every ``reset()`` (MPMD re-initialisation). The default gust is the file shipped with the case: the
    superposed m-AIA slug (the streamwise surge of the released inflow-BC gust, fitted to it). Older cases without
    the property keep the released workflow (gust contained in the restart).

    Attributes:
        unperturbed_avg_drag: Baseline drag coefficient without gust.
        unperturbed_avg_lift: Baseline lift coefficient without gust.
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
        gust_field_file: Path of the case's gust file (None for cases without ``lbGustFieldFile``).
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 gust environment.

        Args:
            env_config: Environment configuration dictionary. Gust options (cases with ``lbGustFieldFile`` only):
                - gust_files (str or list of str): gust files (MAIAGUST format, e.g. written by
                  ``write_maia_gust.py``) used instead of the shipped default; before each reset the next one is
                  copied over the case's gust file (cyclic, or random with ``gust_sampling='random'``).
                - gust_sampling (str): 'cycle' (default) or 'random'.
                - gust_seed (int): seed of the random choice.
        """
        super().__init__(env_config)
        self.unperturbed_avg_drag = self.cfg.maia.unperturbed_avg_drag
        self.unperturbed_avg_lift = self.cfg.maia.unperturbed_avg_lift

        gf = self.runtime_property_file_data.get("lbGustFieldFile")
        self.gust_field_file: Optional[str] = os.path.join(self.env_data_path, gf) if gf else None
        files = env_config.get("gust_files")
        self.gust_files: List[str] = [files] if isinstance(files, str) else list(files or [])
        if self.gust_files and self.gust_field_file is None:
            raise ValueError(
                "gust_files given, but the case's properties_run.toml has no lbGustFieldFile "
                "(gust superposition needs a case with a no-gust restart and the m-AIA gust-field option)"
            )
        for f in self.gust_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"gust file not found: {f}")
        self.gust_sampling = env_config.get("gust_sampling", "cycle")
        if self.gust_sampling not in ("cycle", "random"):
            raise ValueError(f"gust_sampling must be 'cycle' or 'random', not {self.gust_sampling!r}")
        self._gust_rng = np.random.default_rng(env_config.get("gust_seed"))
        self._gust_episode = 0
        self.current_gust_file: Optional[str] = self.gust_field_file

        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment; with ``gust_files`` the episode's gust file is put in place first (m-AIA reads it
        when it reloads the restart). ``options={'gust_file': path}`` sets the gust of this episode explicitly.
        """
        path = (options or {}).get("gust_file")
        if path is None and self.gust_files:
            if self.gust_sampling == "random":
                path = self.gust_files[int(self._gust_rng.integers(len(self.gust_files)))]
            else:
                path = self.gust_files[self._gust_episode % len(self.gust_files)]
        if path is not None:
            if self.gust_field_file is None:
                raise ValueError("options['gust_file'] needs a case with lbGustFieldFile")
            if os.path.abspath(path) != os.path.abspath(self.gust_field_file):
                if self.gust_field_file and os.path.exists(self.gust_field_file) and not os.path.exists(
                    self.gust_field_file + ".default"
                ):
                    shutil.copyfile(self.gust_field_file, self.gust_field_file + ".default")
                shutil.copyfile(path, self.gust_field_file)
            self.current_gust_file = path
        self._gust_episode += 1
        return super().reset(seed=seed, options=options)

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert RL action to CFD actuation format.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
        return action

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on deviation from baseline forces.

        Minimizes deviation of lift and drag from unperturbed values.

        Returns:
            Tuple containing:
                - reward: Negative weighted deviation from baseline
                - obj_dict: Dictionary with force information
        """
        forces = self.maiaInterface.getForce(self.bcId[0])

        nondim_forces = self.compute_nondim_coefficients(
            forces=forces,
            density=1.0,
            referenceVelocity=self.Ma / np.sqrt(3),
            projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
        )

        print("nonDim forces:", nondim_forces, flush=True)

        obj_dict = {"forces": forces}

        reward = -np.abs(nondim_forces[1] - self.unperturbed_avg_lift) - self.omega * np.abs(
            nondim_forces[0] - self.unperturbed_avg_drag
        )

        return reward, obj_dict


# Register environment types with the factory
register_environment("NACA0012", NACA0012)
register_environment("NACA0012Gust", NACA0012Gust)
