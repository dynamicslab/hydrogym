"""
NACA 0012 Airfoil Environment
==============================

This module provides NACA 0012 airfoil flow CFD environments for
reinforcement learning, including configurations for steady flow
and gust response control.
"""

from typing import Dict, List, Tuple

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

    Attributes:
        unperturbed_avg_drag: Baseline drag coefficient without gust.
        unperturbed_avg_lift: Baseline lift coefficient without gust.
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 gust environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)
        self.unperturbed_avg_drag = self.cfg.maia.unperturbed_avg_drag
        self.unperturbed_avg_lift = self.cfg.maia.unperturbed_avg_lift

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
