"""
Cube Flow Environment
=====================

This module provides 3D cube flow CFD environments for reinforcement learning.
"""

from typing import Dict, List, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class CubeBase(MaiaFlowEnv):
    """
    Base class for cube flow environments with Hugging Face integration.

    This class provides common functionality for 3D cube-based CFD environments,
    including reward computation based on drag and lift/side forces.

    The reward is computed as:
        reward = -|C_D| - omega * |C_L| - omega * |C_S|

    where C_D is the drag coefficient, C_L is the lift coefficient,
    C_S is the side force coefficient, and omega is a weighting factor.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the cube base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on aerodynamic force coefficients.

        Calculates non-dimensional force coefficients and returns a reward
        that penalizes drag, lift, and side forces.

        Returns:
            Tuple containing:
                - reward: Scalar reward (or list for multiple boundaries)
                - obj_dict: Dictionary with force information
        """
        rewards = []
        forces_list = []

        for bc_id in self.bcId:
            forces = self.maiaInterface.getForce(bc_id)
            nondim_coefficients = self.compute_nondim_coefficients(
                forces=forces,
                density=1.0,
                referenceVelocity=self.Ma / np.sqrt(3),
                projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
            )

            reward = (
                -np.abs(nondim_coefficients[0]).sum()
                - self.omega * np.abs(nondim_coefficients[1]).sum()
                - self.omega * np.abs(nondim_coefficients[2]).sum()
            )
            rewards.append(reward)
            forces_list.append(forces)

        obj_dict = {"forces": forces_list}

        return (rewards[0] if len(self.bcId) == 1 else rewards), obj_dict

    def drag_coefficient_from_objective(self, obj_dict):
        """Total drag coefficient over the boundary segments (normalised like the reward's force coefficients)."""
        return float(
            sum(
                self.compute_nondim_coefficients(
                    forces=np.asarray(f),
                    density=1.0,
                    referenceVelocity=self.Ma / np.sqrt(3),
                    projectionLength=self.power_projection(),
                )[0]
                for f in obj_dict["forces"]
            )
        )


class Cube(CubeBase):
    """
    3D cube environment with flow control.

    This environment simulates flow around a 3D cube geometry.

    Actuation power (``hydrogym.maia.wall_jets``, one actuator per jet): the slot jets of BC 2007 object class 2
    (parabolic profile across ``lbJetSlotAxis`` within ``lbJetLimits``, direction ``lbJetNormals``) are the fluid
    cells next to the cube faces (faces on cell faces: exactly the solver's jet cells; along the slot every k-th cell
    with k-fold weight, at most 64), ``C_P = 2 P / (U^3 D zLength)`` like ``C_D``. The jets are independent (not
    zero-net-mass-flux); ``p_ref`` is the mean wall pressure over the jets. The jet geometry is checked against the
    BC-2007 rule in the unit tests.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the cube environment.

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
register_environment("Cube", Cube)
