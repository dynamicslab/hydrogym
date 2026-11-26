"""
Sphere Flow Environment
=======================

This module provides 3D sphere flow CFD environments for reinforcement learning.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class SphereBase(MaiaFlowEnv):
    """
    Base class for sphere flow environments with Hugging Face integration.

    This class provides common functionality for 3D sphere-based CFD environments,
    including reward computation based on drag and lift/side forces.

    The projection area for force coefficient calculation uses the
    sphere's circular cross-section: A = pi * D^2 / 4

    The reward is computed as:
        reward = -|C_D| - omega * |C_L| - omega * |C_S|
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the sphere base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on aerodynamic force coefficients.

        Uses circular cross-section area for coefficient calculation.

        Returns:
            Tuple containing:
                - reward: Scalar reward (or list for multiple boundaries)
                - obj_dict: Dictionary with force information
        """
        rewards = []
        forces_list = []

        for bc_id in self.bcId:
            forces = self.maiaInterface.getForce(bc_id)

            # Circular cross-section: pi * D^2 / 4
            projection_area = (math.pi * self.referenceLength / self.dX * self.referenceLength / self.dX) / 4

            nondim_coefficients = self.compute_nondim_coefficients(
                forces=forces, density=1.0, referenceVelocity=self.Ma / np.sqrt(3), projectionLength=projection_area
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

    def power_projection(self) -> float:
        """Projected area of the sphere in cells^2 (pi D^2 / 4), the normalisation of its force coefficients."""
        return math.pi * (self.referenceLength / self.dX) ** 2 / 4

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


class Sphere(SphereBase):
    """
    3D sphere environment with flow control.

    This environment simulates flow around a 3D sphere geometry.

    Actuation power (``hydrogym.maia.wall_jets``, one actuator per jet): the ring jets of BC 2007 object class 3
    (parabolic profile in arc length along ``lbJetSlotAxis``, azimuthal range ``lbJetRanges``, radial direction)
    at the level of the lattice links on the analytic sphere; ``C_P = 2 P / (U^3 pi D^2 / 4)`` like ``C_D``; wall
    pressure from 2-5 cells out (the 3D stencils of 1.5 cells reach into the sphere). The jets are independent;
    ``p_ref`` is the mean wall pressure over the jets.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the sphere environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)
        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()
        self.setup_actuation_power("wall_jets", geometry="sphere")

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
register_environment("Sphere", Sphere)
