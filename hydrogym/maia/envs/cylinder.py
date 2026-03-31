"""
Cylinder Flow Environment
=========================

This module provides cylinder flow CFD environments for reinforcement learning,
including standard cylinder and rotary cylinder configurations.
"""

from typing import Dict, List, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class CylinderBase(MaiaFlowEnv):
  """
    Base class for cylinder flow environments with Hugging Face integration.

    This class provides the common functionality for cylinder-based CFD
    environments, including reward computation based on drag and lift forces.

    The reward is computed as:
        reward = -|C_D| - omega * |C_L|

    where C_D is the drag coefficient, C_L is the lift coefficient,
    and omega is a weighting factor.
    """

  def __init__(self, env_config: Dict):
    """
        Initialize the cylinder base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
    super().__init__(env_config)

  def get_reward(self) -> Tuple[float, Dict]:
    """
        Compute the reward based on aerodynamic force coefficients.

        Calculates non-dimensional force coefficients and returns a reward
        that penalizes drag and lift forces.

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
          projectionLength=self.referenceLength / self.dX * self.zLength /
          self.dX)

      reward = (-np.abs(nondim_coefficients[0]).sum() -
                self.omega * np.abs(nondim_coefficients[1]).sum())
      rewards.append(reward)
      forces_list.append(forces)

    obj_dict = {'forces': forces_list}

    return (rewards[0] if len(self.bcId) == 1 else rewards), obj_dict


class Cylinder(CylinderBase):
  """
    Standard cylinder environment with jet actuation.

    This environment simulates flow around a circular cylinder with
    synthetic jet actuators for flow control. Jets are configured
    in pairs for zero net mass flux actuation.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

  def __init__(self, env_config: Dict):
    """
        Initialize the cylinder environment.

        Args:
            env_config: Environment configuration dictionary.
        """
    super().__init__(env_config)
    self.numJetsInSimulation = self._get_property(
        self.runtime_property_file_data, "lbNoJets")

    # Configure observation and action space
    self.configure_observations()
    self.configure_probe_dimensions()
    self.set_observation_action_spaces()

    # Handle normalization factors
    self.setup_normalization()

  def convert_action(self, action: np.ndarray) -> List[float]:
    """
        Convert RL action to CFD actuation sequence.

        Implements zero net mass flux by pairing jets with opposite signs.
        For each jet pair, the first jet uses +action and the second uses -action.

        Args:
            action: Action array from the RL agent.

        Returns:
            Actuation sequence for the CFD solver.
        """
    maia_action_sequence = []

    for jet_pair in range(int(self.numJetsInSimulation / 2)):
      maia_action_sequence.extend([action[jet_pair], -action[jet_pair]])

    return maia_action_sequence


class RotaryCylinder(CylinderBase):
  """
    Rotary cylinder environment with rotational actuation.

    This environment simulates flow around a circular cylinder that can
    rotate. The actuation controls the cylinder's angular velocity.
    """

  def __init__(self, env_config: Dict):
    """
        Initialize the rotary cylinder environment.

        Args:
            env_config: Environment configuration dictionary.
        """
    super().__init__(env_config)

    # Configure observation and action space
    self.configure_observations()
    self.configure_probe_dimensions()
    self.set_observation_action_spaces()

    # Handle normalization factors
    self.setup_normalization()

  def convert_action(self, action: np.ndarray) -> np.ndarray:
    """
        Convert RL action to CFD actuation format.

        For rotary cylinder, the action directly controls angular velocity.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
    return action


# Register environment types with the factory
register_environment("Cylinder", Cylinder)
register_environment("RotaryCylinder", RotaryCylinder)
