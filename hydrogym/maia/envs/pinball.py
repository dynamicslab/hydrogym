"""
Pinball Flow Environment
========================

This module provides pinball (multi-cylinder) flow CFD environments
for reinforcement learning.
"""

from typing import Dict, List, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


class PinballBase(MaiaFlowEnv):
  """
    Base class for pinball flow environments with Hugging Face integration.

    The pinball configuration consists of multiple cylinders arranged in a
    triangular pattern. Forces from all cylinders are summed for the reward.

    The reward is computed as:
        reward = -|sum(C_D)| - omega * |sum(C_L)|

    where C_D and C_L are summed across all cylinders.
    """

  def __init__(self, env_config: Dict):
    """
        Initialize the pinball base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
    super().__init__(env_config)

  def get_reward(self) -> Tuple[float, Dict]:
    """
        Compute the reward based on total aerodynamic force coefficients.

        Aggregates forces from all cylinders in the pinball configuration.

        Returns:
            Tuple containing:
                - reward: Scalar reward based on total drag and lift
                - obj_dict: Dictionary with force coefficient information
        """
    forces = []
    for bc_id in self.bcId:
      forces.append(self.maiaInterface.getForce(bc_id))

    forces = np.stack(forces)

        nondim_coefficients = self.compute_nondim_coefficients(
            forces=forces,
            density=1.0,
            referenceVelocity=self.Ma / np.sqrt(3),
            projectionLength=self.referenceLength / self.dX * self.zLength / self.dX
        )

        obj_dict = {'forces': nondim_coefficients}

    reward = (-np.abs(nondim_coefficients[:, 0].sum()) -
              self.omega * np.abs(nondim_coefficients[:, 1].sum()))

    return reward, obj_dict


class Pinball(PinballBase):
  """
  Rotary pinball environment with rotational actuation.

  This environment controls flow using cylinder rotation.
  Each cylinder can rotate independently.
  """

  def __init__(self, env_config: Dict):
    """
    Initialize the pinball environment.

    Args:
      env_config: Environment configuration dictionary.
    """
    super().__init__(env_config)

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


class JetPinball(PinballBase):
  """
  Jet-actuated pinball environment.

  This environment controls flow using synthetic jets on the cylinders.
  Jets are paired for zero net mass flux actuation.

  Attributes:
    numJetsInSimulation: Number of jet actuators in the CFD simulation.
  """

  def __init__(self, env_config: Dict):
    """
    Initialize the jet pinball environment.

    Args:
      env_config: Environment configuration dictionary.
    """
    super().__init__(env_config)
    self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

    self.configure_observations()
    self.configure_probe_dimensions()
    self.set_observation_action_spaces()

    self.setup_normalization()

  def convert_action(self, action: np.ndarray) -> List[float]:
    """
    Convert RL action to CFD actuation sequence.

    Implements zero net mass flux by pairing jets with opposite signs.

    Args:
      action: Action array from the RL agent.

    Returns:
      Actuation sequence for the CFD solver.
    """
    maia_action_sequence = []

    for jet_pair in range(int(self.numJetsInSimulation / 2)):
      maia_action_sequence.extend([action[jet_pair], -action[jet_pair]])

    return maia_action_sequence


# Register environment types with the factory
register_environment("Pinball", Pinball)
register_environment("JetPinball", JetPinball)
