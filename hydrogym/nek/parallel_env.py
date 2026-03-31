"""
Dict-based multi-agent wrapper for NekEnv.
Converts between array-based and dict-based interfaces.
"""

from typing import Dict, Tuple

import numpy as np

from .env import NekEnv


class NekParallelEnv:
    """
    Multi-agent wrapper for NekEnv with dict-based interface.

    This wrapper treats each actuator as a separate agent with its own
    observation and action. Useful for multi-agent RL experiments or when
    you need per-actuator control.

    Args:
      nek_env: The base NekEnv instance to wrap
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, nek_env: NekEnv):
        self.env = nek_env

        # Create agent names based on actuator info
        self.possible_agents = [
            self._name_agent(
                nid=self.env.actuator_info["NID"][i],
                gllid=self.env.actuator_info["GLLID"][i],
                iface=self.env.actuator_info["FACEID"][i],
                ix=self.env.actuator_info["ix"][i],
                iy=self.env.actuator_info["iy"][i],
                iz=self.env.actuator_info["iz"][i],
            )
            for i in range(self.env.n_actuators)
        ]
        self.agents = self.possible_agents[:]

        # Per-agent observation and action sizes
        self.obs_per_agent = self.env.obs_per_actuator
        self.act_per_agent = 1  # Each actuator has scalar action

    @staticmethod
    def _name_agent(nid, gllid, iface, ix, iy, iz):
        """Create agent name from grid information."""
        return f"jet_np{nid:08d}_gid{gllid:08d}_iface{iface}_ix{ix:08d}_iy{iy:08d}_iz{iz:08d}"

    def observation_space(self, agent):
        """Get observation space for a specific agent."""
        import gymnasium as gym

        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_per_agent,), dtype=np.float32)

    def action_space(self, agent):
        """Get action space for a specific agent."""
        import gymnasium as gym

        idx = self.possible_agents.index(agent)
        return gym.spaces.Box(
            low=self.env.action_space.low[idx], high=self.env.action_space.high[idx], shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment and return dict of observations."""
        obs_array, info = self.env.reset(seed=seed, options=options)
        obs_dict = self._split_observations(obs_array)
        return obs_dict, info

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """
        Step environment with dict of actions.

        Args:
          actions: Dict mapping agent names to actions

        Returns:
          observations: Dict of observations per agent
          rewards: Dict of rewards per agent
          terminated: Dict of terminated flags per agent
          truncated: Dict of truncated flags per agent
          infos: Dict of info dicts per agent
        """
        # Convert dict to array
        action_array = self._concat_actions(actions)

        # Step base environment
        obs_array, reward, terminated, truncated, info = self.env.step(action_array)

        # Convert results to dicts
        obs_dict = self._split_observations(obs_array)

        # Get per-agent rewards if available
        if "reward_per_actuator" in info:
            rewards_dict = {agent: float(info["reward_per_actuator"][i]) for i, agent in enumerate(self.agents)}
        else:
            # Uniform reward for all agents
            rewards_dict = {agent: reward for agent in self.agents}

        # Terminated/truncated same for all agents
        terminated_dict = {agent: terminated for agent in self.agents}
        truncated_dict = {agent: truncated for agent in self.agents}

        # Info per agent
        infos_dict = {agent: {"time": info.get("time", 0.0)} for agent in self.agents}

        return obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict

    def _concat_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert dict of actions to flat array."""
        action_array = np.zeros(self.env.n_actuators, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            if agent in actions:
                action_array[i] = float(np.asarray(actions[agent]).reshape(-1)[0])
            else:
                raise ValueError(f"Missing action for agent {agent}")
        return action_array

    def _split_observations(self, obs_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat observation array to dict."""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            start = i * self.obs_per_agent
            end = (i + 1) * self.obs_per_agent
            obs_dict[agent] = obs_array[start:end].astype(np.float32)
        return obs_dict

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment."""
        self.env.close()
