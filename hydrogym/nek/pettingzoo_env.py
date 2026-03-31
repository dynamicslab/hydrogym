"""
Optional PettingZoo-compatible wrapper for NekParallelEnv.
Only needed if you want to use PettingZoo-specific libraries.
"""

try:
    from pettingzoo import ParallelEnv

    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    ParallelEnv = object  # Fallback base class

import functools

from .parallel_env import NekParallelEnv


class NekPettingZooEnv(ParallelEnv):
    """
    PettingZoo-compatible wrapper for NekParallelEnv.

    This wrapper makes the environment compatible with PettingZoo's API,
    allowing use with PettingZoo-specific libraries and tools.

    Args:
      parallel_env: NekParallelEnv instance to wrap
    """

    metadata = {"render_modes": ["human"], "name": "nek_v1"}

    def __init__(self, parallel_env: NekParallelEnv, render_mode=None):
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo is not installed. Install it with: pip install pettingzoo")

        self.env = parallel_env
        self.render_mode = render_mode
        self.possible_agents = self.env.possible_agents
        self.agents = self.env.agents

    @property
    def unwrapped(self):
        """Return the base environment without wrappers."""
        return self

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for agent (cached)."""
        return self.env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for agent (cached)."""
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        # PettingZoo ParallelEnv.reset() can return just obs or (obs, info)
        # We return both for compatibility
        return obs_dict, info

    def step(self, actions):
        """Step the environment."""
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = self.env.step(actions)

        # PettingZoo uses "terminations" and "truncations" instead of "terminated" and "truncated"
        # But ParallelEnv API matches Gymnasium, so we can return as-is
        return obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment."""
        self.env.close()


def make_pettingzoo_env(nek_env, render_mode=None):
    """
    Convenience function to create PettingZoo environment from NekEnv.

    Args:
      nek_env: Base NekEnv instance
      render_mode: Render mode for the environment (default: None)

    Returns:
      NekPettingZooEnv instance
    """
    parallel_env = NekParallelEnv(nek_env)
    return NekPettingZooEnv(parallel_env, render_mode=render_mode)
