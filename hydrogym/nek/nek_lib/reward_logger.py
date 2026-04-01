"""
Real-time Reward Logger for NEK5000 DRL Environment
@yuningw
"""

import os
import csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union


class RewardLogger:
    """
    A real-time reward logger that writes rewards to CSV files for monitoring.
    Supports both per-agent and aggregated reward logging.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        log_name: str = "rewards",
        log_per_agent: bool = True,
        log_aggregated: bool = True,
        flush_frequency: int = 10,
    ):
        """
        Initialize the reward logger.

        Args:
            log_dir: Directory to save log files
            log_name: Base name for log files
            log_per_agent: Whether to log individual agent rewards
            log_aggregated: Whether to log aggregated rewards (mean, std, etc.)
            flush_frequency: How often to flush data to disk (every N steps)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_name = log_name
        self.log_per_agent = log_per_agent
        self.log_aggregated = log_aggregated
        self.flush_frequency = flush_frequency

        # Initialize log files
        self._init_log_files()

        # Data buffers
        self.step_count = 0
        self.episode_count = 0
        self.reward_buffer = []
        self.timestamp_buffer = []

        # Agent tracking
        self.agent_names = []
        self.agent_rewards = {}

    def _init_log_files(self):
        """Initialize CSV log files with headers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.log_per_agent:
            self.per_agent_file = self.log_dir / f"{self.log_name}_per_agent_{timestamp}.csv"
            self.per_agent_writer = None
            self.per_agent_file_handle = None
        else:
            self.per_agent_file = None
            self.per_agent_writer = None
            self.per_agent_file_handle = None

        if self.log_aggregated:
            self.aggregated_file = self.log_dir / f"{self.log_name}_aggregated_{timestamp}.csv"
            self.aggregated_writer = None
            self.aggregated_file_handle = None
        else:
            self.aggregated_file = None
            self.aggregated_writer = None
            self.aggregated_file_handle = None

        # Episode summary file
        self.episode_file = self.log_dir / f"{self.log_name}_episodes_{timestamp}.csv"
        self.episode_writer = None
        self.episode_file_handle = None

    def _open_files(self):
        """Open log files for writing."""
        if self.log_per_agent and self.per_agent_writer is None:
            self.per_agent_file_handle = open(self.per_agent_file, "w", newline="")
            self.per_agent_writer = csv.writer(self.per_agent_file_handle)
            # Write header
            header = ["timestamp", "episode", "step", "agent_name", "reward", "dUdy_raw"]
            self.per_agent_writer.writerow(header)

        if self.log_aggregated and self.aggregated_writer is None:
            self.aggregated_file_handle = open(self.aggregated_file, "w", newline="")
            self.aggregated_writer = csv.writer(self.aggregated_file_handle)
            # Write header
            header = [
                "timestamp",
                "episode",
                "step",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
                "total_reward",
                "num_agents",
            ]
            self.aggregated_writer.writerow(header)

        if self.episode_writer is None:
            self.episode_file_handle = open(self.episode_file, "w", newline="")
            self.episode_writer = csv.writer(self.episode_file_handle)
            # Write header
            header = [
                "timestamp",
                "episode",
                "total_steps",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
                "total_reward",
                "duration_seconds",
            ]
            self.episode_writer.writerow(header)

    def log_rewards(
        self,
        rewards: Dict[str, float],
        dUdy_raw: Optional[Dict[str, float]] = None,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Log rewards for all agents.

        Args:
            rewards: Dictionary mapping agent names to reward values
            dUdy_raw: Dictionary mapping agent names to raw dUdy values (optional)
            episode: Episode number (optional)
            step: Step number within episode (optional)
        """
        self._open_files()

        current_time = datetime.now().isoformat()
        if episode is None:
            episode = self.episode_count
        if step is None:
            step = self.step_count

        # Update agent names if first time
        if not self.agent_names:
            self.agent_names = list(rewards.keys())
            self.agent_rewards = {name: [] for name in self.agent_names}

        # Log per-agent rewards
        if self.log_per_agent:
            for agent_name, reward in rewards.items():
                dUdy_value = dUdy_raw.get(agent_name, 0.0) if dUdy_raw else 0.0
                row = [current_time, episode, step, agent_name, reward, dUdy_value]
                self.per_agent_writer.writerow(row)
                self.agent_rewards[agent_name].append(reward)

        # Calculate aggregated statistics
        reward_values = list(rewards.values())
        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values)
        min_reward = np.min(reward_values)
        max_reward = np.max(reward_values)
        total_reward = np.sum(reward_values)
        num_agents = len(rewards)

        # Log aggregated rewards
        if self.log_aggregated:
            row = [
                current_time,
                episode,
                step,
                mean_reward,
                std_reward,
                min_reward,
                max_reward,
                total_reward,
                num_agents,
            ]
            self.aggregated_writer.writerow(row)

        # Store in buffer for episode summary
        self.reward_buffer.extend(reward_values)
        self.timestamp_buffer.append(current_time)

        # Flush periodically
        if self.step_count % self.flush_frequency == 0:
            self.flush()

        self.step_count += 1

    def log_episode_summary(self, episode: Optional[int] = None):
        """
        Log summary statistics for the current episode.

        Args:
            episode: Episode number (optional)
        """
        if not self.reward_buffer:
            return

        self._open_files()

        current_time = datetime.now().isoformat()
        if episode is None:
            episode = self.episode_count

        # Calculate episode statistics
        mean_reward = np.mean(self.reward_buffer)
        std_reward = np.std(self.reward_buffer)
        min_reward = np.min(self.reward_buffer)
        max_reward = np.max(self.reward_buffer)
        total_reward = np.sum(self.reward_buffer)
        total_steps = len(self.reward_buffer)

        # Calculate duration (approximate)
        if len(self.timestamp_buffer) >= 2:
            start_time = datetime.fromisoformat(self.timestamp_buffer[0])
            end_time = datetime.fromisoformat(self.timestamp_buffer[-1])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0.0

        # Log episode summary
        row = [
            current_time,
            episode,
            total_steps,
            mean_reward,
            std_reward,
            min_reward,
            max_reward,
            total_reward,
            duration,
        ]
        self.episode_writer.writerow(row)

        # Reset buffers
        self.reward_buffer = []
        self.timestamp_buffer = []
        self.agent_rewards = {name: [] for name in self.agent_names}

        self.episode_count += 1
        self.step_count = 0

        # Flush after episode
        self.flush()

    def flush(self):
        """Flush all open files."""
        if self.per_agent_file_handle:
            self.per_agent_file_handle.flush()
        if self.aggregated_file_handle:
            self.aggregated_file_handle.flush()
        if self.episode_file_handle:
            self.episode_file_handle.flush()

    def close(self):
        """Close all open files."""
        if self.per_agent_file_handle:
            self.per_agent_file_handle.close()
        if self.aggregated_file_handle:
            self.aggregated_file_handle.close()
        if self.episode_file_handle:
            self.episode_file_handle.close()

    def get_latest_stats(self) -> Dict[str, float]:
        """Get the latest reward statistics."""
        if not self.reward_buffer:
            return {}

        return {
            "mean_reward": np.mean(self.reward_buffer),
            "std_reward": np.std(self.reward_buffer),
            "min_reward": np.min(self.reward_buffer),
            "max_reward": np.max(self.reward_buffer),
            "total_reward": np.sum(self.reward_buffer),
            "num_rewards": len(self.reward_buffer),
        }

    def get_agent_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each agent."""
        stats = {}
        for agent_name, rewards in self.agent_rewards.items():
            if rewards:
                stats[agent_name] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "min_reward": np.min(rewards),
                    "max_reward": np.max(rewards),
                    "total_reward": np.sum(rewards),
                    "num_rewards": len(rewards),
                }
        return stats


class SimpleRewardLogger:
    """
    A simplified reward logger for basic CSV logging.
    """

    def __init__(self, log_file: Union[str, Path]):
        """
        Initialize simple reward logger.

        Args:
            log_file: Path to the CSV log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open file and write header
        self.file_handle = open(self.log_file, "w", newline="")
        self.writer = csv.writer(self.file_handle)
        header = ["timestamp", "episode", "step", "agent_name", "reward", "dUdy_raw"]
        self.writer.writerow(header)

        self.step_count = 0
        self.episode_count = 0

    def log_reward(
        self,
        agent_name: str,
        reward: float,
        dUdy_raw: float = 0.0,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Log a single reward.

        Args:
            agent_name: Name of the agent
            reward: Reward value
            dUdy_raw: Raw dUdy value
            episode: Episode number
            step: Step number
        """
        current_time = datetime.now().isoformat()
        if episode is None:
            episode = self.episode_count
        if step is None:
            step = self.step_count

        row = [current_time, episode, step, agent_name, reward, dUdy_raw]
        self.writer.writerow(row)

        self.step_count += 1

    def new_episode(self):
        """Mark the start of a new episode."""
        self.episode_count += 1
        self.step_count = 0

    def flush(self):
        """Flush the file."""
        self.file_handle.flush()

    def close(self):
        """Close the file."""
        self.file_handle.close()
