#!/usr/bin/env python3
"""
Train SB3 agent on parallel_env using DIY centralized wrapper.

parallel_env returns dict-based obs/actions, which is not directly compatible with SB3.
This example shows how to create a simple centralized wrapper that concatenates
all agents' observations and actions into single arrays for SB3 training.

Educational approach: Shows how the conversion works under the hood.
For production, see chapter 3 (PettingZoo + SuperSuit).
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import gymnasium as gym

from hydrogym.nek import NekEnv, NekParallelEnv


class CentralizedParallelWrapper(gym.Env):
    """
    DIY wrapper that converts dict-based parallel_env to SB3-compatible array-based interface.

    This wrapper concatenates all agents' observations and actions, allowing a single
    centralized policy to control all agents simultaneously.

    Educational purpose: Shows how multi-agent → single-agent conversion works.
    """

    def __init__(self, parallel_env: NekParallelEnv, reward_agg="mean"):
        self.env = parallel_env
        self.reward_agg = reward_agg
        self.n_agents = len(self.env.agents)

        # Build concatenated observation and action spaces
        # Get dimensions from first agent
        sample_agent = self.env.agents[0]
        single_obs_space = self.env.observation_space(sample_agent)
        single_act_space = self.env.action_space(sample_agent)

        total_obs_dim = single_obs_space.shape[0] * self.n_agents
        total_act_dim = single_act_space.shape[0] * self.n_agents

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.tile(single_act_space.low, self.n_agents),
            high=np.tile(single_act_space.high, self.n_agents),
            shape=(total_act_dim,),
            dtype=np.float32,
        )

        # Call parent constructor
        super().__init__()

    def reset(self, seed=None, options=None):
        """Reset and concatenate observations."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        obs_array = self._concat_obs(obs_dict)
        return obs_array, info

    def step(self, action_array):
        """Split actions, step env, concatenate observations, aggregate rewards."""
        # Convert array → dict
        actions_dict = self._split_actions(action_array)

        # Step the dict-based environment
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = self.env.step(actions_dict)

        # Convert dict → array
        obs_array = self._concat_obs(obs_dict)

        # Aggregate rewards across agents
        rewards_list = list(rewards_dict.values())
        if self.reward_agg == "mean":
            reward = np.mean(rewards_list)
        elif self.reward_agg == "sum":
            reward = np.sum(rewards_list)
        else:
            reward = rewards_list[0]

        # All agents share termination status
        terminated = all(terminated_dict.values())
        truncated = all(truncated_dict.values())

        return obs_array, reward, terminated, truncated, infos_dict

    def _concat_obs(self, obs_dict):
        """Concatenate dict of observations into single array."""
        return np.concatenate([obs_dict[agent] for agent in self.env.agents], dtype=np.float32)

    def _split_actions(self, action_array):
        """Split action array into dict for each agent."""
        actions_dict = {}
        act_per_agent = self.env.act_per_agent
        for i, agent in enumerate(self.env.agents):
            start = i * act_per_agent
            end = (i + 1) * act_per_agent
            actions_dict[agent] = action_array[start:end]
        return actions_dict

    def close(self):
        """Close underlying environment."""
        self.env.close()


def train_parallel_centralized(args):
    """Train using DIY centralized wrapper."""
    print("=" * 70)
    print("Training SB3 on parallel_env (DIY Centralized Wrapper)")
    print("=" * 70)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print("Approach: DIY centralized (single policy controls all agents)")
    print("=" * 70 + "\n")

  # Create base environment with direct instantiation
  print("Creating NekEnv...")
  env_config = {
      'environment_name': args.env,
      'nproc': args.nproc,
      'use_clean_cache': False,
      'local_fallback_dir': args.local_dir,
      'configuration_file': args.config_file,
  }
  base_env = NekEnv(env_config=env_config)
  # [YW-MOD] Rewrite the par file to ensure the simulation configuration is correct
  from hydrogym.nek.nek_lib.nek_utils import NEK_INIT
  nek_init = NEK_INIT(nek=base_env.conf.simulation, drl=base_env.conf.runner, rank_folder=base_env.run_folder)
  nek_init.rewrite_REA_v19() # Rewrite the par file, v19 corresponds to the new Nek5000 format
  # [YW-MOD] End

    # Wrap with parallel interface (dict-based)
    print("Wrapping with NekParallelEnv (dict-based)...")
    parallel_env = NekParallelEnv(base_env)

    # Wrap with DIY centralized wrapper (array-based, SB3-compatible)
    print("Wrapping with CentralizedParallelWrapper (SB3-compatible)...")
    env = CentralizedParallelWrapper(parallel_env, reward_agg="mean")

    print("\nEnvironment created:")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Total observation space: {env.observation_space.shape}")
    print(f"  Total action space: {env.action_space.shape}")
    print()

    # Import SB3
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback

        if args.algo == "PPO":
            from stable_baselines3 import PPO as Algorithm
        elif args.algo == "TD3":
            from stable_baselines3 import TD3 as Algorithm
        elif args.algo == "SAC":
            from stable_baselines3 import SAC as Algorithm
    except ImportError:
        print("✗ Error: Stable-Baselines3 not installed!")
        sys.exit(1)

    # Wrap with Monitor, DummyVecEnv, VecNormalize
    print("Wrapping with Monitor, DummyVecEnv, VecNormalize...")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Final environment:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")

    # Setup checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.algo}_Parallel_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"Creating {args.algo} model (centralized policy)...")

    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "verbose": 1,
        "tensorboard_log": str(log_dir),
    }

    if args.algo == "PPO":
        model_kwargs["n_steps"] = args.n_steps
        model_kwargs["batch_size"] = args.batch_size
    else:
        model_kwargs["batch_size"] = args.batch_size
        model_kwargs["buffer_size"] = 100000
        model_kwargs["learning_starts"] = 100
        model_kwargs["train_freq"] = 1
        model_kwargs["gradient_steps"] = 1

    model = Algorithm(**model_kwargs)
    print("✓ Model created\n")

    # Calculate safe save frequency
    safe_save_freq = max(args.save_freq // env.num_envs, 1)

    # Custom callback to save VecNormalize stats
    class SaveVecNormalizeCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            super_result = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                stats_path = Path(self.save_path) / f"vec_normalize_{self.num_timesteps}_steps.pkl"
                self.training_env.save(str(stats_path))
            return super_result

    checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=safe_save_freq, save_path=str(log_dir), name_prefix="model"
    )

    print(f"Log directory: {log_dir}\n")

    # Train
    print("=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, tb_log_name=f"{args.algo}_run")

        # Save final model and normalization stats
        final_model_path = log_dir / "model_final.zip"
        final_stats_path = log_dir / "vec_normalize_final.pkl"

        model.save(final_model_path)
        env.save(str(final_stats_path))

        print("\n" + "=" * 70)
        print("✓ Training completed!")
        print(f"✓ Model saved to: {final_model_path}")
        print(f"✓ Normalization stats saved to: {final_stats_path}")
        print("\nNote: This DIY centralized wrapper shows how dict→array conversion works.")
        print("For production use, see chapter 3 (PettingZoo + SuperSuit).")
        print("For decentralized MARL, use RLlib or similar frameworks.")
        print("=" * 70 + "\n")

    finally:
        env.close()
        print("✓ Environment closed\n")


def main():
    parser = argparse.ArgumentParser(description="Train SB3 on parallel_env (DIY centralized wrapper)")

    # Environment
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--local-dir", default=None, help="Local env directory")
    parser.add_argument("--nproc", type=int, default=10, help="Nek5000 processes")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")

    # Algorithm
    parser.add_argument("--algo", default="PPO", choices=["PPO", "TD3", "SAC"])
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Logging
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--save-freq", type=int, default=10000)

    args = parser.parse_args()

    try:
        train_parallel_centralized(args)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
