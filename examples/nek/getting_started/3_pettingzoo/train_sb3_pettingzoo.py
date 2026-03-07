#!/usr/bin/env python3
"""
Train SB3 on PettingZoo environment using SuperSuit (Production approach).

PettingZoo provides ecosystem-standard wrappers via SuperSuit that handle
the dict→array conversion needed for SB3. This is the recommended production
approach for multi-agent environments with SB3.

Educational approach (DIY wrapper): See chapter 2
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from hydrogym.nek import NekEnv, make_pettingzoo_env


def train_pettingzoo_with_supersuit(args):
  """Train using SuperSuit wrapper (production approach)."""
  print("=" * 70)
  print("Training SB3 on PettingZoo (with SuperSuit)")
  print("=" * 70)
  print(f"Algorithm: {args.algo}")
  print(f"Environment: {args.env}")
  print("Approach: SuperSuit (production-ready ecosystem wrapper)")
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

  # Wrap with PettingZoo interface
  print("Wrapping with PettingZoo interface...")
  env = make_pettingzoo_env(base_env)

  print("PettingZoo environment created:")
  print(f"  Number of agents: {len(env.possible_agents)}\n")

  # Convert to SB3-compatible format using SuperSuit
  try:
    from supersuit import black_death_v3, pad_observations_v0, pad_action_space_v0
    from pettingzoo.utils import parallel_to_aec
  except ImportError:
    print("✗ Error: PettingZoo/SuperSuit not installed!")
    print("Install with: pip install pettingzoo supersuit")
    sys.exit(1)

  try:
    import numpy as np
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
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

  # Apply SuperSuit wrappers to make SB3-compatible
  print("Applying SuperSuit wrappers for SB3 compatibility...")

  # Pad observations and actions to uniform sizes (required for vectorization)
  env = pad_observations_v0(env)
  env = pad_action_space_v0(env)

  # Handle agent death (for environments where agents can be removed)
  env = black_death_v3(env)

  # Convert parallel to AEC (Agent Environment Cycle) format
  # Then wrap to make it single-agent-like for SB3
  from supersuit import pettingzoo_env_to_vec_env_v1

  print("Converting to vectorized Gym environment...")
  env = pettingzoo_env_to_vec_env_v1(env)

  # Create compatibility wrapper for old VecEnv API
  class VecEnvCompatWrapper(VecEnvWrapper):
    """Wrapper to make new Gymnasium API compatible with old VecEnv API."""

    def reset(self, **kwargs):
      result = self.venv.reset(**kwargs)
      # Handle both old API (just obs) and new API (obs, info)
      if isinstance(result, tuple):
        return result[0]  # Return only obs for old API compatibility
      return result

    def step_wait(self):
      result = self.venv.step_wait()
      # Handle new API: (obs, reward, terminated, truncated, info)
      if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
        return obs, reward, done, info
      # Already old API: (obs, reward, done, info)
      return result

  print("Wrapping for VecEnv API compatibility...")
  env = VecEnvCompatWrapper(env)

  # Wrap with VecNormalize
  print("Wrapping with VecNormalize...")
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  print("Final environment:")
  print(f"  Observation space: {env.observation_space}")
  print(f"  Action space: {env.action_space}\n")

  # Setup checkpointing
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = Path(args.log_dir) / f"{args.algo}_PettingZoo_{timestamp}"
  log_dir.mkdir(parents=True, exist_ok=True)

  # Create model
  print(f"Creating {args.algo} model...")

  model_kwargs = {
      "policy": "MlpPolicy",
      "env": env,
      "learning_rate": args.learning_rate,
      "gamma": args.gamma,
      "verbose": 1,
      "tensorboard_log": str(log_dir)
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
        stats_path = Path(
            self.save_path) / f"vec_normalize_{self.num_timesteps}_steps.pkl"
        self.training_env.save(str(stats_path))
      return super_result

  checkpoint_callback = SaveVecNormalizeCallback(
      save_freq=safe_save_freq, save_path=str(log_dir), name_prefix="model")

  print(f"Log directory: {log_dir}\n")

  # Train
  print("=" * 70)
  print("Starting training...")
  print("=" * 70 + "\n")

  try:
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"{args.algo}_run")

    # Save final model and normalization stats
    final_model_path = log_dir / "model_final.zip"
    final_stats_path = log_dir / "vec_normalize_final.pkl"

    model.save(final_model_path)
    env.save(str(final_stats_path))

    print("\n" + "=" * 70)
    print("✓ Training completed!")
    print(f"✓ Model saved to: {final_model_path}")
    print(f"✓ Normalization stats saved to: {final_stats_path}")
    print(
        "\nNote: SuperSuit provides ecosystem-standard PettingZoo→SB3 conversion."
    )
    print("For educational DIY wrapper, see chapter 2.")
    print("=" * 70 + "\n")

  finally:
    env.close()
    print("✓ Environment closed\n")


def main():
  parser = argparse.ArgumentParser(
      description="Train SB3 on PettingZoo environment (SuperSuit wrapper)")

  # Environment
  parser.add_argument("--env", required=True, help="Environment name")
  parser.add_argument("--local-dir", default=None, help="Local env directory")
  parser.add_argument("--nproc", type=int, default=10, help="Nek5000 processes")
  parser.add_argument(
      "--config-file",
      type=str,
      default=None,
      help="Config file (None = auto-detect)")

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
    train_pettingzoo_with_supersuit(args)
    sys.exit(0)
  except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


if __name__ == "__main__":
  main()
