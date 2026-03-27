#!/usr/bin/env python3
"""
Train SB3 agent using NekEnv.from_hf() pattern (MAIA pattern).

Demonstrates the convenient from_hf() method for loading environments
with minimal configuration. This is the recommended pattern for most users.

Pattern: NekEnv.from_hf(env_name, nproc, ...)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from hydrogym.nek import NekEnv


def train_with_from_hf(args):
    """Train using from_hf() pattern (MAIA pattern)."""
    print("=" * 70)
    print("Training SB3 with from_hf() Pattern")
    print("=" * 70)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print("Pattern: NekEnv.from_hf()")
    print("=" * 70 + "\n")

  # Create environment using from_hf() pattern
  print("Creating environment with NekEnv.from_hf()...")
  env = NekEnv.from_hf(
      args.env,
      nproc=args.nproc,
      use_clean_cache=False,
      local_fallback_dir=args.local_dir,
  )
  # [YW-MOD] Rewrite the par file to ensure the simulation configuration is correct
  from hydrogym.nek.nek_lib.nek_utils import NEK_INIT
  nek_init = NEK_INIT(nek=env.conf.simulation, drl=env.conf.runner, rank_folder=env.run_folder)
  nek_init.rewrite_REA_v19() # Rewrite the par file, v19 corresponds to the new Nek5000 format
  # [YW-MOD] End

    print("\nEnvironment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print()

    # Import SB3
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback

    if args.algo == "PPO":
      # NOTE: PPO is not used in the literature, so it is not guaranteed to work.
      from stable_baselines3 import PPO as Algorithm
    elif args.algo == "TD3":
      from stable_baselines3 import TD3 as Algorithm
    elif args.algo == "SAC":
      from stable_baselines3 import SAC as Algorithm
  except ImportError:
    print("✗ Stable-Baselines3 not installed!")
    sys.exit(1)

  # Wrap with Monitor, DummyVecEnv, VecNormalize
  print("Wrapping with Monitor, DummyVecEnv, VecNormalize...")
  env = Monitor(env)
  env = DummyVecEnv([lambda: env])
  # NOTE: Add VecNormalize is not used in the literature, so it is not guaranteed to work.
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Environment wrapped:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.algo}_fromHF_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"Creating {args.algo} model...")
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
        print("\nfrom_hf() Pattern Benefits:")
        print("  - Minimal configuration (just env name + nproc)")
        print("  - Auto-detects config files")
        print("  - Handles local and HuggingFace sources")
        print("  - Recommended for most users")
        print("=" * 70 + "\n")

    finally:
        env.close()
        print("✓ Environment closed\n")


def main():
    parser = argparse.ArgumentParser(description="Train SB3 with from_hf() pattern")

    # Environment
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--local-dir", default=None, help="Local fallback directory")
    parser.add_argument("--nproc", type=int, default=10, help="Nek5000 processes")

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
        train_with_from_hf(args)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
