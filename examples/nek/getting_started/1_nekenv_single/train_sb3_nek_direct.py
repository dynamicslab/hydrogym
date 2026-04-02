#!/usr/bin/env python3
"""
Train SB3 agent on NEK5000 single-agent environment (NekEnv).
Includes Monitor, DummyVecEnv, TensorBoard, and VecNormalize best practices.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from hydrogym.nek import NekEnv


def train_single_agent(args):
    print("=" * 70)
    print("Training SB3 Agent on NekEnv (Single Agent)")
    print("=" * 70)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print("=" * 70 + "\n")

    # Setup checkpointing and logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.algo}_NekEnv_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Import SB3 components
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

    # 1. Safe Environment Creation Function
    def make_env():
        env_config = {
            'environment_name': args.env,
            'nproc': args.nproc,
            'use_clean_cache': False,
            'local_fallback_dir': args.local_dir,
            'configuration_file': args.config_file,
        }
        env = NekEnv(env_config=env_config)

        # Modify the par file to ensure the simulation configuration is correct before training
        from hydrogym.nek.nek_lib.nek_utils import NEK_INIT
        nek_init = NEK_INIT(nek=env.conf.simulation, drl=env.conf.runner, rank_folder=env.run_folder)
        nek_init.rewrite_REA_v19() # Rewrite the par file, v19 corresponds to the new Nek5000 format
        env = Monitor(env)  # CRITICAL: Enables episode reward/length logging
        return env

    # 2. Wrap in DummyVecEnv
    env = DummyVecEnv([make_env])

    # 3. Apply VecNormalize (Crucial for Fluid Dynamics)
    # This scales inputs to mean 0, std 1 so the Neural Net learns faster.
    # VecNormalize is not used in the literature, so it is not guaranteed to work.
    # Please see MARL set for more details.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Environment created (Wrapped in Monitor, DummyVecEnv, VecNormalize):")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")

    # Check for SAC/TD3 continuous action space requirement
    if args.algo in ["TD3", "SAC"] and not hasattr(env.action_space, "high"):
        print(f"✗ Error: {args.algo} requires a continuous action space (Box).")
        sys.exit(1)

    # Create model
    print(f"Creating {args.algo} model...")
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "verbose": 1,
        "tensorboard_log": str(log_dir),  # Enables TensorBoard tracking
    }

    if args.algo == "PPO":
        model_kwargs["n_steps"] = args.n_steps
        model_kwargs["batch_size"] = args.batch_size
    else:  # TD3, SAC setup optimized for slower CFD step times
        model_kwargs.update(
            {
                "batch_size": args.batch_size,
                "buffer_size": 100000,  # Use 10k only if you run out of RAM
                "learning_starts": 100,  # Don't wait too long in expensive CFD envs
                "train_freq": 1,
                "gradient_steps": 1,
            }
        )

    model = Algorithm(**model_kwargs)
    print("✓ Model created\n")

    # Calculate safe save frequency based on VecEnv logic
    safe_save_freq = max(args.save_freq // env.num_envs, 1)

    # Custom CheckpointCallback that also saves the VecNormalize statistics
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

        # Save final model AND normalization stats
        final_model_path = log_dir / "model_final.zip"
        final_stats_path = log_dir / "vec_normalize_final.pkl"

        model.save(final_model_path)
        env.save(str(final_stats_path))

        print("\n" + "=" * 70)
        print("✓ Training completed!")
        print(f"✓ Model saved to: {final_model_path}")
        print(f"✓ Normalization stats saved to: {final_stats_path}")
        print("=" * 70 + "\n")

    finally:
        env.close()
        print("✓ Environment closed\n")


def main():
    parser = argparse.ArgumentParser(description="Train SB3 on NekEnv")

    # Environment
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--local-dir", default=None, help="Local env directory")
    parser.add_argument("--nproc", type=int, default=10, help="Nek5000 processes")
    parser.add_argument("--work-dir", default="./train_run", help="Work directory")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")

    # Algorithm
    parser.add_argument("--algo", default="PPO", choices=["PPO", "TD3", "SAC"])
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO: steps per update")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Logging
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--save-freq", type=int, default=10000)

    args = parser.parse_args()

    try:
        train_single_agent(args)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
