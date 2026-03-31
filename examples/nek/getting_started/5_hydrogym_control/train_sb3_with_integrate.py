#!/usr/bin/env python3
"""
Train SB3 agent then evaluate using integrate().

This demonstrates the complete workflow:
1. Train RL agent using standard SB3 with Monitor & VecNormalize
2. Save trained model and normalization stats
3. Evaluate trained model using integrate() for rollouts
4. Compare RL policy with classical controllers

Shows how integrate() works with both RL policies and classical control functions.

Usage:
    # MAIA pattern (recommended)
    mpirun -np 1 python train_sb3_with_integrate.py --env MiniChannel_Re180 --nproc 10 : -np 10 nek5000

    # Legacy pattern with config
    mpirun -np 1 python train_sb3_with_integrate.py --config config.yml --nproc 10 : -np 10 nek5000
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from hydrogym.nek import NekEnv, integrate


def create_environment(env_name: Optional[str],
                       nproc: int,
                       config_path: Optional[str],
                       local_dir: Optional[str],
                       config_file: Optional[str] = None):
  """
    Create NekEnv with flexible initialization.

    Args:
        env_name: Environment name for MAIA pattern
        nproc: Number of Nek5000 processes
        config_path: Path to YAML config (legacy)
        local_dir: Local fallback directory
        config_file: Override config file path

    Returns:
        NekEnv instance
    """
  if env_name:
    # MAIA pattern (recommended)
    print(f"  Using MAIA pattern: environment={env_name}, nproc={nproc}")
    return NekEnv.from_hf(
        env_name,
        nproc=nproc,
        use_clean_cache=False,
        local_fallback_dir=local_dir,
        configuration_file=config_file,
    )
  elif config_path:
    # Legacy pattern with config file
    print(f"  Using legacy pattern with config: {config_path}")
    from omegaconf import OmegaConf
    from hydrogym.nek.configs import Config

    config = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(config_path),
    )
    return NekEnv(conf=config, reward_agg='mean')
  else:
    raise ValueError(
        "Must provide either env_name (MAIA) or config_path (legacy)")


def classical_controllers():
  """Return dictionary of classical control functions."""

  def opposition_control(t, obs, env):
    """Opposition control: v = -alpha * observation."""
    return -obs[:env.action_space.shape[0]] * 0.5

  def zero_control(t, obs, env):
    """Zero control (baseline)."""
    return np.zeros(env.action_space.shape, dtype=np.float32)

  def sinusoidal_control(t, obs, env):
    """Sinusoidal control."""
    return np.sin(2 * np.pi * t * 0.1) * 0.5 * np.ones(
        env.action_space.shape, dtype=np.float32)

  return {
      'opposition': opposition_control,
      'zero': zero_control,
      'sinusoidal': sinusoidal_control
  }


def train_and_evaluate(args):
  """Train RL agent then evaluate with integrate()."""
  print("=" * 70)
  print("Train & Evaluate with integrate()")
  print("=" * 70)
  print(f"Environment: {args.env or args.config}")
  print(f"Algorithm: {args.algo}")
  print("=" * 70 + "\n")

  # ====================
  # Phase 1: Training
  # ====================
  print("PHASE 1: Training RL Agent")
  print("-" * 70)

  # Create environment with flexible initialization
  print("Creating training environment...")
  env_train = create_environment(
      env_name=args.env,
      nproc=args.nproc,
      config_path=args.config,
      local_dir=args.local_dir,
      config_file=args.config_file,
  )

  # Import SB3
  try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if args.algo == "PPO":
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
  env_train = Monitor(env_train)
  env_train = DummyVecEnv([lambda: env_train])
  env_train = VecNormalize(
      env_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

  print("Environment created:")
  print(f"  Observation space: {env_train.observation_space}")
  print(f"  Action space: {env_train.action_space}\n")

  # Create model
  print(f"Creating {args.algo} model...")

  model_kwargs = {
      "policy": "MlpPolicy",
      "env": env_train,
      "learning_rate": args.learning_rate,
      "gamma": args.gamma,
      "verbose": 1
  }

  if args.algo == "PPO":
    model_kwargs["n_steps"] = args.n_steps
    model_kwargs["batch_size"] = args.batch_size
  else:
    model_kwargs["batch_size"] = args.batch_size
    model_kwargs["buffer_size"] = 100000
    model_kwargs["learning_starts"] = 100

  model = Algorithm(**model_kwargs)
  print("✓ Model created\n")

  # Train
  print("Training...")
  model.learn(total_timesteps=args.total_timesteps)

  # Save model and normalization stats
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = Path(args.log_dir) / f"{args.algo}_Integrate_{timestamp}"
  log_dir.mkdir(parents=True, exist_ok=True)

  model_path = log_dir / "trained_model.zip"
  stats_path = log_dir / "vec_normalize.pkl"

  model.save(model_path)
  env_train.save(str(stats_path))

  print("\n✓ Training completed")
  print(f"✓ Model saved: {model_path}")
  print(f"✓ Normalization stats saved: {stats_path}\n")

  env_train.close()

  # ====================
  # Phase 2: Evaluation with integrate()
  # ====================
  print("\nPHASE 2: Evaluation with integrate()")
  print("-" * 70)

  # Create fresh environment for evaluation (unwrapped)
  print("Creating evaluation environment...")
  env_eval = create_environment(
      env_name=args.env,
      nproc=args.nproc,
      config_path=args.config,
      local_dir=args.local_dir,
      config_file=args.config_file,
  )

  # Load trained model with normalization stats
  print("Loading trained model and normalization stats...")
  trained_model = Algorithm.load(model_path)

  # Load VecNormalize stats for wrapping evaluation env
  from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
  env_eval_wrapped = DummyVecEnv([lambda: env_eval])
  env_eval_wrapped = VecNormalize.load(str(stats_path), env_eval_wrapped)
  env_eval_wrapped.training = False  # Don't update stats during evaluation
  env_eval_wrapped.norm_reward = False  # Don't normalize rewards during evaluation

  # Get classical controllers
  classical = classical_controllers()

  print("\nEvaluating controllers with integrate()...")
  print(f"  Running {args.eval_steps} steps per controller\n")

  for name, controller in [('RL Policy', trained_model),
                           ('Opposition', classical['opposition']),
                           ('Zero Control', classical['zero']),
                           ('Sinusoidal', classical['sinusoidal'])]:
    print(f"  Testing: {name}")

    # Reset environment
    if name == 'RL Policy':
      # Use wrapped env for RL policy (needs normalization)
      env_eval_wrapped.reset()
      eval_env = env_eval_wrapped
    else:
      # Use unwrapped env for classical controllers
      env_eval.reset()
      eval_env = env_eval

    # Run with integrate
    try:
      integrate(eval_env, controller=controller, num_steps=args.eval_steps)
      print(f"    ✓ Completed {args.eval_steps} steps")

    except Exception as e:
      print(f"    ✗ Failed: {e}")

  print("\n" + "=" * 70)
  print("Workflow Complete!")
  print("=" * 70)
  print("\nDemonstrated:")
  print("  1. Training RL agent with SB3 (Monitor, VecNormalize)")
  print("  2. Saving model and normalization stats")
  print("  3. Loading model and stats for evaluation")
  print("  4. Evaluating with integrate() function")
  print("  5. Comparing RL policy vs classical controllers")
  print(f"\nModel saved at: {model_path}")
  print(f"Stats saved at: {stats_path}")
  print("=" * 70 + "\n")

  env_eval.close()


def main():
  parser = argparse.ArgumentParser(
      description="Train then evaluate with integrate()",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__)

  # Environment specification (either --env or --config)
  parser.add_argument(
      "--env",
      type=str,
      default=None,
      help="Environment name from HuggingFace (e.g., MiniChannel_Re180)")
  parser.add_argument(
      "--config",
      type=str,
      default=None,
      help="Path to YAML configuration file (legacy, optional)")
  parser.add_argument(
      "--nproc",
      type=int,
      required=True,
      help="Number of Nek5000 processes (required)")
  parser.add_argument(
      "--local-dir",
      type=str,
      default=None,
      help="Local fallback directory for environments")
  parser.add_argument(
      "--config-file",
      type=str,
      default=None,
      help="Override config file path (None = auto-detect)")

  # Algorithm parameters
  parser.add_argument(
      "--algo",
      default="PPO",
      choices=["PPO", "TD3", "SAC"],
      help="RL algorithm to use (default: PPO)")
  parser.add_argument(
      "--total-timesteps",
      type=int,
      default=50000,
      help="Total timesteps for training (default: 50000)")
  parser.add_argument(
      "--n-steps",
      type=int,
      default=2048,
      help="Number of steps per update (PPO only, default: 2048)")
  parser.add_argument(
      "--learning-rate",
      type=float,
      default=3e-4,
      help="Learning rate (default: 3e-4)")
  parser.add_argument(
      "--batch-size",
      type=int,
      default=64,
      help="Batch size for training (default: 64)")
  parser.add_argument(
      "--gamma",
      type=float,
      default=0.99,
      help="Discount factor (default: 0.99)")

  # Evaluation
  parser.add_argument(
      "--eval-steps",
      type=int,
      default=100,
      help="Steps for integrate() evaluation (default: 100)")

  # Logging
  parser.add_argument(
      "--log-dir",
      default="./logs",
      help="Directory for saving logs and models (default: ./logs)")

  args = parser.parse_args()

  # Validate arguments
  if not args.env and not args.config:
    parser.error(
        "Must provide either --env (MAIA pattern) or --config (legacy pattern)")

  try:
    train_and_evaluate(args)
    sys.exit(0)
  except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


if __name__ == "__main__":
  main()
