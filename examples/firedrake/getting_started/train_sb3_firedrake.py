#!/usr/bin/env python3
"""
Train SB3 agent on Firedrake environment.
Includes Monitor, DummyVecEnv, VecNormalize, and TensorBoard best practices.

Unlike MAIA/Nek, Firedrake runs directly in Python (no MPMD coupling).
This makes training simpler - just run: python train_sb3_firedrake.py [args]

Usage Examples:
    # Basic training (single process)
    python train_sb3_firedrake.py --env cylinder --algo PPO --total-timesteps 10000

    # With custom Reynolds number and mesh
    python train_sb3_firedrake.py --env cylinder --reynolds 100 --mesh medium --algo SAC

    # With custom observation type
    python train_sb3_firedrake.py --env cylinder --obs-type velocity_probes --algo TD3

Available environments:
    - cylinder: Flow around circular cylinder with jet actuation
    - rotary_cylinder: Flow around rotating cylinder
    - pinball: Flow around three cylinders (pinball configuration)
    - cavity: Lid-driven cavity flow
    - step: Backward-facing step flow

Recommended dt by environment:
    - cylinder: 1e-2
    - rotary_cylinder: 1e-2
    - pinball: 1e-2
    - cavity: 1e-4 (very stiff!)
    - step: 1e-2 to 1e-3
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np


def train_single_agent(args):
  print("=" * 70)
  print("Training SB3 Agent on Firedrake Environment")
  print("=" * 70)
  print(f"Algorithm: {args.algo}")
  print(f"Environment: {args.env}")
  print(f"Total timesteps: {args.total_timesteps}")
  print("=" * 70 + "\n")

  # Setup checkpointing and logging directory
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = Path(args.log_dir) / f"{args.algo}_Firedrake_{args.env}_{timestamp}"
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
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

  # Import Firedrake components
  try:
    from hydrogym import FlowEnv
    import hydrogym.firedrake as firedrake
  except ImportError as e:
    print(f"✗ Error importing Firedrake: {e}")
    print(
        "Make sure Firedrake is installed: https://www.firedrakeproject.org/download.html"
    )
    sys.exit(1)

  # Map environment names to flow classes
  env_map = {
      'cylinder': firedrake.Cylinder,
      'rotary_cylinder': firedrake.RotaryCylinder,
      'pinball': firedrake.Pinball,
      'cavity': firedrake.Cavity,
      'step': firedrake.Step,
  }

  if args.env not in env_map:
    print(f"✗ Error: Unknown environment: {args.env}")
    print(f"Available: {list(env_map.keys())}")
    sys.exit(1)

  flow_class = env_map[args.env]

  # Default Reynolds numbers and timesteps by environment
  defaults = {
      'cylinder': {
          'Re': 100,
          'dt': 1e-2,
          'mesh': 'medium'
      },
      'rotary_cylinder': {
          'Re': 100,
          'dt': 1e-2,
          'mesh': 'medium'
      },
      'pinball': {
          'Re': 30,
          'dt': 1e-2,
          'mesh': 'medium'
      },
      'cavity': {
          'Re': 7500,
          'dt': 1e-4,
          'mesh': 'medium'
      },
      'step': {
          'Re': 600,
          'dt': 1e-2,
          'mesh': 'medium'
      },
  }

  # Build flow configuration
  flow_config = {}
  if args.reynolds is not None:
    flow_config['Re'] = args.reynolds
  else:
    flow_config['Re'] = defaults[args.env]['Re']

  if args.mesh is not None:
    flow_config['mesh'] = args.mesh
  else:
    flow_config['mesh'] = defaults[args.env]['mesh']

  # Observation type configuration
  if args.obs_type is not None:
    flow_config['observation_type'] = args.obs_type

    # If using probes, create a default grid
    if 'probe' in args.obs_type:
      wake_probes = [(x, y)
                     for x in np.linspace(1.0, 8.0, 10)
                     for y in np.linspace(-1.0, 1.0, 5)]
      flow_config['probes'] = wake_probes
      print(f"Using {len(wake_probes)} wake probes for observation\n")

  # Solver configuration
  if args.dt is not None:
    dt = args.dt
  else:
    dt = defaults[args.env]['dt']

  solver_config = {
      'dt': dt,
      'order': 3,
      'stabilization': 'none',
  }

  # Actuation configuration (multi-substep)
  actuation_config = {
      'num_substeps': args.num_substeps,
      'reward_aggregation': 'mean',
  }

  # Firedrake callbacks (empty list - no callbacks needed for training)
  fd_callbacks = []

  print("Environment configuration:")
  print(f"  Flow: {args.env}")
  print(f"  Reynolds: {flow_config['Re']}")
  print(f"  Mesh: {flow_config['mesh']}")
  print(f"  dt: {dt}")
  print(f"  num_substeps: {args.num_substeps}")
  if args.obs_type:
    print(f"  Observation type: {args.obs_type}")
  print()

  # 1. Safe Environment Creation Function
  def make_env():
    """Create Firedrake environment with proper configuration."""
    env_config = {
        'flow': flow_class,
        'flow_config': flow_config,
        'solver': firedrake.SemiImplicitBDF,
        'solver_config': solver_config,
        'actuation_config': actuation_config,
        'callbacks': fd_callbacks,
        'max_steps': int(1e6),
    }
    env = FlowEnv(env_config)
    env = Monitor(env)  # CRITICAL: Enables episode reward/length logging
    return env

  # 2. Wrap in DummyVecEnv
  env = DummyVecEnv([make_env])

  # 3. Apply VecNormalize (Crucial for Fluid Dynamics)
  # This scales inputs to mean 0, std 1 so the Neural Net learns faster.
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
      "tensorboard_log":
          str(log_dir)  # Enables TensorBoard tracking
  }

  if args.algo == "PPO":
    model_kwargs["n_steps"] = args.n_steps
    model_kwargs["batch_size"] = args.batch_size
  else:  # TD3, SAC setup optimized for slower CFD step times
    model_kwargs.update({
        "batch_size": args.batch_size,
        "buffer_size": 100000,  # Use 10k only if you run out of RAM
        "learning_starts": 100,  # Don't wait too long in expensive CFD envs
        "train_freq": 1,
        "gradient_steps": 1
    })

  model = Algorithm(**model_kwargs)
  print("✓ Model created\n")

  # Calculate safe save frequency based on VecEnv logic
  safe_save_freq = max(args.save_freq // env.num_envs, 1)

  # Custom CheckpointCallback that also saves the VecNormalize statistics
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
  parser = argparse.ArgumentParser(
      description="Train SB3 on Firedrake Environment",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  # Basic PPO training
  python train_sb3_firedrake.py --env cylinder --algo PPO --total-timesteps 10000

  # SAC on cavity with custom mesh
  python train_sb3_firedrake.py --env cavity --algo SAC --mesh fine --total-timesteps 50000

  # MPI parallel for larger simulations
  mpirun -np 4 python train_sb3_firedrake.py --env pinball --algo PPO

Note: Unlike MAIA/Nek, Firedrake runs directly in Python (no MPMD required).
        """)

  # Environment
  parser.add_argument(
      "--env",
      required=True,
      choices=['cylinder', 'rotary_cylinder', 'pinball', 'cavity', 'step'],
      help="Environment type")
  parser.add_argument(
      "--reynolds",
      type=float,
      default=None,
      help="Reynolds number (default: env-specific)")
  parser.add_argument(
      "--mesh",
      type=str,
      choices=['coarse', 'medium', 'fine'],
      default=None,
      help="Mesh resolution (default: env-specific)")
  parser.add_argument(
      "--dt",
      type=float,
      default=None,
      help="Time step (default: env-specific)")
  parser.add_argument(
      "--obs-type",
      type=str,
      choices=[
          'lift_drag', 'stress_sensor', 'velocity_probes', 'pressure_probes',
          'vorticity_probes'
      ],
      default='pressure_probes',
      help="Observation type (default: env-specific)")
  parser.add_argument(
      "--num-substeps",
      type=int,
      default=1,
      help="Number of solver steps per action (default: 1)")

  # Algorithm
  parser.add_argument("--algo", default="PPO", choices=["PPO", "TD3", "SAC"])
  parser.add_argument("--total-timesteps", type=int, default=100000)
  parser.add_argument(
      "--n-steps", type=int, default=200, help="PPO: steps per update")
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
