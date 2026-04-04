#!/usr/bin/env python3
"""
Train SB3 agent on MAIA single-agent environment.
Includes Monitor, DummyVecEnv, VecNormalize, and TensorBoard best practices.

IMPORTANT: MAIA requires MPMD execution (Multi-Program Multiple Data).
This script must be run with mpirun using the ':' separator:

    mpirun -np 1 python train_sb3_maia.py [args] : -np N maia properties.toml

The ':' separator indicates two separate programs:
    - Process 0: Python script (this file)
    - Processes 1-N: MAIA solver with properties.toml configuration

Usage Examples:
    # Train PPO on Cylinder with 1 MAIA process
    cd work_dir
    mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO \
        --total-timesteps 10000 : -np 1 maia properties.toml

    # Train SAC on Pinball with 4 parallel MAIA processes
    cd work_dir
    mpirun -np 1 python ../train_sb3_maia.py --env Pinball_2D_Re100 --algo SAC \
        --total-timesteps 50000 : -np 4 maia properties.toml

    # Train TD3 with custom probes and probewise normalization
    cd work_dir
    mpirun -np 1 python ../train_sb3_maia.py --env RotaryCylinder_2D_Re1000 \
        --algo TD3 --obs-norm probewise_mean_std : -np 2 maia properties.toml

Workflow:
    1. Prepare workspace (once, outside MPMD):
       python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./train_run_000

    2. Train with MPMD:
       cd train_run_000
       mpirun -np 1 python ../train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO : -np 1 maia properties.toml

    3. Monitor with TensorBoard:
       tensorboard --logdir logs/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Force unbuffered output for MPMD mode
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1)

import hydrogym.maia as maia  # noqa: E402


def parse_range_arg(arg_str: str) -> Tuple[float, float, int]:
    """Parse range argument in format 'min,max,num'."""
    parts = arg_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Range must be in format 'min,max,num', got: {arg_str}")
    return float(parts[0]), float(parts[1]), int(parts[2])


def create_probe_locations(x_range: Tuple[float, float, int], y_range: Tuple[float, float, int]) -> List[float]:
    """
    Create probe location grid for 2D flow field sampling.
    Returns flattened list [x0, y0, x1, y1, ...].
    """
    x_min, x_max, num_x = x_range
    y_min, y_max, num_y = y_range

    xp = np.linspace(x_min, x_max, num_x)
    yp = np.linspace(y_min, y_max, num_y)
    X, Y = np.meshgrid(xp, yp)

    probe_list = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
    probe_locations = [coord for point in probe_list for coord in point]

    return probe_locations


def train_single_agent(args):
    print("=" * 70)
    print("Training SB3 Agent on MAIA Environment (Single Agent)")
    print("=" * 70)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Observation normalization: {args.obs_norm}")
    print("=" * 70 + "\n")

    # Setup checkpointing and logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.algo}_MAIA_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Import SB3 components
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

    # Parse probe locations
    try:
        x_range = parse_range_arg(args.probe_x_range)
        y_range = parse_range_arg(args.probe_y_range)
        probe_locations = create_probe_locations(x_range, y_range)
        num_probes = len(probe_locations) // 2
        print("Probe grid configuration:")
        print(f"  X range: [{x_range[0]}, {x_range[1]}] with {x_range[2]} points")
        print(f"  Y range: [{y_range[0]}, {y_range[1]}] with {y_range[2]} points")
        print(f"  Total probes: {num_probes}\n")
    except ValueError as e:
        print(f"✗ Error parsing probe ranges: {e}")
        sys.exit(1)

    # 1. Safe Environment Creation Function
    def make_env():
        """
        Create MAIA environment with proper configuration.
        Note: This runs in MPMD mode, so MAIA solver must be launched separately.
        """
        env = maia.from_hf(
            args.env,
            use_clean_cache=False,  # Use pre-prepared workspace
            local_fallback_dir=args.local_dir,
            probe_locations=probe_locations,
            obs_normalization_strategy=args.obs_norm,
        )
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
    parser = argparse.ArgumentParser(
        description="Train SB3 on MAIA Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic PPO training
  mpirun -np 1 python train_sb3_maia.py --env Cylinder_2D_Re200 --algo PPO : -np 1 maia properties.toml

  # SAC with custom probes
  mpirun -np 1 python train_sb3_maia.py --env Pinball_2D_Re100 --algo SAC \
      --probe-x-range 1,10,16 : -np 4 maia properties.toml

Note: This script requires MPMD execution. See docstring for details.
        """,
    )

    # Environment
    parser.add_argument("--env", required=True, help="Environment name (e.g., Cylinder_2D_Re200)")
    parser.add_argument("--local-dir", default=None, help="Local env directory (for offline usage)")
    parser.add_argument(
        "--probe-x-range",
        type=str,
        default="1.0,8.0,8",
        help="X-axis probe range as 'min,max,num' (default: 1.0,8.0,8)",
    )
    parser.add_argument(
        "--probe-y-range",
        type=str,
        default="-1.0,1.0,5",
        help="Y-axis probe range as 'min,max,num' (default: -1.0,1.0,5)",
    )
    parser.add_argument(
        "--obs-norm",
        type=str,
        default="none",
        choices=["U_inf", "probewise_mean_std", "none", "customized"],
        help="Observation normalization strategy (default: none)",
    )

    # Algorithm
    parser.add_argument("--algo", default="PPO", choices=["PPO", "TD3", "SAC"])
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=200, help="PPO: steps per update")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Logging
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--save-freq", type=int, default=1000)

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
