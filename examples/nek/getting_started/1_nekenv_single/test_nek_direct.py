#!/usr/bin/env python3
"""
Simple NEK5000 Environment Test (Direct Instantiation)
=======================================================

Minimal example showing how to:
1. Create a Nek environment by directly calling NekEnv (not using from_hf)
2. Use auto-detection for config files
3. Run a simple control loop with env.step()
4. Apply opposition control (a = -v' of each actuator)

Usage:
    # Run with default steps
    mpirun -np 1 python test_nek_direct.py : -np 10 nek5000

    # Override number of steps
    mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000

Note:
    - This directly instantiates NekEnv(env_config=...) instead of from_hf()
    - Config auto-detection: if configuration_file is not specified,
      it will look for 'environment_config.yaml' or 'config.yaml'
      in the environment directory
    - use_clean_cache=False uses existing cached/prepared workspace
    - Zero control: action = 0 (baseline test)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hydrogym.nek import NekEnv


def main():
    parser = argparse.ArgumentParser(description="Simple Nek5000 test with direct instantiation")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--env", type=str, default="TCFmini_3D_Re180", help="Environment name")
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 processes")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")
    parser.add_argument(
        "--baseline-dudy",
        type=float,
        default=None,
        help="Override baseline dU/dy for reward normalization. Required for wing-type envs "
        "(e.g. NACA4412_3D_Re75000_AOA5) whose config has no normalization.dUdy/runner.dUdy "
        "(reward there varies with streamwise position, not dU/dy) - see zeroshot_demo_pettingzoo.py.",
    )
    args = parser.parse_args()

    # Create environment by directly calling NekEnv
    # (alternative to NekEnv.from_hf(...))
    print(f"\nCreating Nek5000 environment: {args.env}")
    print("Method: Direct NekEnv instantiation with env_config")

    env_config = {
        "environment_name": args.env,
        "nproc": args.nproc,
        "use_clean_cache": False,
        "local_fallback_dir": args.local_dir,
        "configuration_file": args.config_file,  # None = auto-detect
    }

    # Direct instantiation
    env = NekEnv(env_config=env_config)

    if args.baseline_dudy is not None:
        env.baseline_dudy = args.baseline_dudy

    # The case's parameter file (.rea / .par) is read by Nek5000 when it is launched; prepare_workspace.py
    # writes the configuration into the .rea of v17 cases before the launch.

    print("\nEnvironment info:")
    print("=" * 80)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Action bounds: [{env.action_space.low[0]:.6f}, {env.action_space.high[0]:.6f}]")
    print("=" * 80)

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"  Initial observation shape: {obs.shape}")

    # Run simulation with simple control
    max_steps = args.steps
    print(f"\nRunning {max_steps} steps with opposition control...")

    total_reward = 0.0

    for step in range(max_steps):
        # Define action (example: zero control - baseline)
        # action = np.zeros(env.action_space.shape, dtype=np.float32)

        # OR use constant blowing:
        # action = np.full(env.action_space.shape, 0.01, dtype=np.float32)

        # Opposition control: the observation is [u', v'] per actuator, so the odd indices are v'
        action = -obs[1::2]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(f"  Step {step:3d}: reward = {reward:8.6f}, total = {total_reward:10.6f}")

        if terminated or truncated:
            print(f"  Episode terminated at step {step}")
            break

    # Summary
    print("\n" + "=" * 80)
    print("Test completed!")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {total_reward:.6f}")
    print(f"  Average reward: {total_reward / (step + 1):.6f}")
    print("=" * 80)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
