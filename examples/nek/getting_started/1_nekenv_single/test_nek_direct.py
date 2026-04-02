#!/usr/bin/env python3
"""
Simple NEK5000 Environment Test (Direct Instantiation)
=======================================================

Minimal example showing how to:
1. Create a Nek environment by directly calling NekEnv (not using from_hf)
2. Use auto-detection for config files
3. Run a simple control loop with env.step()
4. Test zero control

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
    parser.add_argument("--env", type=str, default="MiniChannel_Re180", help="Environment name")
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 processes")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")
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

    # Modify the par file to ensure the simulation configuration is correct
    from hydrogym.nek.nek_lib.nek_utils import NEK_INIT

    nek_init = NEK_INIT(nek=env.conf.simulation, drl=env.conf.runner, rank_folder=env.run_folder)
    nek_init.rewrite_REA_v19()  # Rewrite the par file, v19 corresponds to the new Nek5000 format
    # The simulation will be reset, so the par file is to be written out at this point

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
    print(f"\nRunning {max_steps} steps with zero control...")

    total_reward = 0.0
    action_dim = env.action_space.shape[0]

    for step in range(max_steps):
        # Define action (example: zero control - baseline)
        # action = np.zeros(action_dim, dtype=np.float32)

        # OR use constant blowing:
        # action = np.ones(action_dim, dtype=np.float32) * 0.01

        # Oppose to the wall-normal vel, as the observation is staggered so we sort the even indices
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
