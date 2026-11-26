#!/usr/bin/env python3
"""
Simple NEK5000 Environment Test (from_hf)
===============================================

Minimal example showing how to:
1. Create a Nek environment from Hugging Face (from_hf)
2. Run a simple control loop with env.step()
3. Test opposition control

Usage:
    # Run with default steps
    mpirun -np 1 python test_nek_DM.py : -np 10 nek5000

    # Override number of steps
    mpirun -np 1 python test_nek_DM.py --steps 100 : -np 10 nek5000

Note:
    - This uses NekEnv.from_hf()
    - use_clean_cache=False uses existing cached/prepared workspace
    - Opposition control: action = -v' of each actuator (obs[1::2])
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hydrogym.nek import NekEnv


def main():
    parser = argparse.ArgumentParser(description="Simple Nek5000 test with NekEnv.from_hf()")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--env", type=str, default="TCFmini_3D_Re180", help="Environment name")
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 processes")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")
    args = parser.parse_args()

    # Create environment with NekEnv.from_hf()
    print(f"\nCreating Nek5000 environment: {args.env}")
    env = NekEnv.from_hf(
        args.env,
        nproc=args.nproc,
        use_clean_cache=False,
        local_fallback_dir=args.local_dir,
    )

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
        # Define action (example: opposition control)
        action = -obs[1::2]  # opposition control: obs is [u', v'] per actuator, oppose v'

        # OR use constant blowing:
        # action = np.full(env.action_space.shape, 0.01, dtype=np.float32)

        # OR use zero control:
        # action = np.zeros(env.action_space.shape, dtype=np.float32)

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
