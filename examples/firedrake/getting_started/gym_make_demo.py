#!/usr/bin/env python3
"""
The standard HydroGym entry point: gymnasium's gym.make().

This is the recommended way to start any HydroGym environment. One call,
the same shape regardless of backend -- what happens underneath (an
in-process Firedrake solve, here) is not something you need to know to
use the environment.

Usage:
    python gym_make_demo.py
    python gym_make_demo.py --env hydrogym/RotaryCylinder-v0 --steps 5

For direct solver access, custom configs beyond env_config overrides, or
anything gym.make()'s registered defaults don't cover, see the
lower-level entry point in test_firedrake_env.py (constructs
hydrogym.firedrake.Cylinder + hydrogym.core.FlowEnv directly) or
config_reference.py (the full env_config schema).
"""

import argparse

import gymnasium as gym

import hydrogym.registration  # noqa: F401  (import performs the registration)


def main():
    parser = argparse.ArgumentParser(description="gym.make() entry point demo (Firedrake)")
    parser.add_argument(
        "--env",
        default="hydrogym/Cylinder-v0",
        choices=[
            "hydrogym/Cylinder-v0",
            "hydrogym/RotaryCylinder-v0",
            "hydrogym/Cavity-v0",
            "hydrogym/Pinball-v0",
            "hydrogym/Step-v0",
        ],
        help="Any registered Firedrake environment id (default: %(default)s)",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of steps to run")
    args = parser.parse_args()

    print(f"gym.make({args.env!r})")
    env = gym.make(args.env)

    obs, info = env.reset()
    print(f"reset OK, observation shape: {obs.shape}")

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i + 1}/{args.steps}: reward={reward:.6f}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("Environment closed. Done.")


if __name__ == "__main__":
    main()
