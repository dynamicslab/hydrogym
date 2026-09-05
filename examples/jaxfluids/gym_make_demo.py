#!/usr/bin/env python3
"""
The standard HydroGym entry point: gymnasium's gym.make(), for JAX-Fluids.

Same one-call shape as Firedrake/MAIA -- in-process, no MPMD needed.

Usage:
    python gym_make_demo.py
    python gym_make_demo.py --env hydrogym-jaxfluids/Nozzle3D-v0 --steps 2

There is no plain "Nozzle2D"/"Nozzle3D" HF environment -- only
resolution-suffixed variants (Nozzle2D_coarse/_fine, Nozzle3D_coarse/_fine).
gym.make() defaults to "_coarse" for both registered ids; pass
env_config={"environment_name": "Nozzle2D_fine"} to use a different one.

For direct solver access, see the lower-level entry point in
test_jaxfluids_env.py (constructs hydrogym.jaxfluids.envs.Nozzle2D directly).
"""

import argparse

import gymnasium as gym

import hydrogym.registration  # noqa: F401  (import performs the registration)


def main():
    parser = argparse.ArgumentParser(description="gym.make() entry point demo (JAX-Fluids)")
    parser.add_argument(
        "--env",
        default="hydrogym-jaxfluids/Nozzle2D-v0",
        choices=["hydrogym-jaxfluids/Nozzle2D-v0", "hydrogym-jaxfluids/Nozzle3D-v0"],
        help="Any registered JAX-Fluids environment id (default: %(default)s)",
    )
    parser.add_argument("--steps", type=int, default=1, help="Number of steps to run")
    args = parser.parse_args()

    print(f"gym.make({args.env!r})")
    env = gym.make(args.env)

    obs, info = env.reset()
    print(f"reset OK, observation shape: {obs.shape}")

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i + 1}/{args.steps}: reward={float(reward):.6f}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("Environment closed. Done.")


if __name__ == "__main__":
    main()
