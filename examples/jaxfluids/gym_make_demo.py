#!/usr/bin/env python3
"""
The standard HydroGym entry point: gymnasium's gym.make(), for JAX-Fluids.

Same one-call shape as Firedrake/MAIA -- in-process, no MPMD needed.

Usage:
    python gym_make_demo.py
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python gym_make_demo.py --env hydrogym-jaxfluids/Nozzle3D-v0 --steps 1
    # Nozzle3D_coarse: about 70 GB of GPU memory; with 4 injectors one action spans the whole episode (README)

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

    # The 3D nozzle needs the number of injectors (4-12); the published Nozzle3D configuration does not set it.
    env_config = {"jaxfluids": {"num_actuators": 4}} if "Nozzle3D" in args.env else {}
    print(f"gym.make({args.env!r}, env_config={env_config})")
    env = gym.make(args.env, env_config=env_config)

    obs, info = env.reset()
    print(f"reset OK, observation shape: {obs.shape}")

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  step {i + 1}/{args.steps}: reward={float(reward):.6f}, terminated={terminated}, truncated={truncated}"
        )
        if (bool(terminated) or bool(truncated)) and i + 1 < args.steps:
            print("  episode ended: reset")
            obs, info = env.reset()

    env.close()
    print("Environment closed. Done.")


if __name__ == "__main__":
    main()
