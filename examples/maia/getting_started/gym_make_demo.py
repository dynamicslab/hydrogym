#!/usr/bin/env python3
"""
The standard HydroGym entry point: gymnasium's gym.make(), for MAIA.

Same one-call shape as every other backend -- gym.make() constructs and
wires up the real MAIA MPMD coupling underneath, so this must still be
launched as an MPMD job (Python + the maia binary sharing one MPI world),
same as every other MAIA example. See prepare_workspace.py for staging
the workspace before this runs.

Usage (from a prepared workspace directory -- run
`python ../prepare_workspace.py --env Cylinder_2D_Re200 --work-dir .` first):
    mpirun -np 1 python gym_make_demo.py : -np 1 maia properties_run.toml

hydrogym-maia/Cylinder_2D_Re200-v0 ships a verified default probe grid, so
no extra arguments are required. Other MAIA ids (e.g.
hydrogym-maia/RotaryCylinder_2D_Re1000-v0) have no universal default probe
grid and require probe_locations=[...] explicitly -- gym.make() raises a
clear error naming this if it's missing, rather than guessing.

For direct solver access or a custom probe grid, see the lower-level
entry point in test_maia_env.py (calls hydrogym.maia.from_hf directly).
"""

import argparse

import gymnasium as gym

import hydrogym.registration  # noqa: F401  (import performs the registration)


def main():
    parser = argparse.ArgumentParser(description="gym.make() entry point demo (MAIA)")
    parser.add_argument(
        "--env",
        default="hydrogym-maia/Cylinder_2D_Re200-v0",
        help="Any registered MAIA environment id (default: %(default)s)",
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
