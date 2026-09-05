#!/usr/bin/env python3
"""
The standard HydroGym entry point: gymnasium's gym.make(), for Nek5000.

Same one-call shape as every other backend -- gym.make() constructs and
wires up the real Nek5000 MPMD coupling underneath, so this must still be
launched as an MPMD job (Python + the nek5000 binary sharing one MPI
world), same as every other Nek example. See prepare_workspace.py for
staging the workspace before this runs.

Usage (from a prepared workspace directory -- run
`python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir .` first):
    mpirun -np 1 python gym_make_demo.py : -np 10 nek5000

nproc is required (unlike Firedrake/MAIA's zero-argument case): it must
match the MPMD launch's actual worker rank count, which gym.make() cannot
know on its own -- passing the wrong value raises a clear error naming the
mismatch, rather than hanging.

For direct solver access, see the lower-level entry point in
1_nekenv_single/test_nek_direct.py (constructs hydrogym.nek.NekEnv
directly).
"""

import argparse

import gymnasium as gym

import hydrogym.registration  # noqa: F401  (import performs the registration)


def main():
    parser = argparse.ArgumentParser(description="gym.make() entry point demo (Nek5000)")
    parser.add_argument(
        "--env",
        default="hydrogym-nek/TCFmini_3D_Re180-v0",
        help="Any registered Nek5000 environment id (default: %(default)s)",
    )
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 ranks (must match the MPMD launch)")
    parser.add_argument("--steps", type=int, default=3, help="Number of steps to run")
    args = parser.parse_args()

    print(f"gym.make({args.env!r}, nproc={args.nproc})")
    env = gym.make(args.env, nproc=args.nproc)

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
