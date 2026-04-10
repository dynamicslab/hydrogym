"""
3D Turbulent Channel Flow JAX Environment (Re_tau = 180)

Usage:
    python test_channel_env.py [mode] [--num-steps N]

Modes:
    no_actuation    Baseline: zero actuation, free turbulent evolution (default)
    drag_reduction  Small uniform suction across all 24 jets
    strong_actuation  Alternating ±0.5 checkerboard jet pattern
"""

import argparse
import time

import jax
import jax.numpy as jnp

from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv


def make_action(mode: str, action_dim: int) -> jnp.ndarray:
    if mode == "no_actuation":
        return jnp.zeros((action_dim,))
    elif mode == "drag_reduction":
        return 0.01 * jnp.ones((action_dim,))
    elif mode == "strong_actuation":
        nx_jets, ny_jets = 6, 4
        i_idx = jnp.arange(nx_jets)[:, None]
        j_idx = jnp.arange(ny_jets)[None, :]
        pattern = jnp.where((i_idx + j_idx) % 2 == 0, 0.5, -0.5)
        return pattern.reshape(-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Channel flow JAX environment runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="no_actuation",
        choices=["no_actuation", "drag_reduction", "strong_actuation"],
    )
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point precision for the solver (default: float32)",
    )
    args = parser.parse_args()

    mode_descriptions = {
        "no_actuation": (
            "Baseline: zero actuation — free turbulent evolution\n"
            "  action = 0 (all 24 jet amplitudes set to zero)\n"
            "  Shows natural WSS fluctuations without control."
        ),
        "drag_reduction": (
            "Drag reduction: small uniform suction at the walls\n"
            "  action = 0.01 (uniform positive amplitude across all 24 jets)\n"
            "  Gentle blowing/suction perturbs near-wall streaks to reduce drag.\n"
            "  Expected: WSS decreases relative to the no-actuation baseline."
        ),
        "strong_actuation": (
            "Strong actuation: alternating high-amplitude jets\n"
            "  action alternates ±0.5 across the 6×4 jet grid\n"
            "  Stress-tests the actuator: large perturbations to near-wall flow.\n"
            "  NOTE: very large actions may destabilize the simulation."
        ),
    }

    print("=== 3D Turbulent Channel Flow JAX Environment (Re_tau=180) ===")
    print(f"Mode:      {args.mode}")
    print(f"Dtype:     {args.dtype}")
    print(f"RL steps:  {args.num_steps}  (= {args.num_steps * 50} DNS sub-steps, Δt_DNS=2e-4)")
    print()
    print(mode_descriptions[args.mode])
    print()

    # HuggingFace download happens here on first run (cached afterwards at ~/.cache/hydrogym/)
    env = ChannelFlowSpectralEnv(env_config={"dtype": args.dtype})
    params = env.default_params

    @jax.jit
    def jit_reset(key, params):
        return env.reset_env(key, params)

    @jax.jit
    def jit_step(key, state, action, params):
        return env.step_env(key, state, action, params)

    print("Compiling the environment (this will take a minute or two)...")
    key = jax.random.PRNGKey(0)
    action = make_action(args.mode, params.action_dim)

    # Warmup: triggers XLA compilation
    obs, state = jit_reset(key, params)
    obs, state, reward, done, info = jit_step(key, state, action, params)
    obs.block_until_ready()
    print("Compilation finished! Now running at full speed.\n")

    # Reset with a fresh key for the actual run
    key = jax.random.PRNGKey(1)
    obs, state = jit_reset(key, params)

    print(f"{'Step':>5}  {'WSS':>12}  {'reward':>12}")
    print("-" * 35)

    start_time = time.time()
    for i in range(args.num_steps):
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = jit_step(subkey, state, action, params)
        # reward = -WSS, so WSS = -reward
        print(f"{i:>5}  {float(-reward):>12.6f}  {float(reward):>12.6f}")

    obs.block_until_ready()
    elapsed = time.time() - start_time
    print()
    print(f"Total time for {args.num_steps} steps: {elapsed:.4f} s")
    print(f"Time per step:                        {elapsed / args.num_steps:.4f} s")


if __name__ == "__main__":
    main()
