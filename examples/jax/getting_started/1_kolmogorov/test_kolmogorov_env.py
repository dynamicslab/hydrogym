"""
Kolmogorov Flow JAX Environment

Usage:
    python test_kolmogorov_env.py [mode] [--num-steps N] [--plot]

Modes:
    minimize_tke   Suppress energy bursts: reward_alpha=1.0  (default)
    maximize_tke   Enhance turbulent mixing: reward_alpha=-1.0
    no_actuation   Baseline: zero action, free turbulence evolution

Options:
    --plot         Save a vorticity comparison PNG (baseline vs selected mode)
"""

import argparse
import sys

# Must be set before JAX initializes — check sys.argv directly
import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

jax.config.update("jax_enable_x64", "float32" not in sys.argv)

# ── Mode definitions ────────────────────────────────────────────────────────

MODE_CONFIGS = {
    "minimize_tke": dict(
        reward_alpha=1.0,
        action=jnp.array([-0.25, -0.03, 0.02, 0.01]),
        description=(
            "Objective: Minimize TKE (suppress energy bursts)\n"
            "  reward_alpha =  1.0  ->  reward = -(TKE + action_penalty)\n"
            "  Action: small forcing to damp energy transfer"
        ),
    ),
    "maximize_tke": dict(
        reward_alpha=-1.0,
        action=jnp.array([0.25, 0.03, -0.02, -0.01]),
        description=(
            "Objective: Maximize TKE (enhance turbulent mixing)\n"
            "  reward_alpha = -1.0  ->  reward = TKE - action_penalty\n"
            "  Action: forcing to drive the flow into a more turbulent regime"
        ),
    ),
    "no_actuation": dict(
        reward_alpha=1.0,
        action=None,  # filled in after env init (needs action_dim)
        description=(
            "Baseline: zero actuation (free turbulence evolution)\n"
            "  reward_alpha = 1.0, action = [0, 0, 0, 0]\n"
            "  Shows natural energy bursts without control"
        ),
    ),
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def to_real(omega_hat):
    return np.asarray(jnp.fft.irfftn(omega_hat))


def run_steps(env, params, action, num_steps):
    jit_reset = jax.jit(env.reset_env)
    jit_step = jax.jit(env.step_env)

    key = jax.random.PRNGKey(0)

    print("Compiling the environment (this may take a moment)...")
    obs, state = jit_reset(key, params)
    obs, state, reward, done, info = jit_step(key, state, action, params)
    obs.block_until_ready()
    print("Compilation finished! Now running at full speed.\n")

    key = jax.random.PRNGKey(1)
    obs, state = jit_reset(key, params)

    rows = []
    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = jit_step(subkey, state, action, params)
        rows.append((i, float(info["mean_tke"]), float(reward)))
        if done:
            break
    obs.block_until_ready()
    return state, rows


def save_plot(env, params, action_actuated, outfile="kolmogorov_comparison.png"):
    import matplotlib.pyplot as plt

    zero_action = jnp.zeros((params.action_dim,))

    print("  Running baseline (zero action)...")
    state_zero, _ = run_steps(env, params, zero_action, num_steps=1)

    print("  Running actuated case...")
    state_act, _ = run_steps(env, params, action_actuated, num_steps=1)

    traj_zero = state_zero.trajectory
    traj_act = state_act.trajectory

    n_snap = min(4, len(traj_zero))
    idxs = np.linspace(0, len(traj_zero) - 1, n_snap, dtype=int)

    fig, axes = plt.subplots(n_snap, 2, figsize=(10, 3 * n_snap))
    if n_snap == 1:
        axes = np.array([axes])

    for i, idx in enumerate(idxs):
        z = to_real(traj_zero[idx])
        a = to_real(traj_act[idx])

        im0 = axes[i, 0].imshow(z.T, origin="lower")
        axes[i, 0].set_title(f"Zero action (t={idx})")
        plt.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(a.T, origin="lower")
        axes[i, 1].set_title(f"Controlled case (t={idx})")
        plt.colorbar(im1, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outfile}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Kolmogorov flow JAX environment runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="minimize_tke",
        choices=list(MODE_CONFIGS),
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="DNS timestep (default: 1e-3). Halve to improve fp32 stability.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Floating-point precision (default: float64 — required for JIT stability)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a vorticity comparison PNG (baseline vs selected mode)",
    )
    args = parser.parse_args()

    cfg = MODE_CONFIGS[args.mode]

    print("=== Kolmogorov Flow JAX Environment ===")
    print(f"Mode:  {args.mode}")
    print(f"Dtype: {args.dtype}")
    print(f"dt:    {args.dt if args.dt is not None else '1e-3 (default)'}")
    print(f"Steps: {args.num_steps}")
    print()
    print(cfg["description"])
    print()

    env_config = {}
    if args.dt is not None:
        env_config["dt"] = args.dt
    env = KolmogorovFlow(env_config=env_config, flow_config={})
    params = env.default_params.replace(reward_alpha=cfg["reward_alpha"])

    action = cfg["action"]
    if action is None:
        action = jnp.zeros((params.action_dim,))

    if args.plot:
        print("Generating vorticity comparison plot...")
        outfile = f"kolmogorov_{args.mode}.png"
        save_plot(env, params, action, outfile=outfile)
        print()

    print(f"{'Step':>5}  {'mean_TKE':>12}  {'reward':>12}")
    print("-" * 35)
    _, rows = run_steps(env, params, action, args.num_steps)
    for step, tke, reward in rows:
        print(f"{step:>5}  {tke:>12.4f}  {reward:>12.4f}")


if __name__ == "__main__":
    main()
