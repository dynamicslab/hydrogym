# test_kolmogorov_save_png.py

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


def to_real(omega_hat):
    return np.asarray(jnp.fft.irfftn(omega_hat))


def run_env(env, params, action, steps=50):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    for _ in range(steps):
        key, subkey = jax.random.split(key)
        obs, state, reward, done, _ = env.step_env(subkey, state, action, params)

    return state.trajectory


def main():
    env = KolmogorovFlow(env_config={}, flow_config={})
    params = env.default_params

    params = params.replace(
        action_time=1.0,
        save_time=0.1,
        max_episode_steps=5,
    )

    zero_action = jnp.zeros((params.action_dim,))
    test_action = jnp.array([-0.15, -0.03, 0.02, 0.01])

    print("Running zero-action...")
    traj_zero = run_env(env, params, zero_action)

    print("Running controlled...")
    traj_act = run_env(env, params, test_action)

    # pick a few snapshots
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
    outfile = "kolmogorov_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    
if __name__ == "__main__":
    main()