"""
GRADIENT-ENHANCED PPO SCRIPT

Essentially, PPO is modified to include an analytical loss term, which encourages the policy to move in the
direction of increasing reward. The analytical gradient term is included in the loss.

The mathematical summary of GPPO is described below:

    The analytical gradient is computed:
    grad_e = ∇_a e(s, a), and normalized to unit length g = grad_e / ||grad_e||2.

    Encourage the policy to move in the direction of decreasing dissipation:
    L_physics = g · a

    Combine with the standard PPO objective:
    L_GPPO = L_PPO + β L_physics

Environment: HydroGym's ``KolmogorovFlow.extreme_events_re40()`` preset (Re 40, k = 4, 64 x 64 grid with the periodic
end point, 4 time units per action, actions in [-0.4, 0.15], 100 actions per episode, reward
-(5 * mean dissipation + 2 * sum a_i**2)).
Run: python gppo.py [--seed 30] [--out-dir runs] [--num-updates 100]

Options:
--resume / --max-minutes / --ckpt-every: as in ppo.py (``<out-dir>/gppo_ckpt.pkl``, ``<out-dir>/gppo_DONE``).
--beta: weight of the physics term (default 2.0); ``--beta 0`` is the control run with the physics term off.
Every history record also carries ``wall_s`` (wall-clock time of the update) and ``grad_wall_s`` (of which the
forward-mode physics gradient, timed to completion).
"""

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ppo import ActorCritic, bounded, load_checkpoint, save_checkpoint

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 30
    out_dir: str = "runs"
    num_updates: int = 100
    num_envs: int = 4
    num_steps: int = 10
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.985
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    hidden_dim: int = 256
    initial_log_std: float = -1.0
    beta: float = 2.0  # GPPO only
    resume: bool = False
    ckpt_every: int = 1
    max_minutes: float = 0.0  # 0: no limit


CONFIG = PPOConfig()


class Transition(NamedTuple):
    obs: jax.Array
    latent: jax.Array
    old_logp: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array
    old_action: jax.Array
    gradient_direction: jax.Array
    guidance_mask: jax.Array
    dissipation: jax.Array
    control_cost: jax.Array


def train(config: PPOConfig = CONFIG, env=None):
    """Train; returns the final params, or None if ``max_minutes`` stopped the run first (checkpoint written).

    ``env`` (default: the extreme-event preset) exists only for the resume test, which uses a cheap variant.
    """
    t_start = time.time()
    updates, seed, out_dir = config.num_updates, config.seed, config.out_dir
    num_envs, num_steps = config.num_envs, config.num_steps
    epochs, minibatches, beta = config.epochs, config.minibatches, config.beta
    assert (num_envs * num_steps) % minibatches == 0
    env = KolmogorovFlow.extreme_events_re40() if env is None else env
    ep = env.default_params
    assert ep.action_dim == 4
    assert abs(float(ep.action_time) - 4.0) < 1e-6
    assert abs(float(ep.max_episode_steps) * float(ep.action_time) - 400.0) < 1e-6  # 100 actions x 4 time units

    net = ActorCritic(action_dim=ep.action_dim, hidden_dim=config.hidden_dim, initial_log_std=config.initial_log_std)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    state = TrainState.create(
        apply_fn=net.apply,
        params=net.init(init_key, jnp.zeros((ep.obs_dim,))),
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )
    reset_batch = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_batch = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    def dissipation_only(s, a):
        # The same four-second trajectory-averaged dissipation as step_env (HydroGym: rollout_objective).
        return env.rollout_objective(s, a, ep)

    #################################
    # PHYSICS GRADIENT: ∇ₐε = ∂ε/∂a #
    #################################
    grad_diss = jax.jit(jax.jacfwd(dissipation_only, argnums=1))

    def update_fn(train_state, batch, advantages, targets, key):
        total = num_steps * num_envs
        flat = jax.tree_util.tree_map(lambda x: x.reshape((total,) + x.shape[2:]), batch)
        adv = advantages.reshape(total)
        tgt = targets.reshape(total)

        def epoch(carry, _):
            ts, key = carry
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, total)
            b = jax.tree_util.tree_map(lambda x: x[perm].reshape((minibatches, -1) + x.shape[1:]), flat)
            aa = adv[perm].reshape(minibatches, -1)
            tt = tgt[perm].reshape(minibatches, -1)

            def minibatch_step(ts, data):
                tr, gae, target = data

                def loss_fn(p):
                    pi, value = net.apply(p, tr.obs)
                    ratio = jnp.exp(pi.log_prob(tr.latent) - tr.old_logp)
                    # Normalize across the full rollout, not individual
                    # minibatches (which can have tiny variance).
                    normalized = (gae - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
                    actor = -jnp.minimum(
                        ratio * normalized, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * normalized
                    ).mean()
                    clipped_value = tr.old_value + jnp.clip(value - tr.old_value, -config.clip_eps, config.clip_eps)
                    critic = (
                        0.5
                        * jnp.maximum(
                            (value - target) ** 2,
                            (clipped_value - target) ** 2,
                        ).mean()
                    )
                    predicted_action = bounded(pi.mean(), ep)
                    # Minimize g dot action, which nudges the actor in the -grad(dissipation) direction #
                    local_dot = jnp.sum(
                        jax.lax.stop_gradient(tr.gradient_direction) * (predicted_action),
                        axis=-1,
                    )
                    # Mask unguided transitions; zero if no guided sample. #
                    physics = jnp.sum(tr.guidance_mask * local_dot) / jnp.maximum(jnp.sum(tr.guidance_mask), 1.0)
                    return actor + config.vf_coef * critic + beta * physics

                loss, grads = jax.value_and_grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), loss

            ts, losses = jax.lax.scan(minibatch_step, ts, (b, aa, tt))
            return (ts, key), losses.mean()

        (train_state, _), losses = jax.lax.scan(epoch, (train_state, key), None, length=epochs)
        return train_state, losses.mean()

    update_jit = jax.jit(update_fn)
    rng, reset_key = jax.random.split(rng)
    obs, env_states = reset_batch(jax.random.split(reset_key, num_envs), ep)
    history = []
    start = 0
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / "gppo_ckpt.pkl"
    if config.resume and ckpt_path.exists():
        start, rng, state, obs, env_states, history = load_checkpoint(ckpt_path, state, env_states)
        print(f"GPPO resumed from {ckpt_path} after update {start}", flush=True)

    for update in range(start, updates):
        t_update = time.time()
        grad_wall = 0.0
        transitions = []
        guided_norm = 0.0
        for t in range(num_steps):
            rng, sample_key, step_key = jax.random.split(rng, 3)
            pi, value = net.apply(state.params, obs)
            latent = pi.sample(seed=sample_key)
            action = bounded(latent, ep)
            mean_action = bounded(pi.mean(), ep)
            grad_direction = jnp.zeros_like(action)
            guidance_mask = jnp.zeros((num_envs,), dtype=jnp.float32)

            if t == update % num_steps:
                selected_env = (update // num_steps) % num_envs
                single_state = jax.tree_util.tree_map(lambda x: x[selected_env], env_states)
                single_action = mean_action[selected_env]
                t_grad = time.time()
                gradient = jax.block_until_ready(grad_diss(single_state, single_action))
                grad_wall = time.time() - t_grad
                gradient_norm = jnp.linalg.norm(gradient)
                unit_gradient = gradient / jnp.maximum(gradient_norm, 1e-12)
                grad_direction = grad_direction.at[selected_env].set(unit_gradient)
                guidance_mask = guidance_mask.at[selected_env].set(1.0)
                guided_norm = float(gradient_norm)

            keys = jax.random.split(step_key, num_envs)
            next_obs, next_states, reward, done, info = step_batch(keys, env_states, action, ep)
            transitions.append(
                Transition(
                    obs,
                    latent,
                    pi.log_prob(latent),
                    value,
                    reward,
                    done,
                    jax.lax.stop_gradient(mean_action),
                    jax.lax.stop_gradient(grad_direction),
                    guidance_mask,
                    info["mean_dissipation"],
                    info["control_cost"],
                )
            )

            # Carry the flow forward; reset only on episode termination.
            rng, reset_key = jax.random.split(rng)
            reset_obs, reset_states = reset_batch(jax.random.split(reset_key, num_envs), ep)
            obs = jnp.where(done[:, None], reset_obs, next_obs)
            env_states = jax.tree_util.tree_map(
                lambda new, reset: jnp.where(done.reshape((-1,) + (1,) * (new.ndim - 1)), reset, new),
                next_states,
                reset_states,
            )

        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)
        _, bootstrap = net.apply(state.params, obs)

        def gae_step(carry, tr):
            gae, next_value = carry
            alive = 1.0 - tr.done.astype(jnp.float32)
            delta = tr.reward + config.gamma * alive * next_value - tr.old_value
            gae = delta + config.gamma * config.gae_lambda * alive * gae
            return (gae, tr.old_value), gae

        _, advantages = jax.lax.scan(gae_step, (jnp.zeros_like(bootstrap), bootstrap), batch, reverse=True)
        targets = advantages + batch.old_value
        rng, update_key = jax.random.split(rng)
        state, loss = update_jit(state, batch, advantages, targets, update_key)
        record = {
            "update": update + 1,
            "env_steps": (update + 1) * num_steps * num_envs,
            "mean_reward": float(batch.reward.mean()),
            "mean_dissipation": float(batch.dissipation.mean()),
            "mean_control_cost": float(batch.control_cost.mean()),
            "gradient_norm": guided_norm,
            "loss": float(loss),
        }
        record["wall_s"] = time.time() - t_update
        record["grad_wall_s"] = grad_wall
        history.append(record)
        print("GPPO", record, flush=True)
        last = update + 1 == updates
        out_of_time = config.max_minutes > 0 and (time.time() - t_start) / 60 > config.max_minutes
        if (update + 1) % config.ckpt_every == 0 or last or out_of_time:
            save_checkpoint(ckpt_path, update + 1, rng, state, obs, env_states, history)
        if out_of_time and not last:
            print(f"GPPO stopped after update {update + 1} (max_minutes {config.max_minutes})", flush=True)
            return None

    with open(path / "gppo_params.pkl", "wb") as f:
        pickle.dump(flax.serialization.to_bytes(state.params), f)
    np.savez(
        path / "gppo_metrics.npz",
        **{k: np.asarray([r[k] for r in history]) for k in history[0]},
    )
    (path / "gppo_DONE").write_text(f"{updates}\n")
    return state.params


def _cli():
    import dataclasses

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    ap.add_argument("--out-dir", default=CONFIG.out_dir)
    ap.add_argument("--num-updates", type=int, default=CONFIG.num_updates)
    ap.add_argument("--beta", type=float, default=CONFIG.beta)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=CONFIG.ckpt_every)
    ap.add_argument("--max-minutes", type=float, default=CONFIG.max_minutes)
    a = ap.parse_args()
    return dataclasses.replace(
        CONFIG,
        seed=a.seed,
        out_dir=a.out_dir,
        num_updates=a.num_updates,
        beta=a.beta,
        resume=a.resume,
        ckpt_every=a.ckpt_every,
        max_minutes=a.max_minutes,
    )


if __name__ == "__main__":
    sys.exit(0 if train(_cli()) is not None else 3)
