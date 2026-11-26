"""
PPO baseline using the Kolmogorov environment with an objective of reducing energy dissipation in the extreme
event regime.

Environment: HydroGym's ``KolmogorovFlow.extreme_events_re40()`` preset (Re 40, k = 4, 64 x 64 grid with the periodic
end point, 4 time units per action, actions in [-0.4, 0.15], 100 actions per episode, reward
-(5 * mean dissipation + 2 * sum a_i**2)).
Run: python ppo.py [--seed 30] [--out-dir runs] [--num-updates 100]

Options:
--resume / --max-minutes / --ckpt-every: checkpoint the train state, RNG, flow states, observations, history and the
update counter after every ``ckpt_every`` updates (``<out-dir>/ppo_ckpt.pkl``); ``--resume`` continues bit for bit
from it; ``--max-minutes`` stops cleanly (exit code 3) once that wall-clock time is used. ``<out-dir>/ppo_DONE`` marks a
finished run. Every history record also carries ``wall_s``, the wall-clock time of the update.
--adv-norm {minibatch,rollout}: advantage normalisation; ``minibatch`` (default) uses the minibatch mean / std,
``rollout`` the mean / std over the whole rollout (as in gppo.py).
"""

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

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
    adv_norm: str = "minibatch"  # 'minibatch' or 'rollout' (as in gppo.py)
    resume: bool = False
    ckpt_every: int = 1
    max_minutes: float = 0.0  # 0: no limit


CONFIG = PPOConfig()


def save_checkpoint(path, update, rng, train_state, obs, env_states, history):
    """Atomically write everything needed to continue training after ``update`` completed updates."""
    import flax.serialization

    ckpt = {
        "update": int(update),
        "rng": np.asarray(rng),
        "train_state": flax.serialization.to_bytes(train_state),
        "obs": np.asarray(obs),
        "env_states": jax.tree_util.tree_map(np.asarray, env_states),
        "history": list(history),
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, train_state, env_states):
    """Inverse of :func:`save_checkpoint`; ``train_state`` / ``env_states`` give the structure."""
    import flax.serialization

    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    train_state = flax.serialization.from_bytes(train_state, ckpt["train_state"])
    env_states = jax.tree_util.tree_map(
        lambda tmpl, x: jnp.asarray(x, dtype=tmpl.dtype), env_states, ckpt["env_states"]
    )
    return (
        ckpt["update"],
        jnp.asarray(ckpt["rng"], dtype=jnp.uint32),
        train_state,
        jnp.asarray(ckpt["obs"]),
        env_states,
        ckpt["history"],
    )


class ActorCritic(nn.Module):
    action_dim: int = 4
    hidden_dim: int = 256
    initial_log_std: float = -1.0

    @nn.compact
    def __call__(self, obs):
        def trunk(x):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
            return nn.tanh(nn.Dense(self.hidden_dim)(x))

        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(trunk(obs))
        log_std = self.param("log_std", nn.initializers.constant(self.initial_log_std), (self.action_dim,))
        value = nn.Dense(1)(trunk(obs))[..., 0]
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), value


def bounded(latent, params):
    """Bounding actions to an allowed range."""
    midpoint = (params.max_action + params.min_action) / 2
    halfwidth = (params.max_action - params.min_action) / 2
    return midpoint + halfwidth * jnp.tanh(latent)


class Transition(NamedTuple):
    obs: jax.Array
    latent: jax.Array
    old_logp: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array
    dissipation: jax.Array
    control_cost: jax.Array


def train(config: PPOConfig = CONFIG, env=None):
    """Train; returns the final params, or None if ``max_minutes`` stopped the run first (checkpoint written).

    ``env`` (default: the extreme-event preset) exists only for the resume test, which uses a cheap variant.
    """
    t_start = time.time()
    assert config.adv_norm in ("minibatch", "rollout"), config.adv_norm
    updates, seed, out_dir = config.num_updates, config.seed, config.out_dir
    num_envs, num_steps = config.num_envs, config.num_steps
    epochs, minibatches = config.epochs, config.minibatches
    assert num_envs * num_steps % minibatches == 0
    env = KolmogorovFlow.extreme_events_re40() if env is None else env
    ep = env.default_params
    net = ActorCritic(action_dim=ep.action_dim, hidden_dim=config.hidden_dim, initial_log_std=config.initial_log_std)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    params = net.init(init_key, jnp.zeros((ep.obs_dim,)))
    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )

    reset_batch = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_batch = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    rng, rk = jax.random.split(rng)
    obs, env_states = reset_batch(jax.random.split(rk, num_envs), ep)

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
                    logp = pi.log_prob(tr.latent)
                    ratio = jnp.exp(logp - tr.old_logp)
                    if config.adv_norm == "rollout":
                        normalized = (gae - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
                    else:
                        normalized = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor = -jnp.minimum(
                        ratio * normalized, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * normalized
                    ).mean()

                    clipped_value = tr.old_value + jnp.clip(value - tr.old_value, -config.clip_eps, config.clip_eps)
                    critic = 0.5 * jnp.maximum((value - target) ** 2, (clipped_value - target) ** 2).mean()
                    return actor + config.vf_coef * critic

                loss, grads = jax.value_and_grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), loss

            ts, losses = jax.lax.scan(minibatch_step, ts, (b, aa, tt))
            return (ts, key), losses.mean()

        (train_state, _), losses = jax.lax.scan(epoch, (train_state, key), None, length=epochs)
        return train_state, losses.mean()

    update_jit = jax.jit(update_fn)
    history = []
    start = 0
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    prefix = "ppo"
    ckpt_path = path / f"{prefix}_ckpt.pkl"
    if config.resume and ckpt_path.exists():
        start, rng, state, obs, env_states, history = load_checkpoint(ckpt_path, state, env_states)
        print(f"PPO resumed from {ckpt_path} after update {start}", flush=True)

    for update in range(start, updates):
        t_update = time.time()
        transitions = []

        for _ in range(num_steps):
            rng, sample_key, step_key = jax.random.split(rng, 3)
            pi, value = net.apply(state.params, obs)
            latent = pi.sample(seed=sample_key)
            action = bounded(latent, ep)
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
                    info["mean_dissipation"],
                    info["control_cost"],
                )
            )
            # Auto-reset finished episodes, preserving terminal flags for GAE.
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

        def gae_step(carry, transition):
            gae, next_value = carry
            delta = transition.reward + config.gamma * (1 - transition.done) * next_value - transition.old_value
            gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
            return (gae, transition.old_value), gae

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
            "loss": float(loss),
        }
        record["wall_s"] = time.time() - t_update
        history.append(record)
        print("PPO", record, flush=True)
        last = update + 1 == updates
        out_of_time = config.max_minutes > 0 and (time.time() - t_start) / 60 > config.max_minutes
        if (update + 1) % config.ckpt_every == 0 or last or out_of_time:
            save_checkpoint(ckpt_path, update + 1, rng, state, obs, env_states, history)
        if out_of_time and not last:
            print(f"PPO stopped after update {update + 1} (max_minutes {config.max_minutes})", flush=True)
            return None

    with open(path / f"{prefix}_params.pkl", "wb") as f:
        import flax.serialization

        pickle.dump(flax.serialization.to_bytes(state.params), f)
    np.savez(path / f"{prefix}_metrics.npz", **{k: np.asarray([r[k] for r in history]) for k in history[0]})
    (path / f"{prefix}_DONE").write_text(f"{updates}\n")
    return state.params


def _cli():
    import argparse
    import dataclasses

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    ap.add_argument("--out-dir", default=CONFIG.out_dir)
    ap.add_argument("--num-updates", type=int, default=CONFIG.num_updates)
    ap.add_argument("--adv-norm", choices=("minibatch", "rollout"), default=CONFIG.adv_norm)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=CONFIG.ckpt_every)
    ap.add_argument("--max-minutes", type=float, default=CONFIG.max_minutes)
    a = ap.parse_args()
    return dataclasses.replace(
        CONFIG,
        seed=a.seed,
        out_dir=a.out_dir,
        num_updates=a.num_updates,
        adv_norm=a.adv_norm,
        resume=a.resume,
        ckpt_every=a.ckpt_every,
        max_minutes=a.max_minutes,
    )


if __name__ == "__main__":
    sys.exit(0 if train(_cli()) is not None else 3)
