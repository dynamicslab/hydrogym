###############################################################

# The PPO implementation below is adapted from PureJaxRL [1] (Copyright 2023 Chris Lu, Apache License 2.0; a copy of
# the license is in LICENSE-purejaxrl next to this file). Changes: the HydroGym JAX environments, the command-line
# options, the value-clipping switch, saving of the trained parameters and the plots.
# [1] https://github.com/luchris429/purejaxrl/

###############################################################

import argparse
import os
import pickle
import sys
import time
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hydrogym.jax.env_core import ClipAction, LogWrapper, NormalizeVecObservation, NormalizeVecReward, VecEnv

# Getting-started settings, the defaults of this script: much shorter episodes than the environments' defaults and, for
# Kolmogorov, a shorter action window with a larger solver time step (200 instead of 10,000 solver steps per action), so
# that a run takes about 10-20 min on a laptop CPU and episodes end (their returns are the learning curve). The
# environments' own settings (Kolmogorov: episodes of 1000 actions of 10 time units at dt 1e-3; channel: episodes of 100
# actions of 50 DNS steps) need a GPU and far more steps; --env-defaults selects them.
GETTING_STARTED = {
    "kolmogorov": {
        "episode_steps": 20,
        "action_time": 1,
        "dt": 5e-3,
        "total_timesteps": 4000,
        "num_envs": 4,
        "num_steps": 20,
        "num_minibatches": 4,
        "lr": 1e-3,
    },
    "channel": {
        "episode_steps": 10,
        "total_timesteps": 120,
        "num_envs": 2,
        "num_steps": 10,
        "num_minibatches": 2,
        "lr": 1e-3,
    },
}


def make_env(config):
    """Instantiate the environment selected by config["ENV_NAME"].

    Optional config entries: ``EPISODE_STEPS`` (actions per episode), ``ACTION_TIME`` (Kolmogorov: time units per
    action, a whole number >= 1), ``DT`` (Kolmogorov: solver time step) and ``REWARD_ALPHA`` (Kolmogorov: weight of
    the turbulent kinetic energy in the reward); None or missing keeps the environment's default.
    """
    env_name = config.get("ENV_NAME", "kolmogorov").lower()
    episode_steps = config.get("EPISODE_STEPS")
    if env_name == "kolmogorov":
        from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

        overrides = {
            "max_episode_steps": episode_steps,
            "action_time": config.get("ACTION_TIME"),
            "dt": config.get("DT"),
            "reward_alpha": config.get("REWARD_ALPHA"),
        }
        overrides = {key: value for key, value in overrides.items() if value is not None}
        env = KolmogorovFlow(env_config={"params": overrides}, flow_config={})
        env_params = env.default_params
    elif env_name == "channel":
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        env = ChannelFlowSpectralEnv(env_config={})
        env_params = env.default_params
        if episode_steps is not None:
            # max_steps_in_episode counts DNS steps; every action advances the flow by nsteps DNS steps
            env_params = env_params.replace(max_steps_in_episode=episode_steps * env_params.nsteps)
    else:
        raise ValueError(f"Unknown ENV_NAME: {env_name!r}. Choose 'kolmogorov' or 'channel'.")
    return env, env_params


def save_model(params, filepath):
    with open(filepath, "wb") as f:
        # Using pickle to serialize params
        pickle.dump(flax.serialization.to_bytes(params), f)


def load_model(filepath):
    with open(filepath, "rb") as f:
        # Deserialize params using pickle
        params_bytes = pickle.load(f)
        params = flax.serialization.from_bytes(None, params_bytes)
    return params


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    # log standard deviation of the Gaussian policy at initialisation (trainable); -1 gives std 0.37
    initial_log_std: float = 0.0

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.constant(self.initial_log_std), (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def rollout(env, params, env_params, num_steps=10, num_envs=4, activation="tanh"):
    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    observations = []
    actions = []
    rewards = []
    dones = []

    # Wrap before reset so the wrapped env is used throughout
    env = ClipAction(env)

    obs, env_state = env.reset(reset_rng, env_params)

    network = ActorCritic(env.action_space(env_params).shape[0], activation=activation)

    for _ in range(num_steps):
        observations.append(obs)

        rng, action_rng = jax.random.split(rng)
        pi, _ = network.apply(params, obs)
        action = pi.sample(seed=action_rng)
        actions.append(action)

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, _ = env.step(step_rng, env_state, action, env_params)
        rewards.append(reward)
        dones.append(done)

    return {
        "observations": jnp.array(observations),
        "actions": jnp.array(actions),
        "rewards": jnp.array(rewards),
        "dones": jnp.array(dones),
    }


def make_train(config):
    total_batch = config["NUM_ENVS"] * config["NUM_STEPS"]
    if total_batch % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            f"NUM_ENVS * NUM_STEPS ({config['NUM_ENVS']} * {config['NUM_STEPS']} = {total_batch}) "
            f"must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}). "
            f"Valid NUM_MINIBATCHES values for your settings: "
            f"{[d for d in range(1, total_batch + 1) if total_batch % d == 0]}"
        )
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = total_batch // config["NUM_MINIBATCHES"]
    env, env_params = make_env(config)
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)

    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            initial_log_std=config.get("INITIAL_LOG_STD", 0.0),
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_losses = jnp.square(value - targets)
                        if config.get("CLIP_VALUE", True):
                            # clipped value loss of purejaxrl: the value may move at most CLIP_EPS per update. This
                            # slows the critic down when the returns are much larger than CLIP_EPS.
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"]
                            )
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_losses = jnp.maximum(value_losses, value_losses_clipped)
                        value_loss = 0.5 * value_losses.mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            # Log the per-step rewards next to the episode statistics of LogWrapper: the rewards exist from the first
            # update on, the episode returns only once an episode has ended
            metric = {**traj_batch.info, "reward": traj_batch.reward}
            rng = update_state[-1]

            if config.get("DEBUG"):

                def callback(info):
                    # info["timestep"] counts the steps of one environment; TOTAL_TIMESTEPS counts all environments
                    step = int(info["timestep"].max()) * config["NUM_ENVS"]
                    total = config["TOTAL_TIMESTEPS"]
                    pct = 100.0 * step / total

                    # Extra env-specific metrics
                    extras = [f"mean_reward={float(info['reward'].mean()):.4f}"]
                    if "mean_tke" in info:
                        extras.append(f"mean_tke={float(info['mean_tke'].mean()):.4f}")

                    # Completed episodes in this rollout batch
                    done_mask = info["returned_episode"]
                    if done_mask.any():
                        mean_return = float(info["returned_episode_returns"][done_mask].mean())
                        extras.append(f"return={mean_return:.4f}")

                    extra_str = "  " + "  ".join(extras)
                    print(f"  step {step:>6}/{total}  ({pct:5.1f}%){extra_str}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def set_publication_style(scale=1.0):
    """Matplotlib settings of the HydroGym paper figures (as in examples/maia/released_policies/plot_policy.py)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.linewidth": 1.0 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 4 * scale,
            "xtick.minor.size": 2 * scale,
            "ytick.major.size": 4 * scale,
            "ytick.minor.size": 2 * scale,
            "xtick.major.width": 0.8 * scale,
            "xtick.minor.width": 0.6 * scale,
            "ytick.major.width": 0.8 * scale,
            "ytick.minor.width": 0.6 * scale,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "axes.formatter.use_mathtext": True,
            "mathtext.default": "regular",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


PPO_COLOR = "#1f77b4"  # colour of PPO in the HydroGym figures


def plot_training(metrics, config, path, title):
    """Training curves: mean reward per step for each PPO update (top) and the returns of the episodes that ended in
    each update (bottom; every completed episode as a dot, their mean as a line).

    ``metrics`` holds arrays of shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS). ``returned_episode_returns`` is 0 until an
    episode has ended, so the returns are masked with ``returned_episode``.
    """
    set_publication_style(0.5)
    plt.rcParams.update({"font.size": 8})
    updates = np.arange(1, config["NUM_UPDATES"] + 1)
    mean_reward = metrics["reward"].mean(axis=(1, 2))
    done = np.asarray(metrics["returned_episode"], dtype=bool)
    num_done = done.sum(axis=(1, 2))
    has_done = num_done > 0

    fig, (ax_reward, ax_return) = plt.subplots(2, 1, sharex=True, figsize=(3.6, 3.6))
    ax_reward.plot(updates, mean_reward, color="0.6", linewidth=0.6, marker="o", markersize=1.5, label="per update")
    # running mean over a tenth of the updates, at least 3 (the first window - 1 updates average over the updates so
    # far); none for fewer than 6 updates
    window = max(3, len(updates) // 10) if len(updates) >= 6 else 1
    if window > 1:
        running = np.convolve(mean_reward, np.ones(window), "full")[: len(updates)] / np.minimum(updates, window)
        ax_reward.plot(updates, running, color=PPO_COLOR, linewidth=1.0, label=f"running mean ({window} updates)")
    ax_reward.legend(loc="lower right", fontsize=7, handlelength=1.5)
    ax_reward.set_ylabel("Mean reward per step")
    if has_done.any():
        update_of_episode = np.broadcast_to(updates[:, None, None], done.shape)[done]
        ax_return.plot(
            update_of_episode,
            metrics["returned_episode_returns"][done],
            linestyle="none",
            marker="o",
            markersize=1.5,
            color="0.6",
            label="episode",
        )
        mean_return = (metrics["returned_episode_returns"] * done).sum(axis=(1, 2))[has_done] / num_done[has_done]
        ax_return.plot(updates[has_done], mean_return, color=PPO_COLOR, linewidth=0.8, label="mean per update")
        ax_return.legend(loc="lower right", fontsize=7, handlelength=1.5)
    else:
        message = (
            f"No episode ended in {config['NUM_UPDATES'] * config['NUM_STEPS']} steps per environment\n"
            "(increase --total-timesteps or reduce --episode-steps)"
        )
        print(message.replace("\n", " "))
        ax_return.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, transform=ax_return.transAxes)
        ax_return.set_yticks([])
    ax_return.set_ylabel("Episode return")
    ax_return.set_xlabel("PPO update")
    for ax in (ax_reward, ax_return):
        ax.minorticks_on()
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.set_axisbelow(True)
    ax_reward.set_title(title, fontsize=6.5, loc="left")
    fig.align_ylabels()
    fig.tight_layout(pad=0.3)
    fig.savefig(path, format="png")
    plt.close(fig)
    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO training for HydroGym JAX environments. The defaults are reduced getting-started settings "
        "(short episodes, a short Kolmogorov action window) that run in about 10-20 min on a laptop CPU; "
        "--env-defaults uses the environments' own episode length and action window (GPU recommended).",
    )
    parser.add_argument(
        "--env",
        default="kolmogorov",
        choices=["kolmogorov", "channel"],
        help="Environment to train on (default: kolmogorov)",
    )
    parser.add_argument(
        "--env-defaults",
        action="store_true",
        help="Use the environment's own episode length and action window (Kolmogorov: 1000 actions of 10 time "
        "units; channel: 100 actions of 50 DNS steps) instead of the getting-started ones",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Actions per episode (getting started: kolmogorov 20, channel 10)",
    )
    parser.add_argument(
        "--action-time",
        type=int,
        default=None,
        help="Kolmogorov only: time units per action, a whole number >= 1 (getting started: 1; environment: 10)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Kolmogorov only: solver time step (getting started: 5e-3; environment: 1e-3)",
    )
    parser.add_argument(
        "--reward-alpha",
        type=float,
        default=1.0,
        help="Kolmogorov only: weight of the turbulent kinetic energy in the reward -(alpha * TKE + sum |a|) "
        "(default 1: suppress TKE; 0, the environment's default, leaves only the action penalty)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total environment steps, all environments together (getting started: kolmogorov 4000, channel 120)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None, help="Parallel environments (getting started: kolmogorov 4, channel 2)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Rollout length per environment and update (getting started: kolmogorov 20, channel 10)",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=None,
        help="Must divide NUM_ENVS * NUM_STEPS (getting started: kolmogorov 4, channel 2)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate (getting started: 1e-3)")
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=-1.0,
        help="Initial log standard deviation of the policy, trained afterwards (default -1, std 0.37; "
        "0 gives std 1, whose exploration noise dominates the Kolmogorov action penalty)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Discount factor (default 0.5: a short horizon, so that the advantages of the few getting-started "
        "samples are dominated by the immediate actuation cost and TKE change rather than by the chaotic flow "
        "further ahead; use 0.99 for long runs)",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.9, help="GAE lambda (default 0.9)")
    parser.add_argument(
        "--clip-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip the value-function update to CLIP_EPS = 0.2 as in purejaxrl (default: on)",
    )
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus coefficient (default 0)")
    parser.add_argument("--model-save-path", default=None, help="Path to save trained model (.pkl)")
    parser.add_argument("--plot-path", default=None, help="Path to save reward plot (.png)")
    args = parser.parse_args()

    # Unset options take the getting-started values; --env-defaults keeps the environment's episode length and
    # action window unless they are given explicitly
    defaults = GETTING_STARTED[args.env]
    env_window_keys = ("episode_steps", "action_time", "dt")
    for key, value in defaults.items():
        if getattr(args, key, None) is None and not (args.env_defaults and key in env_window_keys):
            setattr(args, key, value)

    model_save_path = args.model_save_path or f"trained_model_{args.env}.pkl"
    plot_path = args.plot_path or f"plot_reward_{args.env}.png"

    config = {
        "ENV_NAME": args.env,
        "EPISODE_STEPS": args.episode_steps,
        "ACTION_TIME": args.action_time if args.env == "kolmogorov" else None,
        "DT": args.dt if args.env == "kolmogorov" else None,
        "REWARD_ALPHA": args.reward_alpha if args.env == "kolmogorov" else None,
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": 0.2,
        "CLIP_VALUE": args.clip_value,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "INITIAL_LOG_STD": args.initial_log_std,
        "ANNEAL_LR": False,
        # The channel runs in float64 (x64) mode and NormalizeVecReward would return float64 rewards, which the
        # float32 GAE scan does not accept; keep False for the channel
        "NORMALIZE_ENV": False,
        "DEBUG": True,
        "MODEL_SAVE_PATH": model_save_path,
        "PLOT_TRAINING_PATH": plot_path,
    }
    if args.action_time is not None and args.action_time < 1:
        raise ValueError("--action-time must be at least 1 (time unit)")

    env_probe, env_params_probe = make_env(config)
    if args.env == "kolmogorov":
        episode = env_params_probe.max_episode_steps
        solver_steps = round(env_params_probe.action_time / env_params_probe.dt)
        window = (
            f"{float(env_params_probe.action_time):g} time unit(s) = {solver_steps} solver steps "
            f"of {env_params_probe.dt:g}"
        )
    else:
        episode = env_params_probe.max_steps_in_episode // env_params_probe.nsteps
        window = f"{env_params_probe.nsteps} DNS steps"
    del env_probe

    print(f"=== PPO Training: {args.env} environment ===")
    print(f"  JAX devices     : {jax.devices()}")
    print(f"  Episode length  : {episode} actions of {window}")
    print(f"  Total timesteps : {config['TOTAL_TIMESTEPS']}")
    print(f"  Num envs        : {config['NUM_ENVS']}")
    print(f"  Num steps       : {config['NUM_STEPS']}")
    print(f"  Learning rate   : {config['LR']}")
    print(
        f"  PPO             : gamma {config['GAMMA']}, GAE lambda {config['GAE_LAMBDA']}, "
        f"initial log std {config['INITIAL_LOG_STD']}, entropy coef {config['ENT_COEF']}, "
        f"value clipping {'on' if config['CLIP_VALUE'] else 'off'}"
    )
    print(f"  Model save path : {model_save_path}")
    print(f"  Plot save path  : {plot_path}")
    print("")

    t_start = time.time()
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    jax.block_until_ready(out)
    print(f"Training took {time.time() - t_start:.0f} s")

    # after training
    trained_params = out["runner_state"][0].params
    save_model(trained_params, config["MODEL_SAVE_PATH"])

    # Training curves; the metrics have the shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS)
    metrics = jax.device_get(out["metrics"])
    title = (
        f"PPO, {args.env}: {config['NUM_ENVS']} envs x {config['NUM_STEPS']} steps per update,\n"
        f"episodes of {episode} actions of {window}"
    )
    mean_reward = plot_training(metrics, config, plot_path, title)
    np.save("rewardovertime", mean_reward)
