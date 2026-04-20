---
sidebar_position: 4
---

# PPO Training

Pure-JAX PPO training for the Kolmogorov and turbulent channel environments, based on [purejaxrl](https://github.com/luchris429/purejaxrl/) with HydroGym integrations (`VecEnv`, normalization wrappers, etc.).

## Setting up the JAX Environment

We begin by setting up the JAX environment with all required software dependencies:

```python
import argparse
import pickle
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
```

From HydroGym, we will need first need the functions to wrap the environment in a `VecEnv` and normalize the observations and rewards.

```python
from hydrogym.jax.env_core import ClipAction, LogWrapper, NormalizeVecObservation, NormalizeVecReward, VecEnv
```

## Constructing the Reinforcement Learning Environment

To be able to construct the reinforcement learning environment, we then need to construct an utility function which takes in the environment configuration, and validated its configuration for the chosen case.

```python
def make_env(config):
    """Instantiate the environment selected by config["ENV_NAME"]."""
    env_name = config.get("ENV_NAME", "kolmogorov").lower()
    if env_name == "kolmogorov":
        from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

        env = KolmogorovFlow(env_config={}, flow_config={})
    elif env_name == "channel":
        from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

        env = ChannelFlowSpectralEnv(env_config={})
    else:
        raise ValueError(f"Unknown ENV_NAME: {env_name!r}. Choose 'kolmogorov' or 'channel'.")
    return env, env.default_params
```

In addition, we require utility functions around the saving and loading of the model

```python
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
```

## Defining Reinforcement Learning Training

For the reinforcement learning training, we will need to first define an Actor-Critic network, before we can move on to define the transition, and then conclude by defining the actual training loop finally.

```python
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

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
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))  # changed actor_mean to jnp.exp

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)
```

The transition class is then defined as follows:

```python
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
```

With the rollout function following the purejaxrl implementation:

```python
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
```

Culminating in the following training loop:

```python
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

    # @partial(jax.jit, static_argnums=(1,))
    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
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
                action = pi.sample(seed=_rng)  # clip action here
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
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

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
            metric = traj_batch.info
            rng = update_state[-1]

            if config.get("DEBUG"):

                def callback(info):
                    step = int(info["timestep"].max())
                    total = config["TOTAL_TIMESTEPS"]
                    pct = 100.0 * step / total

                    # Extra env-specific metrics
                    extras = []
                    if "mean_tke" in info:
                        extras.append(f"mean_tke={float(info['mean_tke'].mean()):.4f}")

                    # Completed episodes in this rollout batch
                    done_mask = info["returned_episode"]
                    if done_mask.any():
                        mean_return = float(info["returned_episode_returns"][done_mask].mean())
                        extras.append(f"return={mean_return:.4f}")

                    extra_str = "  " + "  ".join(extras) if extras else ""
                    print(f"  step {step:>6}/{total}  ({pct:5.1f}%){extra_str}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train
```

## Performing the Training

At this point, we can now define the configuration of our training hyperparameters, and pull the individual pieces together

```python
config = {
    "LR": 1e-4,  # try 3e-4 - 1e-5 (play around with it) 1e-4
    "NUM_ENVS": 4,
    "NUM_STEPS": 40,  # 40
    "TOTAL_TIMESTEPS": 100,  # 4000
    "UPDATE_EPOCHS": 10,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.985,  # can tune to go up to 0.995. 0.98
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,  # can be increased to approx 0.1 or 0.2 or stay the same
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",  # mish activation function is good to try
    "ANNEAL_LR": False,  # can try
    "NORMALIZE_ENV": False,
    "DEBUG": True,
}
```

define our training parameters more custom to HydroGym

```python
parser = argparse.ArgumentParser(description="PPO training for HydroGym JAX environments")
parser.add_argument(
    "--env",
    default="kolmogorov",
    choices=["kolmogorov", "channel"],
    help="Environment to train on (default: kolmogorov)",
)
parser.add_argument("--total-timesteps", type=int, default=4000)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-steps", type=int, default=10)
parser.add_argument("--num-minibatches", type=int, default=8, help="Must divide NUM_ENVS * NUM_STEPS (default: 8)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--model-save-path", default=None, help="Path to save trained model (.pkl)")
parser.add_argument("--plot-path", default=None, help="Path to save reward plot (.png)")
args = parser.parse_args()
```

set the paths for the model to be saved, and where plots are to be saved

```python
model_save_path = args.model_save_path or f"trained_model_{args.env}.pkl"
plot_path = args.plot_path or f"plot_reward_{args.env}.png"
```

just for our own sanity, inspect the configuration and paths to be sure that they are set correctly before beginning the training.

```python
print(f"=== PPO Training: {args.env} environment ===")
print(f"  Total timesteps : {config['TOTAL_TIMESTEPS']}")
print(f"  Num envs        : {config['NUM_ENVS']}")
print(f"  Num steps       : {config['NUM_STEPS']}")
print(f"  Learning rate   : {config['LR']}")
print(f"  Model save path : {model_save_path}")
print(f"  Plot save path  : {plot_path}")
print("")
```

at which point we can run the full training

```python
rng = jax.random.PRNGKey(30)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)
```

After the training is completed, we can save the trained model

```python
trained_params = out["runner_state"][0].params
save_model(trained_params, config["MODEL_SAVE_PATH"])
```

and plot the training results:

```python
plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
plt.xlabel("Updates")
plt.ylabel("Return")
plt.show()
plt.savefig(config["PLOT_TRAINING_PATH"], format="png")
jnp.save("rewardovertime", out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
```

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
