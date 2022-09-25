import gym
import ppo

n_hidden = 64
n_layers = 2
gamma = 0.99
seed = 42
steps = 4000
epochs = 50

ppo.ppo(
    lambda: gym.make("Pendulum-v0"),
    actor_critic=ppo.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[n_hidden] * n_layers),
    gamma=gamma,
    seed=seed,
    steps_per_epoch=steps,
    epochs=epochs,
)
