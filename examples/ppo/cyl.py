import gym
import ppo

import hydrogym

# from firedrake import logging

# logging.set_log_level(logging.DEBUG)

gym.register(id="Cylinder-v0", entry_point="hydrogym.env:CylEnv")

# Set up the printing callback
log = hydrogym.io.LogCallback(
    postprocess=lambda flow: flow.get_observations(),
    nvals=2,
    interval=10,
    print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
    filename=None,
)

env_config = {
    "Re": 100,
    "mesh": "coarse",
    "callbacks": [log],
    "checkpoint": "../cylinder/demo/checkpoint-coarse.h5",
}

n_hidden = 64
n_layers = 2
gamma = 0.99
seed = 42
steps = 1000
epochs = 50

ppo.ppo(
    lambda: gym.make("Cylinder-v0", env_config=env_config),
    actor_critic=ppo.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[n_hidden] * n_layers),
    gamma=gamma,
    seed=seed,
    steps_per_epoch=steps,
    epochs=epochs,
)
