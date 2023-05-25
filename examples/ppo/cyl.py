import gymnasium as gym
import ppo

import hydrogym

# from firedrake import logging

# logging.set_log_level(logging.DEBUG)


class CylEnv(hydrogym.FlowEnv):
    def __init__(self, env_config):
        config = {
            "flow": hydrogym.firedrake.Cylinder,
            "flow_config": env_config["flow"],
            "solver": hydrogym.firedrake.IPCS,
            "solver_config": env_config["solver"],
        }
        super().__init__(config)


gym.register(id="Cylinder-v0", entry_point=CylEnv)

# Set up the printing callback
log = hydrogym.firedrake.utils.io.LogCallback(
    postprocess=lambda flow: flow.get_observations(),
    nvals=2,
    interval=10,
    print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
    filename=None,
)

env_config = {
    "flow": {
        "Re": 100,
        "mesh": "coarse",
        "restart": "../demo/checkpoint-coarse.h5",
    },
    "solver": {
        "dt": 1e-3,
        "callbacks": [log],
    },
}

n_hidden = 64
n_layers = 2
gamma = 0.99
seed = 42
steps = 100
epochs = 1000

ppo.ppo(
    lambda: gym.make("Cylinder-v0", env_config=env_config),
    actor_critic=ppo.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[n_hidden] * n_layers),
    gamma=gamma,
    seed=seed,
    steps_per_epoch=steps,
    epochs=epochs,
)
