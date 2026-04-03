import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#####################################################
# Import channel flow and environment configuration #
#####################################################

from hydrogym.jax.envs.channel import *

env_config = {}
env = ChannelFlowSpectralEnv(env_config)
params = env.default_params

#####################################################
#   Reset environment to initial conditions         #
# provided in the HuggingFace initial fields folder #
#####################################################

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)
print("Initial state shape U:", state.U.shape)
print("Initial state shape V:", state.U.shape)
print("Initial state shape W:", state.U.shape)
print("Initial mean observation value: ", jnp.mean(obs))

#####################################################
#         Run environment step                      #
#        actual RK steps = params.nsteps            #
#####################################################

action = jnp.zeros((params.action_dim,))

obs, state, reward, done, info = env.step_env(key, state, action, params)

print("Mean observation value: ", jnp.mean(obs))
print("Reward:", reward)

#####################################################
#         Run environment steps                     #
#             with actuation                        #
#####################################################

# Testing full suction jets #
key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

action = 0.01 * jnp.ones((params.action_dim,))
num_steps = 5

for i in range(num_steps):
    obs, state, reward, done, info = env.step_env(key, state, action, params)
    print("Mean observation value after environment step: ", jnp.mean(obs))
    print("Reward:", reward)
