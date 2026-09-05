---
sidebar_label: env_core
title: hydrogym.jax.env_core
---

Core Jax Gym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

## EnvParams Objects

```python
class EnvParams(environment.EnvParams)
```

Gymnax environment parameters extended with the environment config dictionary.

**Attributes**:

- `config` - Environment configuration (grid sizes, control bounds, etc.)
  as loaded from the environment&#x27;s configuration file.

## JAXFlowEnv Objects

```python
class JAXFlowEnv(HFEnvConfigMixin, environment.Environment[EnvState,
                                                           EnvParams])
```

Base JAXFlowEnv with Hugging Face Hub integration for configuration management.

This environment provides a Gymnax-compatible interface for CFD simulations
using JAX solvers. It handles:
- Environment data management via Hugging Face Hub
- Configuration file resolution and loading
- Action space configuration

**Attributes**:

- `environment_name` - Name of the CFD environment configuration.
- `env_data_path` - Path to the local environment data directory.
- `cfg` - OmegaConf configuration object.
- `observation_space` - Gymnax observation space.
- `action_space` - Gymnax action space.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the JAXFlowEnv environment.

**Arguments**:

- `env_config` - Configuration dictionary containing:
  - environment_name (str): Required. Name of the environment.
  - hf_repo_id (str): HF repository ID. Default: &#x27;dynamicslab/HydroGym-environments&#x27;
  - local_fallback_dir (str): Optional local fallback directory.
  - use_clean_cache (bool): Whether to use clean cache. Default: True
  - configuration_file (str): Optional path to config file.
  - is_testing (bool): Whether in testing mode. Default: False
  - probe_locations (list): Probe coordinate locations.
  - obs_normalization_strategy (str): One of &#x27;U_inf&#x27;, &#x27;probewise_mean_std&#x27;,
  &#x27;none&#x27;, &#x27;customized&#x27;.
  - obs_loc (list): Required if strategy is &#x27;customized&#x27;.
  - obs_scale (list): Required if strategy is &#x27;customized&#x27;.

**Raises**:

- `ConfigError` - If required configuration is missing or invalid.

#### get\_environment\_files\_info

```python
def get_environment_files_info() -> Dict
```

Get information about where environment files are stored.

Useful for debugging and understanding file locations.

**Returns**:

  Dictionary containing environment name, paths, and file information.

#### create\_from\_hf\_env

```python
@classmethod
def create_from_hf_env(cls,
                       environment_name: str,
                       hf_repo_id: str = "your-username/maiagym-envs",
                       local_fallback_dir: Optional[str] = None,
                       **kwargs)
```

Create environment directly from HF environment name.

**Arguments**:

- `environment_name` - Name of environment (e.g., &#x27;Cylinder_2D_Re200&#x27;).
- `hf_repo_id` - Hugging Face repository ID.
- `local_fallback_dir` - Local fallback directory.
- `**kwargs` - Additional environment configuration parameters.
  

**Returns**:

  Configured MaiaFlowEnv instance.

#### get\_available\_environments

```python
def get_available_environments() -> List[str]
```

Get list of all available environments from HF Hub.

**Returns**:

  List of environment names.

#### update\_environment\_data

```python
def update_environment_data(force_download: bool = False) -> None
```

Update environment data from HF Hub.

**Arguments**:

- `force_download` - Force re-download even if cached.

#### reset\_env

```python
def reset_env(key: chex.PRNGKey,
              params: EnvParams) -> Tuple[chex.Array, EnvState]
```

Reset the environment to its initial condition.

**Arguments**:

- `key` - PRNG key for stochastic resets.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, state)`` of the initial observation and state.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the actual reset.

#### get\_obs

```python
def get_obs(state: EnvState, params: EnvParams, key=None) -> chex.Array
```

Compute the observation from the current environment state.

**Arguments**:

- `state` - Current environment state.
- `params` - Environment parameters.
- `key` - Optional PRNG key for stochastic observations.
  

**Returns**:

  Observation array.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the observation model.

#### is\_terminal

```python
def is_terminal(state: EnvState, params: EnvParams) -> jnp.ndarray
```

Whether the episode has ended.

An episode terminates when the state flags ``terminal`` or the elapsed
time reaches ``params.max_episode_steps``.

**Arguments**:

- `state` - Current environment state.
- `params` - Environment parameters.
  

**Returns**:

  Boolean array indicating termination.

#### step\_env

```python
def step_env(key: chex.PRNGKey, state: EnvState, action: jnp.array,
             params: EnvParams)
```

Advance the environment by one step under the given action.

**Arguments**:

- `key` - PRNG key for stochastic transitions.
- `state` - Current environment state.
- `action` - Control (actuation) input.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, next_state, reward, done, info)``.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the dynamics.

## JAXFlowEnvBase Objects

```python
class JAXFlowEnvBase(environment.Environment[EnvState, EnvParams])
```

Base JAXFlowEnv without Hugging Face Hub integration.
Contains core environment interface methods required by Gymnax.

#### \_\_init\_\_

```python
def __init__(env_config: Optional[dict] = None)
```

Initialize the environment with an optional configuration dictionary.

**Arguments**:

- `env_config` - Environment configuration options (e.g.
  ``max_episode_steps``); defaults to an empty dict.

#### default\_params

```python
def default_params() -> EnvParams
```

Return the environment&#x27;s default parameters.

Sets ``self.max_episode_steps`` from ``env_config`` (default 1000) as a
side effect.

**Returns**:

  Default environment parameters.
  

**Raises**:

- `NotImplementedError` - Always; subclasses must provide the parameters.

#### name

```python
def name() -> str
```

Return the environment&#x27;s name.

**Raises**:

- `NotImplementedError` - Always; subclasses must provide a name.

#### action\_space

```python
def action_space(params: Optional[EnvParams] = None)
```

Return the (Box) action space bounded by the parameters&#x27; action limits.

**Arguments**:

- `params` - Environment parameters; defaults to ``self.default_params``.
  

**Returns**:

  Gymnax Box space of shape ``(params.action_dim,)`` with bounds
  ``[params.min_action, params.max_action]``.

#### observation\_space

```python
def observation_space(params: EnvParams)
```

Return the (Box) observation space bounded by the parameters&#x27; observation limits.

**Arguments**:

- `params` - Environment parameters.
  

**Returns**:

  Gymnax Box space of shape ``(params.obs_dim,)`` with bounds
  ``[params.min_obs, params.max_obs]``.

#### is\_terminal

```python
def is_terminal(state: EnvState, params: EnvParams) -> jnp.ndarray
```

Whether the episode has ended (state flags ``terminal`` or time limit reached).

**Arguments**:

- `state` - Current environment state.
- `params` - Environment parameters.
  

**Returns**:

  Boolean array indicating termination.

#### reset\_env

```python
def reset_env(key: chex.PRNGKey,
              params: EnvParams) -> Tuple[chex.Array, EnvState]
```

Reset the environment to its initial condition.

**Arguments**:

- `key` - PRNG key for stochastic resets.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, state)`` of the initial observation and state.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the actual reset.

#### get\_obs

```python
def get_obs(state: EnvState, params: EnvParams, key=None) -> chex.Array
```

Compute the observation from the current environment state.

**Arguments**:

- `state` - Current environment state.
- `params` - Environment parameters.
- `key` - Optional PRNG key for stochastic observations.
  

**Returns**:

  Observation array.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the observation model.

#### step\_env

```python
def step_env(key: chex.PRNGKey, state: EnvState, action: chex.Array,
             params: EnvParams)
```

Advance the environment by one step under the given action.

**Arguments**:

- `key` - PRNG key for stochastic transitions.
- `state` - Current environment state.
- `action` - Control (actuation) input.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, next_state, reward, done, info)``.
  

**Raises**:

- `NotImplementedError` - Always; subclasses define the dynamics.

## GymnaxWrapper Objects

```python
class GymnaxWrapper(object)
```

Base class for Gymnax wrappers.

#### \_\_init\_\_

```python
def __init__(env)
```

Wrap the given environment.

**Arguments**:

- `env` - The Gymnax environment (or wrapper) being wrapped.

#### \_\_getattr\_\_

```python
def __getattr__(name)
```

Proxy attribute access through to the wrapped environment.

## FlattenObservationWrapper Objects

```python
class FlattenObservationWrapper(GymnaxWrapper)
```

Flatten the observations of the environment.

#### \_\_init\_\_

```python
def __init__(env: environment.Environment)
```

Wrap the given environment (see :class:`GymnaxWrapper`).

#### observation\_space

```python
def observation_space(params) -> spaces.Box
```

Return the wrapped environment&#x27;s observation space, flattened to 1D.

**Arguments**:

- `params` - Environment parameters.
  

**Returns**:

  Gymnax Box space whose shape is the product of the underlying
  space&#x27;s shape, with the same bounds and dtype.
  

**Raises**:

- `AssertionError` - If the wrapped space is not a Box.

#### reset

```python
@partial(jax.jit, static_argnums=(0, ))
def reset(
    key: chex.PRNGKey,
    params: Optional[environment.EnvParams] = None
) -> Tuple[chex.Array, environment.EnvState]
```

Reset the wrapped environment and flatten the initial observation.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Environment parameters (None uses the environment default).
  

**Returns**:

  Tuple ``(obs, state)`` with the flattened observation.

#### step

```python
@partial(jax.jit, static_argnums=(0, ))
def step(
    key: chex.PRNGKey,
    state: environment.EnvState,
    action: Union[int, float],
    params: Optional[environment.EnvParams] = None
) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]
```

Step the wrapped environment and flatten the returned observation.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current environment state.
- `action` - Action to apply.
- `params` - Environment parameters (None uses the environment default).
  

**Returns**:

  Tuple ``(obs, state, reward, done, info)`` with the flattened
  observation; all other values are passed through unchanged.

## LogEnvState Objects

```python
@struct.dataclass
class LogEnvState()
```

State bookkeeping for :class:`LogWrapper`.

**Attributes**:

- `env_state` - The wrapped environment&#x27;s own state.
- `episode_returns` - Return accumulated so far in the current episode.
- `episode_lengths` - Number of steps taken so far in the current episode.
- `returned_episode_returns` - Return of the most recently completed episode.
- `returned_episode_lengths` - Length of the most recently completed episode.
- `timestep` - Total number of steps taken across all episodes.

## LogWrapper Objects

```python
class LogWrapper(GymnaxWrapper)
```

Log the episode returns and lengths.

#### \_\_init\_\_

```python
def __init__(env: environment.Environment)
```

Wrap the given environment (see :class:`GymnaxWrapper`).

#### reset

```python
@partial(jax.jit, static_argnums=(0, ))
def reset(
    key: chex.PRNGKey,
    params: Optional[environment.EnvParams] = None
) -> Tuple[chex.Array, environment.EnvState]
```

Reset the wrapped environment and zero the episode bookkeeping.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Environment parameters (None uses the environment default).
  

**Returns**:

  Tuple ``(obs, LogEnvState)`` with all counters initialized to zero.

#### step

```python
@partial(jax.jit, static_argnums=(0, ))
def step(
    key: chex.PRNGKey,
    state: environment.EnvState,
    action: Union[int, float],
    params: Optional[environment.EnvParams] = None
) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]
```

Step the wrapped environment, updating and exporting episode statistics.

Accumulates the running episode return/length and, on episode end,
freezes them into ``returned_episode_returns``/``returned_episode_lengths``
before resetting the running counters. These values (plus the global
timestep and an ``returned_episode`` done flag) are added to ``info``.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current :class:``0.
- ``1 - Action to apply to the wrapped environment.
- ``2 - Environment parameters (None uses the environment default).
  

**Returns**:

  Tuple ``(obs, new LogEnvState, reward, done, info)`` with the
  logging fields added to ``info``.

## NavixGymnaxWrapper Objects

```python
class NavixGymnaxWrapper()
```

Adapter exposing a Navix environment through the Gymnax-style API.

Wraps a Navix environment created from ``env_name`` and translates its
reset/step/space API into the ``(obs, state, reward, done, info)`` tuple
convention used by the other wrappers here.

#### \_\_init\_\_

```python
def __init__(env_name)
```

Create the underlying Navix environment.

**Arguments**:

- `env_name` - Name of the Navix environment (passed to ``navix.make``).

#### reset

```python
def reset(key, params=None)
```

Reset the Navix environment.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Unused; accepted for Gymnax API compatibility.
  

**Returns**:

  Tuple ``(observation, timestep)`` from the Navix reset.

#### step

```python
def step(key, state, action, params=None)
```

Step the Navix environment.

**Arguments**:

- `key` - Unused; accepted for Gymnax API compatibility.
- `state` - Current Navix timestep (used as the environment state).
- `action` - Action to apply.
- `params` - Unused; accepted for Gymnax API compatibility.
  

**Returns**:

  Tuple ``(observation, timestep, reward, done, info)`` where ``info``
  is always an empty dict.

#### observation\_space

```python
def observation_space(params)
```

Return the flattened Navix observation space as a Gymnax Box.

**Arguments**:

- `params` - Unused; accepted for Gymnax API compatibility.
  

**Returns**:

  Gymnax Box with the Navix space&#x27;s bounds, flattened shape, and dtype.

#### action\_space

```python
def action_space(params)
```

Return the Navix action space as a Gymnax Discrete space.

**Arguments**:

- `params` - Unused; accepted for Gymnax API compatibility.
  

**Returns**:

  Gymnax Discrete space with the number of Navix action categories.

## ClipAction Objects

```python
class ClipAction(GymnaxWrapper)
```

Wrapper that clips actions to a fixed range before stepping the environment.

#### \_\_init\_\_

```python
def __init__(env, low=-1.0, high=1.0)
```

Wrap the environment and store the clipping bounds.

**Arguments**:

- `env` - The environment being wrapped.
- `low` - Lower bound for clipping actions.
- `high` - Upper bound for clipping actions.

#### step

```python
def step(key, state, action, params=None)
```

Clip the action to ``[low, high]`` and step the wrapped environment.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current environment state.
- `action` - Action to clip and apply.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, state, reward, done, info)`` from the wrapped environment.
  

**Notes**:

- `TODO` - In theory the clip bounds should come from the action space.

## TransformObservation Objects

```python
class TransformObservation(GymnaxWrapper)
```

Wrapper that applies a function to the observation after reset and step.

#### \_\_init\_\_

```python
def __init__(env, transform_obs)
```

Wrap the environment and store the observation transform.

**Arguments**:

- `env` - The environment being wrapped.
- `transform_obs` - Callable applied to every observation.

#### reset

```python
def reset(key, params=None)
```

Reset the wrapped environment and transform the initial observation.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(transformed obs, state)``.

#### step

```python
def step(key, state, action, params=None)
```

Step the wrapped environment and transform the returned observation.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current environment state.
- `action` - Action to apply.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(transformed obs, state, reward, done, info)``.

## TransformReward Objects

```python
class TransformReward(GymnaxWrapper)
```

Wrapper that applies a function to the reward after each step.

#### \_\_init\_\_

```python
def __init__(env, transform_reward)
```

Wrap the environment and store the reward transform.

**Arguments**:

- `env` - The environment being wrapped.
- `transform_reward` - Callable applied to every reward.

#### step

```python
def step(key, state, action, params=None)
```

Step the wrapped environment and transform the returned reward.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current environment state.
- `action` - Action to apply.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, state, transformed reward, done, info)``.

## VecEnv Objects

```python
class VecEnv(GymnaxWrapper)
```

Wrapper that vectorizes the wrapped environment&#x27;s reset and step over batched keys/states/actions.

#### \_\_init\_\_

```python
def __init__(env)
```

Wrap the environment and install vmapped reset/step methods.

**Arguments**:

- `env` - The environment being wrapped.

## NormalizeVecObsEnvState Objects

```python
@struct.dataclass
class NormalizeVecObsEnvState()
```

Running-normalization bookkeeping for :class:`NormalizeVecObservation`.

**Attributes**:

- `mean` - Running per-dimension mean of the observations.
- `var` - Running per-dimension variance of the observations.
- `count` - Running count of observations seen so far.
- `env_state` - The wrapped environment&#x27;s own state.

## NormalizeVecObservation Objects

```python
class NormalizeVecObservation(GymnaxWrapper)
```

Wrapper that normalizes vectorized observations with a running mean/variance.

Statistics are updated from the batch of observations at every reset/step
(Welford-style parallel-variant merge) and observations are returned as
``(obs - mean) / sqrt(var + 1e-8)``.

#### \_\_init\_\_

```python
def __init__(env)
```

Wrap the given environment (see :class:`GymnaxWrapper`).

#### reset

```python
def reset(key, params=None)
```

Reset the wrapped environment and initialize/update the observation statistics.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(normalized obs, NormalizeVecObsEnvState)``.

#### step

```python
def step(key, state, action, params=None)
```

Step the wrapped environment and normalize the observation batch.

The running mean/variance are first merged with the batch statistics of
the new observations; the returned observation is the batch normalized
with the updated statistics.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current :class:`NormalizeVecObsEnvState`.
- `action` - Batched actions to apply.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(normalized obs, updated NormalizeVecObsEnvState, reward,
  done, info)``.

## NormalizeVecRewEnvState Objects

```python
@struct.dataclass
class NormalizeVecRewEnvState()
```

Running-normalization bookkeeping for :class:`NormalizeVecReward`.

**Attributes**:

- `mean` - Running mean of the discounted episode returns.
- `var` - Running variance of the discounted episode returns.
- `count` - Running count of return samples seen so far.
- `return_val` - Current discounted episode return per environment.
- `env_state` - The wrapped environment&#x27;s own state.

## NormalizeVecReward Objects

```python
class NormalizeVecReward(GymnaxWrapper)
```

Wrapper that normalizes vectorized rewards by the running variance of the discounted returns.

Rewards are accumulated per environment with discount factor ``gamma`` and
each reward is divided by ``sqrt(var + 1e-8)`` of the running return
statistics.

#### \_\_init\_\_

```python
def __init__(env, gamma)
```

Wrap the environment and store the discount factor.

**Arguments**:

- `env` - The environment being wrapped.
- `gamma` - Discount factor used to accumulate episode returns.

#### reset

```python
def reset(key, params=None)
```

Reset the wrapped environment and zero the return statistics.

**Arguments**:

- `key` - PRNG key for the reset.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, NormalizeVecRewEnvState)`` with mean 0, variance 1,
  and zeroed per-environment returns.

#### step

```python
def step(key, state, action, params=None)
```

Step the wrapped environment and scale the reward by the running return std.

The discounted episode return is updated per environment
(``return_val * gamma * (1 - done) + reward``) and merged into the
running mean/variance statistics; the reward is then divided by
``sqrt(var + 1e-8)`` before being returned.

**Arguments**:

- `key` - PRNG key for the transition.
- `state` - Current :class:`NormalizeVecRewEnvState`.
- `action` - Batched actions to apply.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, updated NormalizeVecRewEnvState, scaled reward, done,
  info)``.

