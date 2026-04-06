---
sidebar_label: env_core
title: hydrogym.maia.env_core
---

Core MaiaGym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

## ConfigError Objects

```python
class ConfigError(Exception)
```

Exception raised for configuration-related errors.

## MaiaFlowEnv Objects

```python
class MaiaFlowEnv(gym.Env)
```

Base MaiaFlowEnv with Hugging Face Hub integration for configuration management.

This environment provides a Gymnasium-compatible interface for CFD simulations
using the m-AIA solver. It handles:
- Environment data management via Hugging Face Hub
- Configuration file resolution and loading
- MPI communication with the CFD solver
- Observation normalization strategies
- Action space configuration

**Attributes**:

- `environment_name` - Name of the CFD environment configuration.
- `env_data_path` - Path to the local environment data directory.
- `cfg` - OmegaConf configuration object.
- `maiaInterface` - MPI interface for communication with m-AIA.
- `observation_space` - Gymnasium observation space.
- `action_space` - Gymnasium action space.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the MaiaFlowEnv environment.

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

#### configure\_observations

```python
def configure_observations() -> None
```

Configure the number of observations based on observation type and probes.

Sets self.num_outputs based on which observation types are enabled
(forces, u, v, w, rho, p) and the number of probes.

#### setup\_normalization

```python
def setup_normalization() -> None
```

Setup observation normalization factors based on the selected strategy.

Strategies:
- &#x27;U_inf&#x27;: Normalize by freestream velocity
- &#x27;none&#x27;: No normalization (loc=0, scale=1)
- &#x27;probewise_mean_std&#x27;: Compute mean/std from simulation episodes
- &#x27;customized&#x27;: Use user-provided loc and scale values

**Raises**:

- `ConfigError` - If normalization configuration is invalid.

#### compute\_normalization\_factors

```python
def compute_normalization_factors(zero_actuation: bool = False) -> None
```

Compute normalization factors for observations by running simulation episodes.

Runs several episodes with random (or zero) actuation to collect probe data,
then computes mean and standard deviation for normalization.

**Arguments**:

- `zero_actuation` - If True, use zero actuation. Otherwise, use random actuation.

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

#### step

```python
def step(
    action: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

Advance the state of the environment by one step.

**Arguments**:

- `action` - Action array with values in [-1, 1], scaled by MAX_CONTROL.
  

**Returns**:

  Tuple of (observation, reward, terminated, truncated, info).

#### reset

```python
def reset(seed: Optional[int] = None,
          options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]
```

Reset the environment to initial state.

**Arguments**:

- `seed` - Optional random seed for reproducibility.
- `options` - Optional reset options.
  

**Returns**:

  Tuple of (initial_observation, info).

#### compute\_nondim\_coefficients

```python
def compute_nondim_coefficients(forces: np.ndarray,
                                density: float = 1.0,
                                referenceVelocity: float = 0.1 / np.sqrt(3),
                                projectionLength: float = 20.0) -> np.ndarray
```

Compute non-dimensionalized force coefficients.

**Arguments**:

- `forces` - Force array from the CFD solver.
- `density` - Reference density.
- `referenceVelocity` - Reference velocity.
- `projectionLength` - Reference projection length.
  

**Returns**:

  Non-dimensional force coefficients.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> List
```

Convert RL action to CFD actuation format.

This method should be overridden by subclasses to implement
environment-specific action conversion.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward for the current state.

This method should be overridden by subclasses to implement
environment-specific reward computation.

**Returns**:

  Tuple of (reward, objective_dict).

#### check\_complete

```python
def check_complete() -> bool
```

Check if the episode is complete.

**Returns**:

  True if the episode has reached max_episode_steps.

#### close

```python
def close() -> None
```

Close the environment and signal maia to finish.

This sends a finish signal to the maia solver to properly
terminate the MPMD coupling.

#### set\_observation\_action\_spaces

```python
def set_observation_action_spaces() -> None
```

Set up the observation and action spaces for Gymnasium compatibility.

#### configure\_probe\_dimensions

```python
def configure_probe_dimensions() -> None
```

Configure probe dimensions based on simulation dimensionality.

Sets noProbeVars (number of variables per probe) and noProbes (total probes).

#### register\_environment

```python
def register_environment(env_prefix: str, env_class: type) -> None
```

Register an environment class for automatic detection.

**Arguments**:

- `env_prefix` - Environment prefix (e.g., &#x27;Cylinder&#x27;).
- `env_class` - Environment class to register.

#### from\_hf

```python
def from_hf(environment_name: str,
            hf_repo_id: str = "dynamicslab/HydroGym-environments",
            **kwargs) -> MaiaFlowEnv
```

Factory function to create any MaiaGym environment from Hugging Face Hub.

**Arguments**:

- `environment_name` - Name of environment (e.g., &#x27;Cylinder_2D_Re200&#x27;).
- `hf_repo_id` - Hugging Face repository ID.
- `**kwargs` - Additional environment configuration parameters.
  

**Returns**:

  Configured environment instance.
  

**Raises**:

- `ValueError` - If environment name format is invalid or type is unknown.
- `ConfigError` - If environment creation fails.
  

**Examples**:

  &gt;&gt;&gt; env = maiaGym.from_hf(&#x27;Cylinder_2D_Re200&#x27;)
  &gt;&gt;&gt; env = maiaGym.from_hf(&#x27;RotaryCylinder_2D_Re1000&#x27;, obs_loc=custom_loc)
  &gt;&gt;&gt; env = maiaGym.from_hf(&#x27;Cavity_2D_Re4140&#x27;)

#### list\_available\_environments

```python
def list_available_environments(
        hf_repo_id: str = "dynamicslab/HydroGym-environments") -> List[str]
```

List all available environments from HF Hub.

**Arguments**:

- `hf_repo_id` - Hugging Face repository ID.
  

**Returns**:

  List of environment names.

#### list\_registered\_types

```python
def list_registered_types() -> Dict[str, type]
```

List all registered environment types.

**Returns**:

  Dictionary mapping environment types to their classes.

