---
sidebar_label: env_core
title: hydrogym.jax.env_core
---

Core Jax Gym Environment Module
================================

This module provides the base environment class for CFD reinforcement learning
with Hugging Face Hub integration for configuration management.

## ConfigError Objects

```python
class ConfigError(Exception)
```

Exception raised for configuration-related errors.

## JAXFlowEnv Objects

```python
class JAXFlowEnv(environment.Environment[EnvState, EnvParams])
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

## JAXFlowEnvBase Objects

```python
class JAXFlowEnvBase(environment.Environment[EnvState, EnvParams])
```

Base JAXFlowEnv without Hugging Face Hub integration.
Contains core environment interface methods required by Gymnax.

## GymnaxWrapper Objects

```python
class GymnaxWrapper(object)
```

Base class for Gymnax wrappers.

## FlattenObservationWrapper Objects

```python
class FlattenObservationWrapper(GymnaxWrapper)
```

Flatten the observations of the environment.

## LogWrapper Objects

```python
class LogWrapper(GymnaxWrapper)
```

Log the episode returns and lengths.

## ClipAction Objects

```python
class ClipAction(GymnaxWrapper)
```

#### step

```python
def step(key, state, action, params=None)
```

TODO: In theory the below line should be the way to do this.

