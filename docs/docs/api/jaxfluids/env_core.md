---
sidebar_label: env_core
title: hydrogym.jaxfluids.env_core
---

Core JAXFluids Gym Environment Module
======================================

This module provides the base environment class for CFD reinforcement learning
using the JAXFluids solver backend, with Hugging Face Hub integration for
configuration and environment data management.

## ConfigError Objects

```python
class ConfigError(Exception)
```

Exception raised for configuration-related errors.

## JAXFluidsFlowEnv Objects

```python
class JAXFluidsFlowEnv(JAXFluidsEnv)
```

Base JAXFluidsFlowEnv with Hugging Face Hub integration for configuration management.

**Attributes**:

- `environment_name` - Name of the CFD environment configuration.
- `hf_repo_id` - Hugging Face repository ID (default: `'dynamicslab/HydroGym-environments'`).
- `env_data_path` - Path to the local environment data directory.
- `conf` - OmegaConf configuration object loaded from the environment's config file.
- `use_clean_cache` - Whether to use a clean workspace copy.
- `local_fallback_dir` - Local directory for offline usage.

#### \_init\_from\_hf

```python
def _init_from_hf(env_config: dict) -> None
```

Initialize HF data manager and load environment configuration.

Must be called in the constructor of any subclass before calling `super().__init__`.

**Arguments**:

- `env_config` - Configuration dictionary containing:
  - `environment_name` (str): Required. Name of the environment.
  - `hf_repo_id` (str): HF repository ID. Default: `'dynamicslab/HydroGym-environments'`.
  - `local_fallback_dir` (str): Optional local fallback directory.
  - `use_clean_cache` (bool): Whether to use clean cache. Default: `True`.
  - `configuration_file` (str): Optional path to the YAML config file.

**Raises**:

- `ConfigError` - If `environment_name` is missing or no configuration file can be found.

#### \_setup\_environment\_data

```python
def _setup_environment_data() -> str
```

Download and set up environment data from HF Hub.

Checks `~/.cache/jaxfluidsgym/<environment_name>` first; falls back to the
HF data manager if the local cache does not exist.

**Returns**:

  Absolute path to the local environment data directory.

**Raises**:

- `ConfigError` - If environment data cannot be retrieved.

#### \_resolve\_configuration\_file

```python
def _resolve_configuration_file(config_file_input: Optional[str]) -> Optional[str]
```

Resolve a configuration file path from various input formats.

**Arguments**:

- `config_file_input` - Can be:
  - `None`: Auto-detect in the HF environment directory.
  - Absolute path: Used directly.
  - Relative path starting with `./` or `../`: Resolved from the current working directory.
  - Plain filename: Searched first in the current directory, then in the environment directory.

**Returns**:

  Absolute path to the configuration file, or `None` if not found.

**Raises**:

- `ConfigError` - If a specified configuration file path does not exist.

#### \_find\_configuration\_file

```python
def _find_configuration_file() -> Optional[str]
```

Auto-detect the configuration file in the environment data directory.

Looks for `config.yaml`, `environment_config.yaml`, `env_config.yaml`,
`environment.yaml`, `<environment_name>.yaml`, and glob patterns
`config_*.yaml` / `config_*.yml`, in that order.

**Returns**:

  Absolute path to the configuration file, or `None` if not found.
