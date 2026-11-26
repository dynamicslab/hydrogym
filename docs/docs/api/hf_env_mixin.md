---
sidebar_label: hf_env_mixin
title: hydrogym.hf_env_mixin
---

Shared HF environment-config resolution for the jaxfluids, jax, maia and nek backends.

`HFEnvConfigMixin` provides three methods:

_setup_environment_data()      -- local ~/.cache/&lt;namespace>/&lt;env> lookup,
falling back to HFDataManager
_resolve_configuration_file()  -- None / abs path / ./relative / filename
_find_configuration_file()     -- auto-detect config.yaml in the env dir

parameterized by the class-level `HF_CACHE_NAMESPACE` (and, for data-manager
construction, `SOLVER_TYPE`).

## ConfigError Objects

```python
class ConfigError(Exception)
```

Exception raised for configuration-related errors.

## HFEnvConfigMixin Objects

```python
class HFEnvConfigMixin()
```

Environment-data download / config-file resolution shared by the
HF-backed backends.

Requires on the consuming class:
- `HF_CACHE_NAMESPACE`: per-backend cache directory name under
`~/.cache` (e.g. "jaxfluidsgym", "jaxgym", "maiagym", "nekgym").
- `self.environment_name`, `self.env_data_path`, and a
`self.data_manager` (HFDataManager) instance before the
resolution methods are called (i.e. `_init_from_hf` order:
data_manager -> environment_name -> _setup_environment_data ->
_resolve_configuration_file).

