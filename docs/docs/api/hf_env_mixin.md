---
sidebar_label: hf_env_mixin
title: hydrogym.hf_env_mixin
---

Shared Hugging-Face environment-config resolution logic (audit Task 3.1).

Four backends (jaxfluids, jax, maia, nek) carried near-identical private
copies of the same three methods:

_setup_environment_data()      -- local ~/.cache/&lt;namespace&gt;/&lt;env&gt; lookup,
falling back to HFDataManager
_resolve_configuration_file()  -- None / abs path / ./relative / filename
_find_configuration_file()     -- auto-detect config.yaml in the env dir

with only the cache-namespace string differing per backend. This module
extracts them into `HFEnvConfigMixin`, parameterized by the class-level
`HF_CACHE_NAMESPACE` (and, for data-manager construction, `SOLVER_TYPE`).
Backends migrate to the mixin one at a time to keep blast radius small;
the method bodies here are copied verbatim from the JAX-Fluids backend,
which is migrated first (Task 3.1).

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
- ``HF_CACHE_NAMESPACE``: per-backend cache directory name under
``~/.cache`` (e.g. &quot;jaxfluidsgym&quot;, &quot;jaxgym&quot;, &quot;maiagym&quot;, &quot;nekgym&quot;).
- ``self.environment_name``, ``self.env_data_path``, and a
``self.data_manager`` (HFDataManager) instance before the
resolution methods are called (i.e. ``_init_from_hf`` order:
data_manager -&gt; environment_name -&gt; _setup_environment_data -&gt;
_resolve_configuration_file).

