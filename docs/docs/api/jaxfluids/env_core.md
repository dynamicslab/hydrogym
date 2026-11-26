---
sidebar_label: env_core
title: hydrogym.jaxfluids.env_core
---

## JAXFluidsFlowEnv Objects

```python
class JAXFluidsFlowEnv(HFEnvConfigMixin, JAXFluidsEnv)
```

Base JAXFluidsFlowEnv with Hugging Face Hub integration for configuration management.

**Arguments**:

  - environment_name: Required. Name of the environment.
  - hf_repo_id: Hugging Face repository (default: 'dynamicslab/HydroGym-environments')
  
  - use_clean_cache: Use clean cache directory (default: True)
  * True - Creates fresh workspace copy (recommended for production)
  * False - Uses cached workspace (faster for development/testing)
  - local_fallback_dir: Local directory for offline usage
  - configuration_file: Custom path to the environment's config.yaml (optional)
  
  Subclass of `jaxfluids_rl.jxf_env.JAXFluidsEnv`; the environment data comes from the Hugging Face Hub.

