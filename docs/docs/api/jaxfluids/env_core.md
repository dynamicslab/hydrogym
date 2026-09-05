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
  - hf_repo_id: Hugging Face repository (default: &#x27;dynamicslab/HydroGym-environments&#x27;)
  
  - use_clean_cache: Use clean cache directory (default: True)
  * True - Creates fresh workspace copy (recommended for production)
  * False - Uses cached workspace (faster for development/testing)
  - local_fallback_dir: Local directory for offline usage
  - configuration_file: Custom path to MAIA config.yaml (optional)
  
  :param JAXFluidsEnv: _description_
  :type JAXFluidsEnv: _type_

