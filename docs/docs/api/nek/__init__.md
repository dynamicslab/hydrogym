---
sidebar_label: nek
title: hydrogym.nek
---

HydroGym Nek5000 backend.

Initialization Patterns:
1. MAIA pattern (recommended):
env = NekEnv.from_hf(&#x27;MiniChannel_Re180&#x27;, nproc=10)

2. Legacy pattern (deprecated):
conf = OmegaConf.load(&#x27;config.yaml&#x27;)
env = NekEnv(conf=conf)

Three-layer architecture:
1. NekEnv - Base single-agent Gym environment (array-based)
2. NekParallelEnv - Dict-based multi-agent wrapper
3. NekPettingZooEnv - Optional PettingZoo compatibility layer

Most users should use NekEnv directly.

#### load\_nek\_config

```python
def load_nek_config(config_path: str, overrides=None)
```

Load Nek config from YAML and apply overrides.

DEPRECATED: This function is kept for backwards compatibility.
The new NekEnv accepts OmegaConf objects directly.

