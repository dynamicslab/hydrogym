---
sidebar_label: env
title: hydrogym.nek.env
---

Core Nek5000 environment with Gymnasium interface.
Single-agent with array-based observations/actions.

Supports two initialization patterns:
1. MAIA pattern (recommended): env = NekEnv.from_hf(&#x27;EnvName&#x27;, nproc=10)
2. Legacy pattern: env = NekEnv(conf=config_obj)

## ConfigError Objects

```python
class ConfigError(Exception)
```

Exception raised for configuration-related errors.

#### mpi\_split

```python
def mpi_split(comm_world: MPI.Comm, nproc: Optional[int] = None) -> MPI.Comm
```

Split MPI world into master/worker inter-communicator.

**Arguments**:

- `comm_world` - MPI communicator
- `nproc` - Expected number of Nek workers (for validation)
  

**Returns**:

  Inter-communicator between controller and workers

## NekEnv Objects

```python
class NekEnv(gym.Env)
```

Core Nek5000 environment with Gymnasium interface.

This is a single-agent environment where the agent controls multiple
actuators (control points) on the mesh. Observations and actions are
flat arrays representing all actuators.

Supports two initialization patterns:

1. MAIA pattern (recommended):
env = NekEnv.from_hf(
&#x27;MiniChannel_Re180&#x27;,
nproc=10,
hostfile=&#x27;&#x27;,
)

2. Legacy pattern (deprecated):
conf = OmegaConf.load(&#x27;config.yaml&#x27;)
env = NekEnv(conf=conf)

Args (MAIA pattern via env_config):
environment_name: Name of environment on HuggingFace
nproc: Number of MPI workers for Nek (required)
hostfile: MPI hostfile path (default: &#x27;&#x27;)
hf_repo_id: HuggingFace repository (default: &#x27;dynamicslab/HydroGym-environments&#x27;)
use_clean_cache: Use fresh workspace (default: True)
local_fallback_dir: Local directory for offline usage
configuration_file: Override config file path
run_root: Root directory for outputs (default: &#x27;runs&#x27;)
run_name: Name for this run (default: &#x27;&#x27; = no subdirectory, use run_root directly)
reward_agg: Reward aggregation method (&quot;mean&quot; or &quot;sum&quot;)
... (runtime overrides for config parameters)

Args (Legacy pattern):
conf: Configuration object (OmegaConf)
run_root: Root directory for run outputs
run_name: Name for this run (defaults to MPI rank)
reward_agg: How to aggregate per-actuator rewards (&quot;mean&quot; or &quot;sum&quot;)

#### \_\_init\_\_

```python
def __init__(conf: Optional[Config] = None,
             env_config: Optional[Dict] = None,
             run_root: str = ".",
             run_name: Optional[str] = None,
             reward_agg: str = "mean",
             **kwargs)
```

Initialize NekEnv with either legacy conf or MAIA env_config pattern.

**Arguments**:

- `conf` - Legacy OmegaConf object (deprecated)
- `env_config` - MAIA-style configuration dict (recommended)
- `run_root` - Output directory root
- `run_name` - Run name (auto-generate if None)
- `reward_agg` - Reward aggregation method
- `**kwargs` - Additional parameters (for backward compatibility)

#### from\_hf

```python
@classmethod
def from_hf(cls,
            environment_name: str,
            nproc: int,
            hostfile: str = "",
            **kwargs)
```

Create environment from HuggingFace Hub (MAIA pattern).

**Arguments**:

- `environment_name` - Name of the environment (e.g., &#x27;MiniChannel_Re180&#x27;)
- `nproc` - Number of MPI workers for Nek (required)
- `hostfile` - MPI hostfile path (default: &#x27;&#x27;)
- `**kwargs` - Additional env_config parameters:
  - hf_repo_id: HF repository (default: &#x27;dynamicslab/HydroGym-environments&#x27;)
  - use_clean_cache: Fresh workspace (default: True)
  - local_fallback_dir: Local directory
  - configuration_file: Override config path
  - run_root: Output directory (default: &#x27;runs&#x27;)
  - run_name: Run name (auto-generate if None)
  - reward_agg: &#x27;mean&#x27; or &#x27;sum&#x27; (default: &#x27;mean&#x27;)
  - normalize_input: Override normalization strategy
  - nb_interactions: Override episode length
  - random_init: Override IC randomization
  - rescale_actions: Override action rescaling
  - rew_mode: Override reward mode
  

**Returns**:

  NekEnv instance
  

**Example**:

  env = NekEnv.from_hf(&#x27;MiniChannel_Re180&#x27;, nproc=10)
  
  env = NekEnv.from_hf(
  &#x27;MiniChannel_Re180&#x27;,
  nproc=10,
  hostfile=&#x27;hosts.txt&#x27;,
  use_clean_cache=True,
  normalize_input=&#x27;utau&#x27;,
  )

#### reset

```python
def reset(seed=None, options=None) -> Tuple[np.ndarray, dict]
```

Reset the environment.

#### step

```python
def step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]
```

Step the environment.

**Arguments**:

- `action` - Flat array of actions for all actuators, shape (n_actuators,)
  

**Returns**:

- `observation` - Flat array of observations, shape (n_actuators * obs_per_actuator,)
- `reward` - Scalar reward
- `terminated` - Whether episode is done
- `truncated` - Whether episode was truncated (always False for Nek)
- `info` - Additional information

#### render

```python
def render(mode="human")
```

Render the environment (not implemented).

#### close

```python
def close()
```

Close the environment.

## RingBuffer Objects

```python
class RingBuffer()
```

N-dimensional ring buffer using numpy arrays.

#### extend

```python
def extend(x)
```

Add array x to ring buffer.

#### get

```python
def get()
```

Return first-in-first-out data in the ring buffer.

#### average

```python
def average()
```

Return average of entries in the ring buffer.

