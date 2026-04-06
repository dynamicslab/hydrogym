---
sidebar_label: pettingzoo_env
title: hydrogym.nek.pettingzoo_env
---

Optional PettingZoo-compatible wrapper for NekParallelEnv.
Only needed if you want to use PettingZoo-specific libraries.

## NekPettingZooEnv Objects

```python
class NekPettingZooEnv(ParallelEnv)
```

PettingZoo-compatible wrapper for NekParallelEnv.

This wrapper makes the environment compatible with PettingZoo&#x27;s API,
allowing use with PettingZoo-specific libraries and tools.

**Arguments**:

- `parallel_env` - NekParallelEnv instance to wrap

#### unwrapped

```python
@property
def unwrapped()
```

Return the base environment without wrappers.

#### observation\_space

```python
@functools.lru_cache(maxsize=None)
def observation_space(agent)
```

Return observation space for agent (cached).

#### action\_space

```python
@functools.lru_cache(maxsize=None)
def action_space(agent)
```

Return action space for agent (cached).

#### reset

```python
def reset(seed=None, options=None)
```

Reset the environment.

#### step

```python
def step(actions)
```

Step the environment.

#### render

```python
def render(mode="human")
```

Render the environment.

#### close

```python
def close()
```

Close the environment.

#### make\_pettingzoo\_env

```python
def make_pettingzoo_env(nek_env, render_mode=None)
```

Convenience function to create PettingZoo environment from NekEnv.

**Arguments**:

- `nek_env` - Base NekEnv instance
- `render_mode` - Render mode for the environment (default: None)
  

**Returns**:

  NekPettingZooEnv instance

