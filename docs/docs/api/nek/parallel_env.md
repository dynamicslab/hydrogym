---
sidebar_label: parallel_env
title: hydrogym.nek.parallel_env
---

Dict-based multi-agent wrapper for NekEnv.
Converts between array-based and dict-based interfaces.

## NekParallelEnv Objects

```python
class NekParallelEnv()
```

Multi-agent wrapper for NekEnv with dict-based interface.

This wrapper treats each actuator as a separate agent with its own
observation and action. Useful for multi-agent RL experiments or when
you need per-actuator control.

**Arguments**:

- `nek_env` - The base NekEnv instance to wrap

#### observation\_space

```python
def observation_space(agent)
```

Get observation space for a specific agent.

#### action\_space

```python
def action_space(agent)
```

Get action space for a specific agent.

#### reset

```python
def reset(seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]
```

Reset environment and return dict of observations.

#### step

```python
def step(
    actions: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[
        str, bool], Dict[str, dict]]
```

Step environment with dict of actions.

**Arguments**:

- `actions` - Dict mapping agent names to actions
  

**Returns**:

- `observations` - Dict of observations per agent
- `rewards` - Dict of rewards per agent
- `terminated` - Dict of terminated flags per agent
- `truncated` - Dict of truncated flags per agent
- `infos` - Dict of info dicts per agent

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

