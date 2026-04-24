---
sidebar_label: cavity
title: hydrogym.maia.envs.cavity
---

Cavity Flow Environment
========================

This module provides cavity flow CFD environments for reinforcement learning,
designed for flow control applications in cavity configurations.

## CavityBase Objects

```python
class CavityBase(MaiaFlowEnv)
```

Base class for cavity flow environments with Hugging Face integration.

This environment implements two reward strategies:
- &#x27;baseline_mean&#x27;: Penalize deviation from a pre-computed baseline state
- &#x27;running_mean&#x27;: Penalize deviation from a running average

**Attributes**:

- `reward_strategy` - Strategy for reward computation (&#x27;baseline_mean&#x27; or &#x27;running_mean&#x27;).

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cavity base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary containing:
  - reward_strategy (str): &#x27;baseline_mean&#x27; or &#x27;running_mean&#x27;. Default: &#x27;baseline_mean&#x27;

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on the selected strategy.

For &#x27;running_mean&#x27;: Penalizes deviation from exponential moving average.
For &#x27;baseline_mean&#x27;: Penalizes deviation from pre-computed baseline.

**Returns**:

  Tuple containing:
  - reward: Negative sum of absolute deviations
  - obj_dict: Empty dictionary (for compatibility)
  

**Raises**:

- `ValueError` - If unknown reward strategy is specified.

## Cavity Objects

```python
class Cavity(CavityBase)
```

Single-jet cavity environment.

This environment simulates cavity flow with a single jet actuator
for flow control.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cavity environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Convert RL action to CFD actuation format.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

## Cavity3Jet Objects

```python
class Cavity3Jet(CavityBase)
```

Three-jet cavity environment.

This environment simulates cavity flow with three independent
jet actuators for flow control.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the 3-jet cavity environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Convert RL action to CFD actuation format.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

