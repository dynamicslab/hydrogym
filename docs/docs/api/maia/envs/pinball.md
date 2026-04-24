---
sidebar_label: pinball
title: hydrogym.maia.envs.pinball
---

Pinball Flow Environment
========================

This module provides pinball (multi-cylinder) flow CFD environments
for reinforcement learning.

## PinballBase Objects

```python
class PinballBase(MaiaFlowEnv)
```

Base class for pinball flow environments with Hugging Face integration.

The pinball configuration consists of multiple cylinders arranged in a
triangular pattern. Forces from all cylinders are summed for the reward.

The reward is computed as:
reward = -|sum(C_D)| - omega * |sum(C_L)|

where C_D and C_L are summed across all cylinders.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the pinball base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on total aerodynamic force coefficients.

Aggregates forces from all cylinders in the pinball configuration.

**Returns**:

  Tuple containing:
  - reward: Scalar reward based on total drag and lift
  - obj_dict: Dictionary with force coefficient information

## Pinball Objects

```python
class Pinball(PinballBase)
```

Rotary pinball environment with rotational actuation.

This environment controls flow using cylinder rotation.
Each cylinder can rotate independently.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the pinball environment.

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

## JetPinball Objects

```python
class JetPinball(PinballBase)
```

Jet-actuated pinball environment.

This environment controls flow using synthetic jets on the cylinders.
Jets are paired for zero net mass flux actuation.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the jet pinball environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> List[float]
```

Convert RL action to CFD actuation sequence.

Implements zero net mass flux by pairing jets with opposite signs.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Actuation sequence for the CFD solver.

