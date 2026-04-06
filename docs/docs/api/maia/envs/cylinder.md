---
sidebar_label: cylinder
title: hydrogym.maia.envs.cylinder
---

Cylinder Flow Environment
=========================

This module provides cylinder flow CFD environments for reinforcement learning,
including standard cylinder and rotary cylinder configurations.

## CylinderBase Objects

```python
class CylinderBase(MaiaFlowEnv)
```

Base class for cylinder flow environments with Hugging Face integration.

This class provides the common functionality for cylinder-based CFD
environments, including reward computation based on drag and lift forces.

The reward is computed as:
reward = -|C_D| - omega * |C_L|

where C_D is the drag coefficient, C_L is the lift coefficient,
and omega is a weighting factor.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cylinder base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on aerodynamic force coefficients.

Calculates non-dimensional force coefficients and returns a reward
that penalizes drag and lift forces.

**Returns**:

  Tuple containing:
  - reward: Scalar reward (or list for multiple boundaries)
  - obj_dict: Dictionary with force information

## Cylinder Objects

```python
class Cylinder(CylinderBase)
```

Standard cylinder environment with jet actuation.

This environment simulates flow around a circular cylinder with
synthetic jet actuators for flow control. Jets are configured
in pairs for zero net mass flux actuation.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cylinder environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> List[float]
```

Convert RL action to CFD actuation sequence.

Implements zero net mass flux by pairing jets with opposite signs.
For each jet pair, the first jet uses +action and the second uses -action.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Actuation sequence for the CFD solver.

## RotaryCylinder Objects

```python
class RotaryCylinder(CylinderBase)
```

Rotary cylinder environment with rotational actuation.

This environment simulates flow around a circular cylinder that can
rotate. The actuation controls the cylinder&#x27;s angular velocity.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the rotary cylinder environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Convert RL action to CFD actuation format.

For rotary cylinder, the action directly controls angular velocity.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

