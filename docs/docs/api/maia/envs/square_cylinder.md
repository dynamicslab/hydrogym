---
sidebar_label: square_cylinder
title: hydrogym.maia.envs.square_cylinder
---

Square Cylinder Flow Environment
=================================

This module provides square cylinder flow CFD environments for reinforcement learning.

## SquareCylinderBase Objects

```python
class SquareCylinderBase(MaiaFlowEnv)
```

Base class for square cylinder flow environments with Hugging Face integration.

This class provides common functionality for square cylinder-based CFD
environments, including reward computation based on drag and lift forces.

The reward is computed as:
reward = -|C_D| - omega * |C_L|

where C_D is the drag coefficient, C_L is the lift coefficient,
and omega is a weighting factor.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the square cylinder base environment.

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

## SquareCylinder Objects

```python
class SquareCylinder(SquareCylinderBase)
```

Square cylinder environment with flow control.

This environment simulates flow around a square cylinder geometry.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the square cylinder environment.

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

