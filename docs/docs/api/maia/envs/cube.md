---
sidebar_label: cube
title: hydrogym.maia.envs.cube
---

Cube Flow Environment
=====================

This module provides 3D cube flow CFD environments for reinforcement learning.

## CubeBase Objects

```python
class CubeBase(MaiaFlowEnv)
```

Base class for cube flow environments with Hugging Face integration.

This class provides common functionality for 3D cube-based CFD environments,
including reward computation based on drag and lift/side forces.

The reward is computed as:
reward = -|C_D| - omega * |C_L| - omega * |C_S|

where C_D is the drag coefficient, C_L is the lift coefficient,
C_S is the side force coefficient, and omega is a weighting factor.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cube base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on aerodynamic force coefficients.

Calculates non-dimensional force coefficients and returns a reward
that penalizes drag, lift, and side forces.

**Returns**:

  Tuple containing:
  - reward: Scalar reward (or list for multiple boundaries)
  - obj_dict: Dictionary with force information

## Cube Objects

```python
class Cube(CubeBase)
```

3D cube environment with flow control.

This environment simulates flow around a 3D cube geometry.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cube environment.

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

