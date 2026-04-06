---
sidebar_label: sphere
title: hydrogym.maia.envs.sphere
---

Sphere Flow Environment
=======================

This module provides 3D sphere flow CFD environments for reinforcement learning.

## SphereBase Objects

```python
class SphereBase(MaiaFlowEnv)
```

Base class for sphere flow environments with Hugging Face integration.

This class provides common functionality for 3D sphere-based CFD environments,
including reward computation based on drag and lift/side forces.

The projection area for force coefficient calculation uses the
sphere&#x27;s circular cross-section: A = pi * D^2 / 4

The reward is computed as:
reward = -|C_D| - omega * |C_L| - omega * |C_S|

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the sphere base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on aerodynamic force coefficients.

Uses circular cross-section area for coefficient calculation.

**Returns**:

  Tuple containing:
  - reward: Scalar reward (or list for multiple boundaries)
  - obj_dict: Dictionary with force information

## Sphere Objects

```python
class Sphere(SphereBase)
```

3D sphere environment with flow control.

This environment simulates flow around a 3D sphere geometry.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the sphere environment.

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

