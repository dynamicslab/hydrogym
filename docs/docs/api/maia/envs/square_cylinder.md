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

#### drag\_coefficient\_from\_objective

```python
def drag_coefficient_from_objective(obj_dict)
```

Total drag coefficient over the boundary segments (normalised like the reward's force coefficients).

## SquareCylinder Objects

```python
class SquareCylinder(SquareCylinderBase)
```

Square cylinder environment with flow control.

This environment simulates flow around a square cylinder geometry.

Actuation power (`hydrogym.maia.wall_jets`, one actuator per jet): the two slot jets of BC 2007 (object
class 0 on the top and bottom faces, parabolic across `lbJetRanges`) are the fluid cells next to the faces
(faces on cell faces: exactly the solver's jet cells), `C_P = 2 P / (U^3 D [zLength])` like `C_D`. The jets are
independent (not zero-net-mass-flux); `p_ref` is the mean wall pressure over both jets.

Stable jet amplitude (`SquareCylinder_2D_Re200`, D/20, BGKC): sustained suction through a jet above about
0.021 lattice units (0.36 U at Ma 0.1) excites a grid-scale odd-even mode across the 4-cell slot (moving-wall
bounce-back with a wall-normal velocity on a face that lies in the near-stagnant fluid under the separated shear
layer); at 0.44 U it saturates and corrupts the forces, at 0.52 U the run diverges after 30-40 actions. Blowing,
sign-alternating commands and the circular cylinder at the same amplitude are stable. The Hugging Face
configuration therefore uses `max_control: 0.021`.

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

