---
sidebar_label: flow
title: hydrogym.firedrake.envs.cylinder.flow
---

## CylinderBase Objects

```python
class CylinderBase(FlowConfig)
```

#### TAU

Time constant for controller damping (0.01*vortex shedding period)

#### cyl\_velocity\_field

```python
@property
def cyl_velocity_field()
```

Velocity vector for boundary condition

#### compute\_forces

```python
def compute_forces(q: fd.Function = None) -> tuple[float]
```

Compute dimensionless lift/drag coefficients on cylinder

**Arguments**:

  q (fd.Function, optional):
  Flow state to compute shear force from, if not the current state.
  

**Returns**:

- `Iterable[float]` - Tuple of (lift, drag) coefficients

#### shear\_force

```python
def shear_force(q: fd.Function = None) -> float
```

Net shear force acting tangentially to the cylinder surface

Implements the general case of the article below:
http://www.homepages.ucl.ac.uk/~uceseug/Fluids2/Notes_Viscosity.pdf

**Arguments**:

  q (fd.Function, optional):
  Flow state to compute shear force from, if not the current state.
  

**Returns**:

- `float` - Tangential shear force

#### evaluate\_objective

```python
def evaluate_objective(q: fd.Function = None) -> float
```

The objective function for this flow is the drag coefficient

## Cylinder Objects

```python
class Cylinder(CylinderBase)
```

#### cyl\_velocity\_field

```python
@property
def cyl_velocity_field()
```

Velocity vector for boundary condition

Blowing/suction actuation on the cylinder wall, following Rabault, et al (2018)
https://arxiv.org/abs/1808.07664

