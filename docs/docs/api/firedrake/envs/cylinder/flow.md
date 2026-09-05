---
sidebar_label: flow
title: hydrogym.firedrake.envs.cylinder.flow
---

## CylinderBase Objects

```python
class CylinderBase(FlowConfig)
```

Base class for circular-cylinder flow configurations (Re 100).

Uniform inflow from the left with symmetry conditions top and bottom,
outflow on the right, and a single rotary/blowing-suction actuator on
the cylinder wall implemented as a ``ScaledDirichletBC`` driven by
``cyl_velocity_field`` (subclasses define the velocity profile).
Default observations are the lift and drag coefficients.

#### TAU

Time constant for controller damping (0.01*vortex shedding period)

#### num\_inputs

```python
@property
def num_inputs() -> int
```

Number of control inputs: one (rotary control on the cylinder).

#### configure\_observations

```python
def configure_observations(obs_type=None,
                           probe_obs_types=`{}`) -> ObservationFunction
```

Select the observation function for the cylinder.

**Arguments**:

- `obs_type` _str, optional_ - Observation type. Defaults to
  &quot;lift_drag&quot;. Probe-based types passed in
  ``probe_obs_types`` are also supported.
- `probe_obs_types` _dict, optional_ - Probe-based observation
  functions provided by ``FlowConfig``.
  

**Returns**:

- `ObservationFunction` - The selected observation function.
  

**Raises**:

- `ValueError` - If ``obs_type`` is not a supported type.

#### init\_bcs

```python
def init_bcs(function_spaces=None)
```

Construct and apply the cylinder boundary conditions.

Creates the inflow, freestream (symmetry), and outflow conditions,
plus the time-varying actuation boundary condition on the cylinder
wall (``ScaledDirichletBC`` with the subclass&#x27;s
``cyl_velocity_field``), then applies the current control state.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to build conditions on; defaults to the subspaces
  of the mixed space.

#### cyl\_velocity\_field

```python
@property
def cyl_velocity_field()
```

Velocity vector for the actuation boundary condition on the cylinder.

**Raises**:

- `NotImplementedError` - In the base class; subclasses must
  override this.

#### collect\_bcu

```python
def collect_bcu() -> list[fd.DirichletBC]
```

List of velocity boundary conditions (inflow, freestream, actuation).

**Returns**:

- `list[fd.DirichletBC]` - All velocity ``DirichletBC`` objects.

#### collect\_bcp

```python
def collect_bcp() -> list[fd.DirichletBC]
```

List of pressure boundary conditions.

**Returns**:

- `list[fd.DirichletBC]` - Pressure ``DirichletBC`` objects (zero
  pressure at the outlet).

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

#### linearize\_bcs

```python
def linearize_bcs(function_spaces=None)
```

Set boundary conditions to zero-amplitude for linearized problems.

Resets the controls to zero (which scales the actuation BC to
zero) and sets the inflow velocity and freestream condition to
zero.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to rebuild the conditions on.

#### evaluate\_objective

```python
def evaluate_objective(q: fd.Function = None) -> float
```

The objective function for this flow is the drag coefficient

#### render

```python
def render(mode="human", clim=None, levels=None, cmap="RdBu", **kwargs)
```

Render the current vorticity field with the cylinder overlaid.

**Arguments**:

- `mode` _str, optional_ - Rendering mode; only &quot;human&quot; plotting is
  implemented.
- `clim` _tuple, optional_ - (min, max) color limits for the
  vorticity plot. Default (-2, 2).
- `levels` _array, optional_ - Contour levels; defaults to 10
  levels spanning ``clim``.
- `cmap` _str, optional_ - Matplotlib colormap name. Default &quot;RdBu&quot;.
- `**kwargs` - Additional keyword arguments passed to
  ``tricontourf``.

## RotaryCylinder Objects

```python
class RotaryCylinder(CylinderBase)
```

Cylinder controlled by tangential (rotary) blowing on the wall.

The actuation boundary condition is a purely tangential velocity field
of constant magnitude around the cylinder, so scaling it with the
control input implements rotational forcing.

#### cyl\_velocity\_field

```python
@property
def cyl_velocity_field()
```

Tangential velocity vector field around the cylinder surface.

**Returns**:

- `ufl.Tensor` - Unit-magnitude tangential velocity
  ``(-rad * sin(theta), rad * cos(theta))`` as a function of the
  angle ``theta`` from the cylinder center.

## Cylinder Objects

```python
class Cylinder(CylinderBase)
```

Cylinder controlled by normal blowing/suction jets on the wall.

Two jets centered at the top and bottom of the cylinder follow
Rabault et al (2018), https://arxiv.org/abs/1808.07664, with a 10-degree
angular width each; the actuation BC is scaled by the control input.

#### cyl\_velocity\_field

```python
@property
def cyl_velocity_field()
```

Velocity vector for boundary condition

Blowing/suction actuation on the cylinder wall, following Rabault, et al (2018)
https://arxiv.org/abs/1808.07664

**Returns**:

- `ufl.Tensor` - Normal (radial) velocity field on the cylinder
  wall, nonzero only within the jet widths around the top and
  bottom of the cylinder.

