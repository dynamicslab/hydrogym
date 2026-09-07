---
sidebar_label: flow
title: hydrogym.firedrake.envs.pinball.flow
---

## Pinball Objects

```python
class Pinball(FlowConfig)
```

Flow over three cylinders in a triangular &quot;pinball&quot; arrangement (Re 30).

Uniform inflow with symmetry conditions top and bottom, and one rotary
(tangential) actuator per cylinder, so the number of control inputs is
three. The default observation is the six lift/drag coefficients (one
lift-drag pair per cylinder) and the objective is the total drag.

#### MAX\_CONTROL

TODO: Limit this based on literature

#### TAU

TODO: Tune this based on vortex shedding period

#### init\_bcs

```python
def init_bcs(function_spaces=None)
```

Construct and apply the pinball boundary conditions.

Creates the inflow, freestream (symmetry), and outflow conditions,
plus one tangential (rotary) actuation boundary condition per
cylinder (each a ``ScaledDirichletBC`` scaled by the corresponding
control input), then applies the current control state.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to build conditions on; defaults to the subspaces
  of the mixed space.

#### num\_inputs

```python
@property
def num_inputs() -> int
```

Number of control inputs: one rotary actuator per cylinder (three).

#### configure\_observations

```python
def configure_observations(obs_type=None,
                           probe_obs_types=`{}`) -> ObservationFunction
```

Select the observation function for the pinball.

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

#### collect\_bcu

```python
def collect_bcu() -> Iterable[fd.DirichletBC]
```

List of velocity boundary conditions (inflow, freestream, actuation).

**Returns**:

- `Iterable[fd.DirichletBC]` - All velocity ``DirichletBC`` objects,
  one actuation BC per cylinder.

#### collect\_bcp

```python
def collect_bcp() -> Iterable[fd.DirichletBC]
```

List of pressure boundary conditions.

**Returns**:

- `Iterable[fd.DirichletBC]` - Pressure ``DirichletBC`` objects
  (zero pressure at the outlet).

#### compute\_forces

```python
def compute_forces(q: fd.Function = None) -> Iterable[float]
```

Compute dimensionless lift/drag coefficients on each cylinder.

**Arguments**:

- `q` _fd.Function, optional_ - Flow state to compute forces from;
  defaults to the current state.
  

**Returns**:

- `Iterable[float]` - Pair of lists ``(CL, CD)`` with one lift and
  one drag value per cylinder, ordered as ``CYLINDER``.

#### linearize\_bcs

```python
def linearize_bcs(function_spaces=None)
```

Set boundary conditions to zero-amplitude for linearized problems.

Resets the controls to zero (which scales the actuation BCs to
zero), reinitializes the boundary conditions, and sets the inflow
velocity and freestream condition to zero.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to rebuild the conditions on.

#### get\_observations

```python
def get_observations()
```

Compute the current observation vector of lift/drag coefficients.

**Returns**:

- `list` - Flattened list of lift coefficients followed by drag
  coefficients, one entry per cylinder.

#### evaluate\_objective

```python
def evaluate_objective(q=None)
```

Compute the objective: the total drag over all cylinders.

**Arguments**:

- `q` _fd.Function, optional_ - Flow state to evaluate; defaults to
  the current state.
  

**Returns**:

- `float` - Sum of the drag coefficients of all cylinders.

#### render

```python
def render(mode="human", clim=None, levels=None, cmap="RdBu", **kwargs)
```

Render the current vorticity field with the cylinder circles overlaid.

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
  

**Raises**:

- `AttributeError` - As written this method reads the vorticity and
  cylinder geometry from a ``self.flow`` attribute, which
  ``FlowConfig`` does not define, so rendering a bare
  ``Pinball`` instance will fail.

