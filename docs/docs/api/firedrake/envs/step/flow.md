---
sidebar_label: flow
title: hydrogym.firedrake.envs.step.flow
---

## Step Objects

```python
class Step(FlowConfig)
```

Backwards-facing step

Notes on meshes:
- &quot;coarse&quot;: outlet at L=15 with &quot;medium&quot; resolution (81k elements)
This mesh is much faster to run, but has differences of up to ~5% in
the separation and reattachment points.  It should not be considered
&quot;validated&quot; but can be used for testing and hyperparameter tuning.
- &quot;medium&quot; - outlet at L=25 (110k elements)
- &quot;fine&quot; - outlet at L=25 (223k elements)
This is the closest to the mesh used by the reference paper
(Boujo &amp; Gallaire 2015, DOI:10.1017/jfm.2014.656)

#### FUNCTIONS

This flow needs a base flow to compute fluctuation KE

#### MAX\_CONTROL

Arbitrary... should tune this

#### TAU

Time constant for controller damping (0.01*instability frequency)

#### \_\_init\_\_

```python
def __init__(**kwargs)
```

Initialize the step flow, including the random forcing parameters.

**Arguments**:

- `**kwargs` - Forwarded to ``FlowConfig``, after removing:
  - noise_amplitude (float): Amplitude of the white-noise
  forcing. Default 1.0.
  - noise_time_constant (float): Time constant of the
  low-pass filter on the noise. Defaults to
  ``10 * TAU``.
  - noise_seed (int, optional): Seed for the PCG64 random
  generator; None gives nondeterministic noise.

#### num\_inputs

```python
@property
def num_inputs() -> int
```

Number of control inputs: one (blowing/suction on the step edge).

#### nu

```python
@property
def nu()
```

Kinematic viscosity ``0.5 / Re`` (half the base-class value).

**Returns**:

- `fd.Constant` - The kinematic viscosity used by this flow.

#### body\_force

```python
@property
def body_force()
```

Localized body force driven by the low-pass-filtered noise state.

**Returns**:

- `fd.Function` - A vector function on the velocity space with a
  Gaussian bump centered at (-1.0, 0.25), scaled by the current
  ``noise_state``.

#### configure\_observations

```python
def configure_observations(obs_type=None,
                           probe_obs_types=`{}`) -> ObservationFunction
```

Select the observation function for the step.

**Arguments**:

- `obs_type` _str, optional_ - Observation type. Defaults to
  &quot;stress_sensor&quot; (shear stress on the downstream wall).
  Probe-based types passed in ``probe_obs_types`` are also
  supported.
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

Construct and apply the step boundary conditions.

Creates a parabolic inflow profile, no-slip walls, and outflow
conditions, plus the time-varying actuation boundary condition on
the step edge (``ScaledDirichletBC``), then applies the current
control state.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to build conditions on; defaults to the subspaces
  of the mixed space.

#### advance\_time

```python
def advance_time(dt, control=None)
```

Advance the flow time by ``dt``, updating the stochastic forcing first.

Draws a white-noise sample (on rank zero of the MPI communicator,
then broadcast to all ranks), low-pass-filters the noise state
with the actuator-style filter (time constant ``noise_tau``), and
then calls the parent ``advance_time`` to update the flow state
and actuator.

**Arguments**:

- `dt` _float_ - Time step size.
- `control` _ArrayLike, optional_ - Control input(s) passed through
  to the parent solver.
  

**Returns**:

- `list` - The updated actuator (control) state, as returned by
  ``FlowConfig.advance_time``.

#### linearize\_bcs

```python
def linearize_bcs(function_spaces=None)
```

Set boundary conditions to zero-amplitude for linearized problems.

Resets the controls to zero (which scales the actuation BC to
zero), reinitializes the boundary conditions, and sets the inflow
profile to zero.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to rebuild the conditions on.

#### collect\_bcu

```python
def collect_bcu()
```

List of velocity boundary conditions (inflow, walls, actuation).

**Returns**:

- `list` - All velocity ``DirichletBC`` objects for this flow.

#### collect\_bcp

```python
def collect_bcp()
```

List of pressure boundary conditions.

**Returns**:

- `list` - Pressure ``DirichletBC`` objects (zero pressure at the outlet).

#### wall\_stress\_sensor

```python
def wall_stress_sensor(q=None)
```

Integral of wall-normal shear stress (see Barbagallo et al, 2009)

#### evaluate\_objective

```python
def evaluate_objective(q=None, qB=None)
```

Compute the fluctuation kinetic energy relative to a base flow.

**Arguments**:

- `q` _fd.Function, optional_ - Flow state to evaluate; defaults to
  the current state.
- `qB` _fd.Function, optional_ - Base flow to subtract; defaults to
  the stored base flow ``self.qB``.
  

**Returns**:

- `float` - ``0.5 * ||u - uB||_L2^2`` of the velocity fields.

#### render

```python
def render(mode="human",
           axes=None,
           clim=None,
           levels=None,
           cmap="RdBu",
           xlim=None,
           **kwargs)
```

Render the current vorticity field with matplotlib.

**Arguments**:

- `mode` _str, optional_ - Rendering mode; only &quot;human&quot; plotting is
  implemented.
- `axes` _optional_ - Matplotlib axes to draw on; a new figure of
  size (12, 2) is created if None.
- `clim` _tuple, optional_ - (min, max) color limits for the
  vorticity plot. Default (-5, 5).
- `levels` _array, optional_ - Contour levels; defaults to 20
  levels spanning ``clim``.
- `cmap` _str, optional_ - Matplotlib colormap name. Default &quot;RdBu&quot;.
- `xlim` _list, optional_ - Horizontal axis limits. Default [-2, 10].
- `**kwargs` - Additional keyword arguments passed to
  ``tricontourf``.

