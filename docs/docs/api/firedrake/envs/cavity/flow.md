---
sidebar_label: flow
title: hydrogym.firedrake.envs.cavity.flow
---

## Cavity Objects

```python
class Cavity(FlowConfig)
```

Open cavity flow configuration (Re 7500).

Rectangular open cavity with inflow on the left, free-slip top, and a
blowing/suction actuator on the leading edge whose velocity profile
follows Barbagallo et al (2009). The default observation is the
integral wall-normal shear stress at the trailing edge, and the
objective is the fluctuation kinetic energy relative to a stored base
flow `qB` (hence `FUNCTIONS = ("q", "qB")`).

Actuation power (one actuator, the leading-edge slot):
`P = int v_n (p - p_ref + |u|^2 / 2) dA` over the `CONTROL` boundary,
`C_P = P / (rho U^3 L / 2)` with density 1, `U_ref = U_inf = 1` and
`L_ref = 1` (cavity length = depth, the length of Re), 2D per unit
span. The objective (fluctuation kinetic energy) is not a force
coefficient, so there is no `drag_coefficient`. The slot only blows
(net mass flux ~0.078 U L per unit control), so `P` depends on the
pressure gauge; `p_ref` is the mean slot pressure (the slot is its own
actuator group), which makes `P` gauge-independent but leaves out the
work `p_ref Q` of supplying the net flow `Q`. Forms per surface point
with the pressure and kinetic terms clipped separately.

#### U\_REF

free-stream velocity

#### L\_REF

cavity length (= depth)

#### FUNCTIONS

This flow needs a base flow to compute fluctuation KE

#### TAU

Time constant for controller damping (0.01*instability frequency)

#### num\_inputs

```python
@property
def num_inputs() -> int
```

Number of control inputs: one (blowing/suction on the leading edge).

#### configure\_observations

```python
def configure_observations(obs_type=None,
                           probe_obs_types={}) -> ObservationFunction
```

Select the observation function for the cavity.

**Arguments**:

- `obs_type` _str, optional_ - Observation type. Defaults to
  "stress_sensor". Probe-based types passed in
  `probe_obs_types` are also supported.
- `probe_obs_types` _dict, optional_ - Probe-based observation
  functions provided by `FlowConfig`.
  

**Returns**:

- `ObservationFunction` - The selected observation function.
  

**Raises**:

- `ValueError` - If `obs_type` is not a supported type.

#### init\_bcs

```python
def init_bcs(function_spaces=None)
```

Construct and apply the cavity boundary conditions.

Creates the inflow, freestream, no-slip wall, free-slip top, and
outflow conditions, plus the time-varying leading-edge actuation
boundary condition (stored in `bcu_actuation` as a
`ScaledDirichletBC`), then applies the current control state.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to build conditions on; defaults to the subspaces
  of the mixed space.

#### collect\_bcu

```python
def collect_bcu()
```

List of velocity boundary conditions (inflow, freestream, walls, slip, actuation).

**Returns**:

- `list` - All velocity `DirichletBC` objects for this flow.

#### collect\_bcp

```python
def collect_bcp()
```

List of pressure boundary conditions.

**Returns**:

- `list` - Pressure `DirichletBC` objects (zero pressure at the outlet).

#### linearize\_bcs

```python
def linearize_bcs(function_spaces=None)
```

Set boundary conditions to zero-amplitude for linearized problems.

Resets the controls to zero (which scales the actuation BC to
zero), reinitializes the boundary conditions, and sets the inflow
velocity to zero.

**Arguments**:

- `function_spaces` _optional_ - Pair of (velocity, pressure)
  spaces to rebuild the conditions on.

#### wall\_stress\_sensor

```python
def wall_stress_sensor(q=None)
```

Integral of wall-normal shear stress (see Barbagallo et al, 2009)

#### actuation\_power\_forms

```python
def actuation_power_forms(q: fd.Function = None)
```

Slot power: `([C_P], forms)` (see `FlowConfig.jet_power`; `p_ref` = mean slot pressure).

#### actuation\_power

```python
def actuation_power(q: fd.Function = None) -> np.ndarray
```

Slot (blowing/suction) power coefficient `[C_P]`.

#### evaluate\_objective

```python
def evaluate_objective(q=None, qB=None)
```

Compute the fluctuation kinetic energy relative to a base flow.

**Arguments**:

- `q` _fd.Function, optional_ - Flow state to evaluate; defaults to
  the current state.
- `qB` _fd.Function, optional_ - Base flow to subtract; defaults to
  the stored base flow `self.qB`.
  

**Returns**:

- `float` - `0.5 * ||u - uB||_L2^2` of the velocity fields.

#### render

```python
def render(mode="human", clim=None, levels=None, cmap="RdBu", **kwargs)
```

Render the current vorticity field with matplotlib.

**Arguments**:

- `mode` _str, optional_ - Rendering mode; only "human" plotting is
  implemented.
- `clim` _tuple, optional_ - (min, max) color limits for the
  vorticity plot. Default (-10, 10).
- `levels` _array, optional_ - Contour levels; defaults to 20
  levels spanning `clim`.
- `cmap` _str, optional_ - Matplotlib colormap name. Default "RdBu".
- `**kwargs` - Additional keyword arguments passed to
  `tricontourf`.

