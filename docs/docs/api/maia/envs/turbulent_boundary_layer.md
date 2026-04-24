---
sidebar_label: turbulent_boundary_layer
title: hydrogym.maia.envs.turbulent_boundary_layer
---

Turbulent Boundary Layer Environments
======================================

This module provides zero-pressure-gradient turbulent boundary layer (ZPG TBL)
CFD environments for reinforcement learning.

Two actuation strategies share the ``ZPGTBLBase`` class:

``ZPGTBLJet``
Jet-based actuation.  Actions lie in ``[−MAX_CONTROL, +MAX_CONTROL]``
and are passed directly to the CFD solver.

``ZPGTBLSurfaceWave``
Actuated traveling surface wave parameterized by
``[amplitude, speed, wavelength]``.  All three parameters are strictly
positive, which requires a per-action asymmetric action space and a
non-zero reset; see the class docstring for config details.

Both environments receive the same solver force output
``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements), where ``C_P`` is
the power coefficient (``0.0`` for the jet case) and ``area`` is the wetted
wall area used for normalisation.  Forces and power are pre-normalised by
the solver by dynamic pressure (``q_∞ = ρ_∞ · ½ · U_∞²``) and
``q_∞ · U_∞`` respectively, so dividing by ``area`` yields the final
dimensionless coefficients.  Observation sizing and normalization for this
extended force vector are handled in ``ZPGTBLBase``.

## ZPGTBLBase Objects

```python
class ZPGTBLBase(MaiaFlowEnv)
```

Base class for zero-pressure-gradient turbulent boundary layer environments.

Uses the structured-grid m-AIA solver (``MAIA_STRCTRD``).

Both actuation variants (jet and surface wave) receive the solver force
output ``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements).
Forces ``F_x … F_z`` and ``C_P`` are pre-normalised by the solver using
the dynamic pressure ``q_∞ = ρ_∞ · ½ · U_∞²`` (forces) and
``q_∞ · U_∞`` (power), so dividing by the returned ``area`` gives the
final dimensionless drag and power coefficients.

This class provides:

* ``configure_observations()`` – sizes the force observation slot to
``nDim + 2`` per boundary.
* ``setup_normalization()`` – handles the ``C_P`` and ``area`` elements
in the ``&#x27;U_inf&#x27;`` normalization strategy.
* ``get_reward()`` – unified reward for both actuation variants:

.. math::

R = -C_D - \omega \cdot C_P

where ``C_D = F_x / area`` and ``C_P = forces[3] / area``.
For the jet case ``C_P = 0``, so the reward reduces to ``-C_D``.

#### configure\_observations

```python
def configure_observations() -> None
```

Configure the number of observations.

Overrides the base class to account for the extended solver force
output ``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements per
boundary instead of ``nDim``).

#### setup\_normalization

```python
def setup_normalization() -> None
```

Set up observation normalization.

Overrides the base class to handle the extended force vector
``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements per boundary).

For the ``&#x27;U_inf&#x27;`` strategy, free-stream values computed from the
isentropic relations in ``__init__`` are used:

* velocities scaled by ``U_T`` (total free-stream speed),
* density by ``rho_inf``,
* pressure by the dynamic pressure ``q_inf = ½ · rho_inf · U_T²``.

Force-slot entries use ``loc = 0, scale = 1`` since the solver
already normalises them by ``q_inf`` (forces) / ``q_inf·U_T`` (power).

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the unified reward: :math:`R = -C_D - \omega \cdot C_P`.

The solver returns ``[F_x, F_y, F_z, C_P, area]``.  Forces and
power are pre-normalised by ``q_∞`` and ``q_∞ · U_∞`` respectively;
dividing by ``area`` yields the final dimensionless coefficients::

C_D  = forces[0] / area
C_P  = forces[3] / area

For the jet environment ``C_P = 0`` and the reward reduces to
``-C_D``.

**Returns**:

  Tuple of ``(reward, info_dict)``.  ``info_dict`` contains
  ``&#x27;forces&#x27;``, ``&#x27;C_D&#x27;``, and ``&#x27;C_P&#x27;``.

## ZPGTBLJet Objects

```python
class ZPGTBLJet(ZPGTBLBase)
```

ZPG turbulent boundary layer with jet-based flow control.

Uses the same actuation setup as the :class:`~hydrogym.maia.envs.Cube`
environment: actions lie in ``[−MAX_CONTROL, +MAX_CONTROL]`` (scaled by
the base-class ``step()``) and are passed directly to the CFD solver.
The number of jet actuators is read from ``lbNoJets`` in the property file.

The solver returns ``[F_x, F_y, F_z, C_P, area]`` with ``C_P = 0.0``,
so the reward reduces to ``-C_D``.  Observation sizing, normalization,
and reward computation are all provided by :class:``3.

**Attributes**:

- ``4 _int_ - Number of jet boundary conditions.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Pass the (already ``MAX_CONTROL``-scaled) jet actuation to the solver.

**Arguments**:

- `action` - Action array scaled by ``MAX_CONTROL``.
  

**Returns**:

  Action array for the CFD solver.

## ZPGTBLSurfaceWave Objects

```python
class ZPGTBLSurfaceWave(ZPGTBLBase)
```

ZPG turbulent boundary layer with actuated traveling surface wave.

The surface is driven by a traveling wave with three strictly positive
control parameters:

============= =============================================
Parameter     Description
============= =============================================
amplitude     Wave amplitude
speed         Wave propagation speed
wavelength    Spatial period of the wave
============= =============================================

**Differences from** :class:`ZPGTBLJet`:

* **Action space** – Per-action ``[lower_bound, upper_bound]`` read from
``maia.action_lower_bounds`` / ``maia.action_upper_bounds`` in the
config.  The ``MAX_CONTROL`` scaling in ``step()`` is *not* applied.
* **Reset** – Uses ``[amplitude_init, speed_init, wavelength_init]`` from
config (zeros would crash the CFD solver).
* **Reward** – :math:``3 (shared with
:class:`ZPGTBLJet` via :class:``5; here ``C_P &gt; 0``
because the surface wave performs work on the fluid).

Required additional entries in the ``maia`` section of the YAML config:

.. code-block:: yaml

maia:
amplitude_init:      0.05
speed_init:          1.0
wavelength_init:     2.0
action_lower_bounds: [0.01, 0.1, 0.5]
action_upper_bounds: [0.2,  3.0, 10.0]
omega:               0.1

#### set\_observation\_action\_spaces

```python
def set_observation_action_spaces() -> None
```

Override to use a normalized ``[0, 1]`` action space per parameter.

The RL agent always operates in ``[0, 1]``; physical values are
recovered inside :meth:`convert_action` via the per-parameter affine
mapping ``lower + action * (upper - lower)``.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> List[float]
```

Scale a normalized ``[0, 1]`` action to physical wave parameters.

Applies the per-parameter affine map::

physical[i] = lower[i] + action[i] * (upper[i] - lower[i])

**Arguments**:

- `action` - Normalized action array with values in ``[0, 1]``,
  corresponding to ``[amplitude, speed, wavelength]``.
  

**Returns**:

  Physical actuation values for the CFD solver.

#### step

```python
def step(
    action: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

Advance the environment by one step.

Overrides the base class to skip ``MAX_CONTROL`` scaling.  The
incoming ``action`` is in ``[0, 1]`` and is converted to physical
units by :meth:`convert_action`.

**Arguments**:

- `action` - Normalized action in ``[0, 1]`` for
  ``[amplitude, speed, wavelength]``.
  

**Returns**:

  Tuple of ``(observation, reward, terminated, truncated, info)``.

#### reset

```python
def reset(seed: Optional[int] = None,
          options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]
```

Reset the environment with physically valid initial wave parameters.

Overrides the base-class reset to avoid setting zero control actions,
which would crash the CFD solver for the surface-wave boundary
condition.  ``init_action`` is passed in physical units directly,
bypassing the ``[0, 1]`` → physical scaling used during normal steps.

**Arguments**:

- `seed` - Optional random seed.
- `options` - Optional reset options (unused).
  

**Returns**:

  Tuple of ``(initial_observation, info)``.

