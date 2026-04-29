---
sidebar_label: nozzle
title: hydrogym.jaxfluids.envs.nozzle
---

Nozzle Environments for Shock Vector Control
=============================================

This module implements 2D and 3D nozzle environments for thrust vector control (TVC)
via secondary injection (shock vector control). Both environments expose a Gymnasium
action space of injector pressure ratios and return observations containing the current
and target thrust angles, with optional wall-pressure probes.

## NozzleBase Objects

```python
class NozzleBase(JAXFluidsFlowEnv)
```

Abstract base class for nozzle TVC environments.

Subclasses must define the class variables `SPEC` and `TARGET_FNS` and implement
the abstract methods `_compute_probe_locations`, `_get_reward`, `_plot_flow_field`,
and `_plot_observations`.

**Class variables**:

- `SPEC` (`TVCSpec`) - Environment specification (dimension, actuator count, geometry, …).
- `TARGET_FNS` (`dict[str, TargetThrustAngleFn]`) - Named target-angle functions keyed by string.

**Attributes**:

- `num_actuators` - Number of secondary-injection actuators.
- `secondary_pressure_ratio` - Injector secondary-to-nozzle pressure ratio (0.7–0.9).
- `resolution` - Spatial grid resolution (`'coarse'` or `'fine'`).
- `target_fn` - Active target-angle callable selected via `env_config['target_fn']`.
- `is_pressure_probes` - Whether wall-pressure probes are included in the observation.
- `is_scale_observations` - Whether thrust angles and pressures are normalised.
- `injector_geometry` (`InjectorGeometry`) - Position, width and count of the injectors.
- `pressure_ratios` (`PressureRatios`) - Nozzle and secondary pressure ratios.

#### \_\_init\_\_

```python
def __init__(env_config: dict) -> None
```

Initialise the nozzle environment from a configuration dictionary.

Calls `_init_from_hf` to download environment data, then constructs the
JAXFluids simulation and Gymnasium spaces.

**Arguments**:

- `env_config` - Configuration dictionary. Keys consumed by this class:
  - `secondary_pressure_ratio` (float): Default `0.7`. Must be in [0.7, 0.9].
  - `resolution` (str): `'coarse'` or `'fine'`. Default `'fine'`.
  - `ngpus` (int): Number of GPUs. Default `1`.
  - `is_pressure_probes` (bool): Include pressure-probe observations. Default `False`.
  - `is_scale_observations` (bool): Normalise observations. Default `True`.
  - `target_fn` (str): Key in `TARGET_FNS`. Default `'sine'`.
  - Plus any keys required by `JAXFluidsFlowEnv._init_from_hf`.

#### compute\_obs

```python
def compute_obs(jxf_buffers: JaxFluidsBuffers) -> ObsData
```

Compute the current observation from simulation state.

Dispatches to `_compute_obs` via `jax.pmap` in multi-GPU mode or
`jax.jit` in single-device mode.

**Arguments**:

- `jxf_buffers` - Current JAXFluids simulation buffers.

**Returns**:

  `ObsData` named tuple with `thrust_angle`, `target_angle`, and optionally `pressure_probes`.

#### compute\_thrust\_angle

```python
def compute_thrust_angle(jxf_buffers: JaxFluidsBuffers) -> Array
```

Compute the current thrust vector angle from nozzle exit fluxes.

Returns a scalar (2-D) or a 2-element array of (pitch, yaw) angles (3-D),
both in radians.

**Arguments**:

- `jxf_buffers` - Current JAXFluids simulation buffers.

**Returns**:

  Thrust angle in radians. Shape `()` for 2D, `(2,)` for 3D.

#### compute\_pressure\_probes

```python
def compute_pressure_probes(jxf_buffers: JaxFluidsBuffers) -> Array
```

Sample static pressure at the pre-computed probe locations.

**Arguments**:

- `jxf_buffers` - Current JAXFluids simulation buffers.

**Returns**:

  Pressure values at each probe, shape `(num_probes,)`.

#### render

```python
def render() -> None
```

Render the current environment state.

Calls `_plot_flow_field` and `_plot_observations` according to the active
`render_mode`. No-ops when `render_mode` is `None`.

## Nozzle2D Objects

```python
class Nozzle2D(NozzleBase)
```

2D nozzle environment for shock vector control.

Two fixed actuators are placed on the upper and lower nozzle walls.
The action space is `Box(0, 1, shape=(2,))` (injector pressure ratios).
The observation contains the scalar thrust angle and its target (radians or
normalised to [-1, 1]) and optionally four wall-pressure probes.

**Configuration keys** (passed as `env_config`):

- `secondary_pressure_ratio` (float): Default `0.7`.
- `resolution` (str): `'coarse'` or `'fine'`. Default `'fine'`.
- `ngpus` (int): Default `1`.
- `is_pressure_probes` (bool): Default `False`.
- `is_scale_observations` (bool): Default `True`.
- `target_fn` (str): `'sine'` or `'step'`. Default `'sine'`.

**Target functions**:

- `'sine'` — sinusoidal target with amplitude 10° and period 4 ms, onset at 0.5 ms.
- `'step'` — step to 5° at 0.5 ms.

## Nozzle3D Objects

```python
class Nozzle3D(NozzleBase)
```

3D nozzle environment for shock vector control.

The number of actuators is user-configurable (4–12) and distributed uniformly
around the nozzle circumference.
The action space is `Box(0, 1, shape=(num_actuators,))`.
The observation contains pitch and yaw thrust angles (radians or normalised)
and optionally 12 wall-pressure probes.

**Configuration keys** (passed as `env_config`):

- `num_actuators` (int): Required. Must be in [4, 12].
- `secondary_pressure_ratio` (float): Default `0.7`.
- `resolution` (str): `'coarse'` or `'fine'`. Default `'fine'`.
- `ngpus` (int): Default `1`.
- `is_pressure_probes` (bool): Default `False`.
- `is_scale_observations` (bool): Default `True`.
- `target_fn` (str): `'sine'` or `'step'`. Default `'sine'`.

**Target functions**:

- `'sine'` — sinusoidal pitch with amplitude 10° and period 4 ms, onset at 1 ms; yaw held at 0°.
- `'step'` — step pitch to 5° at 1 ms; yaw held at 0°.
