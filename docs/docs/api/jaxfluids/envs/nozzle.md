---
sidebar_label: nozzle
title: hydrogym.jaxfluids.envs.nozzle
---

## NozzleBase Objects

```python
class NozzleBase(JAXFluidsFlowEnv)
```

Thrust-vector control of a supersonic nozzle by secondary injection (shock vector control).

Actuation power (one actuator per injector): the injected gas comes from a reservoir at total
temperature `T_inf` and the injector total pressure `p_t` set by the action; its power is the exergy
flow relative to the ambient state, `P_i = sum_cells mdot R T_inf ln(p_t / p_inf)` (minimum, isothermal,
supply work; in the low-Mach limit the incompressible transpiration power `Q (p_t - p_inf)`), with the
injected mass flux `mdot` of the interface-flux function (forms per interface cell). The thrust-angle
objective has no force coefficient, so `C_P = P / P_ref` with `P_ref` the exergy flow rate of the ideal
choked primary jet, `mdot* R T0 ln(NPR)` (2D: per unit span, `A* = 2 R_throat`; 3D: `pi R_throat^2`),
see `primary_reference_power`. No `drag_coefficient`.
Sampling: `"mean"` (default, after every JAX-Fluids step, via a callback) or `"end"`.
Power options (`power_weight`, `power_cost`, `power_sampling`) are read from the `jaxfluids`
section of the environment config and can be overridden in `env_config`.

## Nozzle2D Objects

```python
class Nozzle2D(NozzleBase)
```

Nozzle2D environment for shock vector control.

For the 2D nozzle, two actuators are present: one located on the upper
side of the nozzle, and one located at the lower side of the nozzle.

**Arguments**:

  - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
  Defaults to 0.7.
  - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
  - ngpus: Optional. Number of GPUs for running the environment.
  - is_pressure_probes: Optional. Boolean indicating whether pressure probes
  are part of the observation
  - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
  - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.

## Nozzle3D Objects

```python
class Nozzle3D(NozzleBase)
```

Nozzle3D environment for shock vector control.

For the 3D nozzle, the number of actuators can be set by the user. The minimum number of
actuators is 4, and the maximum number is 12. The actuators are distributed uniformly over
the diameter.

**Arguments**:

  - num_actuators: Required. Integer number of actuators. Must be between 4 and 12.
  - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
  Defaults to 0.7.
  - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
  - ngpus: Optional. Number of GPUs for running the environment.
  - is_pressure_probes: Optional. Boolean indicating whether pressure probes
  are part of the observation
  - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
  - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.

