---
sidebar_label: nozzle
title: hydrogym.jaxfluids.envs.nozzle
---

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
  - resolution: Optional. Spatial resolution of the environment. Choose either &#x27;coarse&#x27; or &#x27;fine&#x27;.
  - ngpus: Optional. Number of GPUs for running the environment.
  - is_pressure_probes: Optional. Boolean indicating whether pressure probes
  are part of the observation
  - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
  - target_fn: Optional. Target thrust vector function. Choose either &#x27;sine&#x27; or &#x27;step&#x27;.

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
  - resolution: Optional. Spatial resolution of the environment. Choose either &#x27;coarse&#x27; or &#x27;fine&#x27;.
  - ngpus: Optional. Number of GPUs for running the environment.
  - is_pressure_probes: Optional. Boolean indicating whether pressure probes
  are part of the observation
  - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
  - target_fn: Optional. Target thrust vector function. Choose either &#x27;sine&#x27; or &#x27;step&#x27;.

