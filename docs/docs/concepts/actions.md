---
sidebar_position: 2
---

# Actions within an action interval

An action is applied for one action interval (`num_substeps` / `steps_per_action` solver steps). How the actuators
follow a new action differs between the backends:

| backend | within the interval |
|---|---|
| Firedrake | first-order low-pass of the commanded value, $x \leftarrow u + (x - u)\,e^{-\Delta t/\tau}$ at every solver step (`DampedActuator`; $\tau$ = `TAU` of the flow: cylinder 0.0556, pinball 0.05, cavity 0.075, step 0.005) |
| JAX `KolmogorovFlow` | the forcing of the action is held constant over the interval (a step at its start) |
| JAX `ChannelFlowSpectralEnv` | the jet gain ramps linearly from zero over the first `ramp_time` of every interval and back to zero over its last `ramp_time` (defaults 10 of 50); it is not an interpolation from the previous action to the new one |
| m-AIA | with `lbUseControlRamping = true` in the case's `properties_run.toml` (e.g. the cavity cases, `Pinball_3D_Re150` and the gust cases), a tanh ramp from the previous action to the new one; otherwise (the solver default) a step |
| Nek5000, NekRS | a step: the wall velocity of the action is applied from the first solver step of the interval |

The actuation power of an interval (`power_sampling = "mean"`) includes the work of the flow's response to each change
of the actuation; with a step this can be much larger than the value at the end of the interval (see
[Actuation power](actuation_power.md#accuracy)).
