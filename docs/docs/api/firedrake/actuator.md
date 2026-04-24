---
sidebar_label: actuator
title: hydrogym.firedrake.actuator
---

## DampedActuator Objects

```python
class DampedActuator(ActuatorBase)
```

Simple damped actuator model.

Dynamics are given by the following ODE:

m * dx/dt = k * (u - x)

where x is the state of the actuator, u is the control input, k is the damping
coefficient, and m is the inertia.  Integrating over a time step `dt` with a zero-
order hold on `u` gives the following exact solution:

x(t + dt) = u + (x(t) - u) * exp(-k * dt / m)

Since only the ratio k/m enters the dynamics as a time scale tau = m/k, we can
think of the dynamics as a low-pass filter with a time constant tau.  The single
remaining parameter is named `damping`, and corresponds to k/m = 1/tau.

#### step

```python
def step(u: float, dt: float)
```

Update the state of the actuator

