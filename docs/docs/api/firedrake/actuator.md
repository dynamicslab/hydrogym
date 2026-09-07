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

#### \_\_init\_\_

```python
def __init__(damping: float, state: float = 0.0)
```

Initialize the actuator.

**Arguments**:

- `damping` _float_ - Damping coefficient ``k/m = 1/tau``, i.e. the
  inverse time constant of the low-pass filter.
- `state` _float, optional_ - Initial actuator state. Default 0.0.

#### state

```python
@property
def state() -> np.ndarray
```

Current actuator state as a scalar value.

#### state

```python
@state.setter
def state(u: float)
```

Set the actuator state directly.

**Arguments**:

- `u` _float_ - New state value.

#### step

```python
def step(u: float, dt: float)
```

Advance the actuator state by ``dt`` with control input ``u``.

Applies the exact zero-order-hold solution of the first-order
damped dynamics: ``x &lt;- u + (x - u) * exp(-alpha * dt)``, and
annotates the update for use with Firedrake/Pyadjoint
differentiable programming.

**Arguments**:

- `u` _float_ - Control input (target state) held over the step.
- `dt` _float_ - Time step size.

