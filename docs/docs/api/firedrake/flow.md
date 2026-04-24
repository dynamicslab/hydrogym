---
sidebar_label: flow
title: hydrogym.firedrake.flow
---

## FlowConfig Objects

```python
class FlowConfig(PDEBase)
```

#### DEFAULT\_VELOCITY\_ORDER

Taylor-Hood elements

#### FUNCTIONS

tuple of functions necessary for the flow

#### set\_state

```python
def set_state(q: fd.Function)
```

Set the current state fields

**Arguments**:

- `q` _fd.Function_ - State to be assigned

#### copy\_state

```python
def copy_state(deepcopy: bool = True) -> fd.Function
```

Return a copy of the current state fields

**Returns**:

- `q` _fd.Function_ - copy of the flow state

#### create\_actuator

```python
def create_actuator(tau=None) -> ActuatorBase
```

Create a single actuator for this flow

#### reset\_controls

```python
def reset_controls(function_spaces=None)
```

Reset the controls to a zero state

Note that this is broken out from `reset` because
the two are not necessarily called together (e.g.
for linearization or deriving the control vector)

TODO: Allow for different kinds of actuators

#### vorticity

```python
def vorticity(u: fd.Function = None) -> fd.Function
```

Compute the vorticity field `curl(u)` of the flow

**Arguments**:

  u (fd.Function, optional):
  If given, compute the vorticity of this velocity
  field rather than the current state.
  

**Returns**:

- `fd.Function` - vorticity field

#### function\_spaces

```python
def function_spaces(mixed: bool = True)
```

Function spaces for velocity and pressure

**Arguments**:

  mixed (bool, optional):
  If True (default), return subspaces of the mixed velocity/pressure
  space. Otherwise return the segregated velocity and pressure spaces.
  

**Returns**:

  Tuple[fd.FunctionSpace, fd.FunctionSpace]: Velocity and pressure spaces

#### collect\_bcu

```python
def collect_bcu() -> Iterable[fd.DirichletBC]
```

List of velocity boundary conditions

#### collect\_bcp

```python
def collect_bcp() -> Iterable[fd.DirichletBC]
```

List of pressure boundary conditions

#### collect\_bcs

```python
def collect_bcs() -> Iterable[fd.DirichletBC]
```

List of all boundary conditions

#### epsilon

```python
def epsilon(u) -> ufl.Form
```

Symmetric gradient (strain) tensor

#### sigma

```python
def sigma(u, p) -> ufl.Form
```

Newtonian stress tensor

#### residual

```python
def residual(q, q_test=None)
```

Nonlinear residual for the incompressible Navier-Stokes equations.

Returns a UFL form F(u, p, v, s) = 0, where (u, p) is the trial function
and (v, s) is the test function.  This residual is also the right-hand side
of the unsteady equations.

A linearized form can be constructed by calling:
```
F = flow.residual((uB, pB), (v, s))
J = fd.derivative(F, qB, q_trial)
```

#### max\_cfl

```python
@pyadjoint.no_annotations
def max_cfl(dt) -> float
```

Estimate of maximum CFL number

#### linearize\_bcs

```python
def linearize_bcs()
```

Sets the boundary conditions appropriately for linearized flow

#### set\_control

```python
def set_control(act: ArrayLike = None)
```

Directly sets the control state

Note that for time-varying controls it will be better to adjust the controls
in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
to change control for a steady-state solve, for instance, and is also used
internally to compute the control matrix

#### inner\_product

```python
def inner_product(q1: fd.Function,
                  q2: fd.Function,
                  assemble=True,
                  augmented=False)
```

Energy inner product for the Navier-Stokes equations.

`augmented` is used to specify whether the function space is
extended to represent complex numbers. In this case the inner
product is the L2 norm of the real and imaginary parts.

#### velocity\_probe

```python
def velocity_probe(probes, q: fd.Function = None) -> list[float]
```

Probe velocity in the wake.

Returns a list of velocities at the probe locations, ordered as
(u1, u2, ..., uN, v1, v2, ..., vN) where N is the number of probes.

#### pressure\_probe

```python
def pressure_probe(probes, q: fd.Function = None) -> list[float]
```

Probe pressure around the cylinder

#### vorticity\_probe

```python
def vorticity_probe(probes, q: fd.Function = None) -> list[float]
```

Probe vorticity in the wake.

