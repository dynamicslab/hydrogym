---
sidebar_label: base
title: hydrogym.firedrake.solvers.base
---

## NewtonSolver Objects

```python
class NewtonSolver()
```

Newton solver for the steady-state Navier-Stokes equations.

Assembles the nonlinear steady residual of ``flow.steady_form`` (with
optional SUPG/GLS-type stabilization) and solves the resulting
nonlinear variational problem with Firedrake&#x27;s Newton solver.

#### \_\_init\_\_

```python
def __init__(flow: FlowConfig,
             stabilization: str = "none",
             solver_parameters: dict = `{}`)
```

Configure the steady solver.

**Arguments**:

- `flow` _FlowConfig_ - Flow configuration defining the mesh,
  function spaces, and boundary conditions.
- `stabilization` _str, optional_ - Stabilization method, one of
  the keys of ``ns_stabilization``. Default &quot;none&quot;.
- `solver_parameters` _dict, optional_ - Parameters passed through
  to ``fd.NonlinearVariationalSolver``.
  

**Raises**:

- `ValueError` - If ``stabilization`` is not a recognized type.

#### solve

```python
def solve(q: fd.Function = None)
```

Solve the steady-state problem from initial guess `q`

#### steady\_form

```python
def steady_form(q: fd.Function, q_test=None)
```

Assemble the nonlinear steady-state Navier-Stokes variational form.

Builds ``F(u, p; v, s)`` with the sign convention used for
steady solves (advection and stress terms positive), plus any
stabilization terms. This differs from ``FlowConfig.residual``,
whose signs are written for the transient problem.

**Arguments**:

- `q` _fd.Function_ - Mixed trial state (u, p).
- `q_test` _optional_ - Pair of test functions (v, s); defaults to
  the test functions of the flow&#x27;s mixed space.
  

**Returns**:

- `ufl.Form` - The nonlinear residual form.

## NavierStokesTransientSolver Objects

```python
class NavierStokesTransientSolver(TransientSolver)
```

Base class for transient Navier-Stokes solvers.

Extends ``TransientSolver`` with a hook-based reset: subclasses
allocate their fields and variational forms in
``initialize_functions`` and ``initialize_operators``, both of which
are invoked on construction and on every ``reset``.

#### \_\_init\_\_

```python
def __init__(flow: FlowConfig, dt: float = None, debug: bool = False)
```

Initialize the transient solver.

**Arguments**:

- `flow` _FlowConfig_ - Flow configuration to advance in time.
- `dt` _float, optional_ - Time step. Defaults to ``TransientSolver``&#x27;s
  handling (typically the flow&#x27;s ``DEFAULT_DT``).
- `debug` _bool, optional_ - Whether to enable debug output.
  Default False.
  

**Notes**:

  This class previously accepted ``eta``/``max_noise_iter``/
  ``noise_cutoff`` for a random white-noise body forcing, but the
  forcing was removed upstream (a067781, 2024-03) and those
  kwargs were silently ignored ever since; they were removed in
  this repo&#x27;s audit Task 2.3.

#### reset

```python
def reset()
```

Reset the solver to its initial condition and rebuild state.

Calls the parent reset, then re-runs ``initialize_functions`` and
``initialize_operators``.

#### initialize\_functions

```python
def initialize_functions()
```

Allocate solver-specific state fields.

No-op in the base class; subclasses should override this to create
the functions they need.

#### initialize\_operators

```python
def initialize_operators()
```

Allocate solver-specific variational forms and operators.

No-op in the base class; subclasses should override this to set up
their forms/operators.

