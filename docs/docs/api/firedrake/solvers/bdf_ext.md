---
sidebar_label: bdf_ext
title: hydrogym.firedrake.solvers.bdf_ext
---

## SemiImplicitBDF Objects

```python
class SemiImplicitBDF(NavierStokesTransientSolver)
```

Semi-implicit BDF transient solver for the incompressible Navier-Stokes equations.

Uses a backward-differentiation formula of order `k` for the time
derivative (treated implicitly, together with the viscous and pressure
terms) and a k-th-order extrapolation of the velocity for the convective
term, which is thereby treated explicitly. The result is a sequence of
linear variational problems, one per BDF order, solved each timestep with
Firedrake's `LinearVariationalSolver`.

For `k > 1` the first `k - 1` timesteps are taken with lower-order
startup solvers (order 1 .. k-1, each with a matching extrapolation),
since the full-order scheme needs `k` previous solutions: step `iter`
uses order `min(iter + 1, k)`. `legacy_startup_order=True` reproduces
the start-up order sequence of HydroGym &lt;= 1.0 (orders 2, 1, 2, 3, 3, ... for
`k = 3`; 1, 1, 2, ... for `k = 2`).

**Attributes**:

- `k` - Order of the BDF/extrapolation scheme (1, 2, or 3).
- `ksp_rtol` - Relative tolerance for the Krylov solver.
- `custom_solver_parameters` - User-supplied PETSc solver parameters,
  overriding the built-in defaults (see `_make_petsc_solver`).
- `stabilization` - Name of the stabilization scheme (a key of
  `ns_stabilization`);
  `"default"` resolves to `flow.DEFAULT_STABILIZATION`.
- `u_prev` - List of the `k` most recent velocity solutions (oldest
  first) used to build the BDF and extrapolation combinations.

#### \_\_init\_\_

```python
def __init__(flow: FlowConfig,
             dt: float = None,
             order: int = 3,
             stabilization: str = "default",
             rtol=1e-6,
             solver_parameters: dict = None,
             legacy_startup_order: bool = False,
             **kwargs)
```

Initialize the solver and build the BDF/startup operators.

**Arguments**:

- `flow` - Flow configuration (mesh, mixed space, BCs, forcing).
- `dt` - Timestep size. Optional; defaults to the flow's
  `DEFAULT_DT` if not given (see
  `NavierStokesTransientSolver`/`TransientSolver`).
- `order` - Order of the BDF/extrapolation scheme (1-3).
- `stabilization` - Stabilization type; `"default"` resolves to the
  flow's `DEFAULT_STABILIZATION`. See `ns_stabilization` for
  the available keys (e.g. "none", "supg", "gls", and their
  "linearized_" variants).
- `rtol` - Relative tolerance for the Krylov (KSP) solver.
- `solver_parameters` - Optional PETSc solver parameters replacing the
  built-in defaults chosen in `_make_petsc_solver`.
- `legacy_startup_order` - Reproduce the start-up order sequence of
  HydroGym &lt;= 1.0 (`startup_solvers[iter - 1]`, full order
  only from `iter = k`). Default False: order
  `min(iter + 1, k)`.
- `**kwargs` - Forwarded to
  `NavierStokesTransientSolver`.
  

**Raises**:

- `ValueError` - If `stabilization` is not a recognized type.

#### initialize\_functions

```python
def initialize_functions()
```

Allocate trial/test functions and the BDF history of previous solutions.

Sets `self.q_trial`/`self.q_test` (velocity-pressure pairs on the
mixed space), aliases the body force `self.f` from the flow
configuration, and creates `k` copies of the current velocity in
`self.u_prev` so the first (startup) steps have valid history.

#### initialize\_operators

```python
def initialize_operators()
```

Build the main order-`k` solver and the lower-order startup solvers.

Initializes the flow's boundary conditions first, then constructs the
full-order BDF solver and, if `k > 1`, one solver per order
`1 .. k-1` for the startup timesteps (`self.startup_solvers`).

#### step

```python
def step(iter, control=None)
```

Advance the flow by one timestep.

The flow's time is advanced (which also applies any actuation
scaling), the appropriate linear problem is solved — the full-order
BDF solver once `iter` exceeds `k - 1`, otherwise the matching
lower-order startup solver — and the velocity history is shifted so
the newest solution enters `u_prev[0]`.

**Arguments**:

- `iter` - Timestep index within the solve (0-based); the first
  `k - 1` iterations use the startup solvers.
- `control` - Optional actuation value(s) forwarded to
  `flow.advance_time` to scale the actuation BCs.
  

**Returns**:

- `FlowConfig` - The updated flow configuration.

## LinearizedBDF Objects

```python
class LinearizedBDF(SemiImplicitBDF)
```

Semi-implicit BDF solver for the Navier-Stokes equations linearized about a base flow.

Instead of the full convective term, solves the linearized form

`du/dt + uB . grad(u) + u . grad(uB) - div(sigma(u,p)) = f`

around the base flow `qB` supplied at construction, with the flow's
boundary conditions linearized (`flow.linearize_bcs()`) and the
base-flow velocity `uB` acting as the "wind" in the stabilization
terms. Intended for adjoint/transient-growth-type analyses about a known
(typically steady) state.

**Attributes**:

- `qB` - Base flow (mixed velocity-pressure `fd.Function`) to
  linearize about.

#### \_\_init\_\_

```python
def __init__(*args, qB: fd.Function, **kwargs)
```

Initialize the linearized solver.

**Arguments**:

- `*args` - Positional arguments forwarded to
  `SemiImplicitBDF` (`flow`, `dt`, ...).
- `qB` - Base flow (mixed velocity-pressure function) to linearize
  the equations about.
- `**kwargs` - Keyword arguments forwarded to `SemiImplicitBDF`.
  `stabilization` defaults to `"none"` (resolved to
  `"linearized_none"`) rather than the unlinearized default;
  a plain name (e.g. `"supg"`) is automatically prefixed with
  `"linearized_"`.

