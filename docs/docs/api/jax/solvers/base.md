---
sidebar_label: base
title: hydrogym.jax.solvers.base
---

## VelocityState Objects

```python
class VelocityState(NamedTuple)
```

Three-component velocity (or velocity-spectrum) state, one array per direction.

**Attributes**:

- `u` - x-component; physical or spectral depending on usage.
- `v` - y-component.
- `w` - z-component.

#### u

physical or spectral depending on usage

## RungeKuttaCrankNicolson Objects

```python
class RungeKuttaCrankNicolson(TransientSolver)
```

IMEX Crank-Nicolson Runge-Kutta transient solver for a split equation.

Advances an :class:`~hydrogym.jax.equation.IMEXEquation` with the nonlinear
terms treated explicitly and the linear terms treated implicitly, using a
low-storage 5-stage scheme; the whole rollout is expressed as a
``lax.scan`` so it can be JIT-compiled.

#### \_\_init\_\_

```python
def __init__(flow: PDEBase, dt: float, save_n: int, equation: IMEXEquation,
             **kwargs)
```

Initialize the solver.

**Arguments**:

- `flow` - Flow configuration providing the state (used by the base class).
- `dt` - Timestep size.
- `save_n` - Number of (inner) steps between saved states.
- `equation` - The split (IMEX) equation being integrated.
- `**kwargs` - Ignored; accepted for API compatibility.

#### RK4\_CN

```python
def RK4_CN(control_field=None)
```

Crank-Nicolson RK4 implicit-explicit time stepping scheme.
Low storage scheme inspired by [1]. Method described in [2].

Implicit-Explicit timestepping for an ODE of the form:
∂u/∂t = g(u,t) + l(u,t)
where g(u,t) is the nonlinear advection term and l(u,t) is the linear diffusion term.

[1] Kochkov, D., et. al. (2021) https://doi.org/10.1073/pnas.2101784118
[2] PK Sweby, (1984). SIAM journal on numerical analysis 21, Appendix D.

#### step

```python
def step(flow: PDEBase,
         dt: float,
         save_n: int,
         callbacks: Callable,
         control_field=None)
```

Build a ``lax.scan`` function that advances a state by ``save_n`` timesteps.

Note this returns the *scan function*, not a stepped state: the caller
nests it inside an outer scan (see :meth:`solve`).

**Arguments**:

- `flow` - Flow configuration (unused here; the equation carries the
  dynamics).
- `dt` - Timestep size (unused here; the one fixed at construction
  applies).
- `save_n` - Number of timesteps each inner scan performs.
- `callbacks` - Unused; per-step callback tracking is not possible
  inside compiled scans.
- `control_field` - Optional control input forwarded to the equation&#x27;s
  nonlinear terms at every step.
  

**Returns**:

  Callable mapping an initial state to the state ``save_n``
  timesteps later.

#### solve

```python
def solve(dt: float,
          flow: PDEBase,
          t_span: Tuple[float, float],
          callbacks: Iterable[CallbackBase] = [],
          controller: Callable = None,
          save_n: int = 1,
          initial_state=None,
          control_field=None) -> PDEBase
```

Integrate the equation from t=0 to ``t_span[1]`` with nested lax scans.

The rollout is run as an outer scan of inner scans, each of ``save_n // dt``
timesteps, so the saved trajectory holds one state per ``save_n`` time
units. Callbacks are invoked once at the end (per-iteration callback
tracking is not possible through the compiled scans).

**Arguments**:

- `dt` - Timestep size.
- `flow` - Flow configuration; also supplies the initial state when
  ``initial_state`` is None.
- ``0 - ``(t0, t1)`` integration interval; ``t1`` must be at least 1.
- ``5 - Callbacks invoked on the flow after the rollout.
- ``6 - Unused; accepted for API compatibility with the
  hydrogym solver interface.
- ``7 - Time interval between saved trajectory states.
- ``8 - Optional starting state (FFT vorticity field);
  defaults to ``flow.initialize_state()``.
- ``1 - Optional control input forwarded to the equation&#x27;s
  nonlinear terms at every step.
  

**Returns**:

  Tuple ``(final_state, outputs)`` where ``outputs`` is the stacked
  trajectory of states at each outer-scan step (also stored on
  ``flow.vorticity``).
  

**Raises**:

- ``8 - If the end time in ``t_span`` is less than 1.

## RungeKutta4 Objects

```python
class RungeKutta4()
```

Explicit classical 4th-order Runge-Kutta stepper for a velocity state.

Each step evaluates the equation&#x27;s full right-hand side four times,
projects the result back onto the constraint manifold
(``equation.project``), and optionally applies a constant-mass-flux
correction by rescaling the mean streamwise velocity in physical space
before re-applying the boundary conditions.

#### \_\_init\_\_

```python
def __init__(equation, dt: float, save_n: int, **kwargs)
```

Initialize the integrator.

**Arguments**:

- `equation` - Equation object providing ``rhs``, ``project``,
  ``to_physical``, ``to_spectral`` and ``enforce_noslip``.
- ``1 - Default timestep used when ``rk4_step`` gets no explicit ``dt``.
- ``6 - Number of steps between saves (stored; not used by
  ``rk4_step`` itself).
- ``9 - Ignored; accepted for API compatibility.

#### rk4\_step

```python
def rk4_step(state_hat: "VelocityState",
             dt: float = None,
             action=None,
             t: float = 0.0,
             fx=0.0,
             fy=0.0,
             fz=0.0,
             enforce_const_massflux=True,
             target_bulk_u=8.0)
```

Advance the state by one RK4 step, then project and correct mass flux.

**Arguments**:

- `state_hat` - Current state (spectral velocity components).
- `dt` - Timestep; defaults to the value given at construction.
- `action` - Control (actuation) input forwarded to the equation&#x27;s rhs,
  projection, and boundary-condition enforcement.
- `t` - Current time, used for time-dependent BCs and forcing.
- `fx` - x-direction body forcing.
- `fy` - y-direction body forcing.
- `fz` - z-direction body forcing.
- `enforce_const_massflux` - If True, rescale the streamwise velocity in
  physical space so the bulk velocity equals ``target_bulk_u``,
  then re-apply the no-slip/jet boundary conditions.
- `dt`0 - Target bulk (domain-mean) streamwise velocity for
  the mass-flux correction.
  

**Returns**:

  The new state in spectral form.

