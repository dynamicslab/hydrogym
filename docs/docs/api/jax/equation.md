---
sidebar_label: equation
title: hydrogym.jax.equation
---

## Equation Objects

```python
class Equation()
```

Base class for a PDE right-hand-side defined on a state vector.

Subclasses implement the full right-hand side in :meth:`rhs`, either
directly or by splitting it into linear and nonlinear parts (see
:class:`SplitEquation`).

#### \_\_init\_\_

```python
def __init__(params)
```

Store the equation parameters.

**Arguments**:

- `params` - Solver/equation parameters (interpreted by subclasses).

#### rhs

```python
def rhs(state, control)
```

Evaluate the right-hand side of the equation.

**Arguments**:

- `state` - Current state vector.
- `control` - Control (actuation) input applied to the flow.
  

**Returns**:

  Time derivative of the state. Not implemented in the base class.

## SplitEquation Objects

```python
class SplitEquation(Equation)
```

Equation whose right-hand side is split into linear and nonlinear terms.

The split exists so implicit-explicit (IMEX) time integrators can treat
the linear (stiff, typically viscous) term implicitly via
:meth:`implicit_timestep` in subclasses, and the nonlinear term explicitly.

#### \_\_init\_\_

```python
def __init__(params)
```

See :class:`Equation`.

#### linear\_terms

```python
def linear_terms(state, control)
```

Evaluate the linear (implicitly treated) term of the equation.

**Arguments**:

- `state` - Current state vector.
- `control` - Control (actuation) input.
  

**Returns**:

  Linear contribution to the state derivative. Not implemented here.

#### nonlinear\_terms

```python
def nonlinear_terms(state, control)
```

Evaluate the nonlinear (explicitly treated) term of the equation.

**Arguments**:

- `state` - Current state vector.
- `control` - Control (actuation) input.
  

**Returns**:

  Nonlinear contribution to the state derivative. Not implemented here.

#### rhs

```python
def rhs(state, control)
```

Evaluate the full right-hand side as linear + nonlinear terms.

**Arguments**:

- `state` - Current state vector.
- `control` - Control (actuation) input.
  

**Returns**:

  Sum of the linear and nonlinear contributions.

#### forcing

```python
def forcing()
```

Evaluate any external forcing added to the right-hand side.

Not implemented in the base class.

## IMEXEquation Objects

```python
class IMEXEquation(SplitEquation)
```

Split equation supporting an implicit linear timestep.

Intended for IMEX schemes: the linear term is advanced implicitly via
:meth:`implicit_timestep` while the nonlinear term is handled explicitly
by the integrator.

#### \_\_init\_\_

```python
def __init__(params)
```

See :class:`SplitEquation`.

#### implicit\_timestep

```python
def implicit_timestep(state)
```

Advance the linear part of the equation implicitly by one timestep.

**Arguments**:

- `state` - Current state vector.
  

**Raises**:

- `NotImplementedError` - Always; subclasses must define the implicit solve.

