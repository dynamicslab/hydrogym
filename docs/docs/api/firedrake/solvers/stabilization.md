---
sidebar_label: stabilization
title: hydrogym.firedrake.solvers.stabilization
---

## NavierStokesStabilization Objects

```python
@dataclasses.dataclass
class NavierStokesStabilization(metaclass=abc.ABCMeta)
```

Base class (and &quot;no stabilization&quot; implementation) for Navier-Stokes stabilization schemes.

Bundles everything the residual-based stabilization terms need from the
calling solver; subclasses build the actual UFL stabilization forms in
:meth:`stabilize`. This base class implements the identity map, i.e. the
unstabilized Galerkin formulation (the ``&quot;none&quot;``/``&quot;linearized_none&quot;``
options in ``ns_stabilization``).

**Attributes**:

- `flow` - Flow configuration (mesh, viscosity, stress/strain operators).
- `q_trial` - ``(u, p)`` pair of trial functions on the mixed space.
- ``1 - ``(v, s)`` pair of test functions on the mixed space.
- ``4 - Velocity field used as the convective wind (the extrapolated
  velocity for the nonlinear solvers, or the base flow for the
  linearized ones).
- ``5 - Timestep (float or ``fd.Constant``); enters the stabilization
  parameter ``tau_M`` when given.
- ``0 - UFL expression for the time derivative estimate (the BDF term),
  included in the strong-form residual when given.
- ``1 - Body forcing, subtracted from the strong-form residual when given.

#### stabilize

```python
def stabilize(weak_form)
```

Return the weak form unchanged (no stabilization terms).

**Arguments**:

- `weak_form` - The unstabilized UFL weak form.
  

**Returns**:

  The same weak form, unmodified.

## UpwindNSStabilization Objects

```python
class UpwindNSStabilization(NavierStokesStabilization)
```

Residual-based upwind stabilization for the (unlinearized) Navier-Stokes equations.

Shared machinery for SUPG and GLS: computes the strong-form momentum
residual ``Lu`` with the given wind, the stabilization parameters
``tau_M``/``tau_C`` from the element size, wind magnitude, viscosity and
(optionally) timestep, and adds ``tau_M &lt;Lu, Lv&gt;`` plus a least-squares
incompressibility (LSIC) term ``tau_C &lt;div u, div v&gt;`` to the weak form.
Subclasses define ``Lv``, the residual operator applied to the test
functions.

Attributes (inherited from :class:``2) supply the
trial/test functions, wind, timestep, time derivative and forcing.

#### h

```python
@property
def h()
```

Mesh cell size (UFL ``CellSize``), the length scale in ``tau_M``.

#### Lu

```python
@property
def Lu()
```

Strong-form momentum residual of the trial functions.

``w . grad(u) - div(sigma(u,p))``, plus the time-derivative estimate
``u_t`` and minus the forcing ``f`` when those are supplied.

**Returns**:

  UFL expression for ``Lu`` (the momentum residual).

#### Lv

```python
@abc.abstractproperty
def Lv()
```

Residual operator applied to the test functions (subclass-specific).

**Returns**:

  UFL expression for ``Lv``, paired with ``Lu`` in the
  stabilization inner product.

#### tau\_M

```python
@property
def tau_M()
```

Stabilization parameter for the momentum residual.

``tau_M = (4|w|^2/h^2 + 9 (4 nu / h^2)^2 [+ 4/dt^2])^(-1/2)``, i.e.
the inverse squared &quot;elemental&quot; advective/diffusive/temporal rates.
Based on:
https://github.com/florianwechsung/alfi/blob/master/alfi/stabilisation.py

**Returns**:

  UFL expression for ``tau_M``.

#### tau\_C

```python
@property
def tau_C()
```

Stabilization parameter for the continuity residual, ``h^2 / tau_M``.

**Returns**:

  UFL expression for ``tau_C``.

#### momentum\_stabilization

```python
@property
def momentum_stabilization()
```

Momentum stabilization term ``tau_M &lt;Lu, Lv&gt; dx``.

**Returns**:

  UFL form to be added to the weak form.

#### lsic\_stabilization

```python
@property
def lsic_stabilization()
```

Least-squares incompressibility (LSIC) term ``tau_C &lt;div u, div v&gt; dx``.

**Returns**:

  UFL form to be added to the weak form.

#### stabilize

```python
def stabilize(weak_form)
```

Add the momentum and LSIC stabilization terms to a weak form.

**Arguments**:

- `weak_form` - The unstabilized UFL weak form.
  

**Returns**:

  The weak form with ``momentum_stabilization`` and
  ``lsic_stabilization`` appended.

## SUPG Objects

```python
class SUPG(UpwindNSStabilization)
```

Streamline-Upwind/Petrov-Galerkin stabilization.

Uses only the streamline derivative ``w . grad(v)`` of the velocity test
function as the residual operator ``Lv``, so the stabilization acts
along the flow direction.

#### Lv

```python
@property
def Lv()
```

Streamline derivative of the velocity test function, ``w . grad(v)``.

**Returns**:

  UFL expression for ``Lv``.

## GLS Objects

```python
class GLS(UpwindNSStabilization)
```

Galerkin least-squares stabilization.

Uses the full momentum operator applied to the test functions,
``Lv = w . grad(v) - div(sigma(v, s))``, so the least-squares term
minimizes the complete momentum residual (not just its streamline
component, as SUPG does).

#### Lv

```python
@property
def Lv()
```

Full momentum operator applied to the test functions.

``w . grad(v) - div(sigma(v, s))``.

**Returns**:

  UFL expression for ``Lv``.

## LinearizedNSStabilization Objects

```python
class LinearizedNSStabilization(UpwindNSStabilization)
```

Residual-based stabilization for the Navier-Stokes equations linearized about a base flow.

Identical machinery to :class:`UpwindNSStabilization`, except the
strong-form momentum residual ``Lu`` uses the linearized convective
operator ``uB . grad(u) + u . grad(uB)``, where the base flow ``uB`` is
supplied as the ``wind``.

#### Lu

```python
@property
def Lu()
```

Strong-form momentum residual of the linearized operator.

``uB . grad(u) + u . grad(uB) - div(sigma(u,p))``, plus the
time-derivative estimate ``u_t`` and minus the forcing ``f`` when
those are supplied.

**Returns**:

  UFL expression for the linearized momentum residual ``Lu``.

## LinearizedSUPG Objects

```python
class LinearizedSUPG(LinearizedNSStabilization)
```

Streamline-upwind stabilization of the base-flow-linearized equations.

The SUPG residual operator about the base flow: both linearized
convective terms applied to the test function.

#### Lv

```python
@property
def Lv()
```

Linearized streamline derivative of the test function.

``uB . grad(v) + v . grad(uB)``.

**Returns**:

  UFL expression for ``Lv``.

## LinearizedGLS Objects

```python
class LinearizedGLS(LinearizedNSStabilization)
```

Galerkin least-squares stabilization of the base-flow-linearized equations.

The full linearized momentum operator applied to the test functions
(linearized convective terms plus the viscous/pressure operator).

#### Lv

```python
@property
def Lv()
```

Full linearized momentum operator applied to the test functions.

``uB . grad(v) + v . grad(uB) - div(sigma(v, s))``.

**Returns**:

  UFL expression for ``Lv``.

