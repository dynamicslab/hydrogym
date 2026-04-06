---
sidebar_label: base
title: hydrogym.jax.solvers.base
---

## VelocityState Objects

```python
class VelocityState(NamedTuple)
```

#### u

physical or spectral depending on usage

## RungeKuttaCrankNicolson Objects

```python
class RungeKuttaCrankNicolson(TransientSolver)
```

#### RK4\_CN

```python
def RK4_CN()
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
def step(flow: FlowConfig, dt: float, save_n: int, callbacks: Callable)
```

Lax.scan to iteratively apply a function given an initial value

**Arguments**:

  initialization(grid array): the initial fft vorticity field
- `steps` _int_ - number of timesteps
- `save_n` _int_ - save every n steps
- `ignore_intermediate_steps` _bool_ - if saving every n steps, ignore intermediate steps.
  this drastically reduces the memory requirements.

