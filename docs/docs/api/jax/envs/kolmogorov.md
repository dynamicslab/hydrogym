---
sidebar_label: kolmogorov
title: hydrogym.jax.envs.kolmogorov
---

## FlowConfig Objects

```python
class FlowConfig(PDEBase)
```

#### DEFAULT\_OBS\_SIZE

This correlates to a total observation size of 8x8 = 64.

#### load\_mesh

```python
def load_mesh(name)
```

Create jax grid given the desired dimensions and spacing in real space

**Returns**:

  jax meshgrid

#### load\_fft\_mesh

```python
def load_fft_mesh()
```

Create jax grid given desired dimensions and spacing in real Fourier space

**Returns**:

  jax meshgrid

#### initialize\_state

```python
def initialize_state()
```

Generate a divergence free velocity field to initialize the state
Initializing with divergence free field specified with the following stream function:

φ(x,y) = sin(x)cos(y)

**Returns**:

  fft vorticity field

#### forcing\_function

```python
def forcing_function(k, x, y)
```

Sinusoidal forcing function that drives the Kolmogorov flow.

**Arguments**:

- `k` _int_ - forcing wavenumber
- `x` _jnp.array_ - spatial coordinates in x
- `y` _jnp.array_ - spatial coordinates in y
  

**Returns**:

- `tuple` - forcing function in (x,y)

#### evaluate\_objective

```python
def evaluate_objective()
```

Return a copy of the flow state

#### num\_inputs

```python
@property
def num_inputs() -> int
```

Length of the control vector (number of actuators)

#### num\_outputs

```python
@property
def num_outputs() -> int
```

Number of scalar observed variables

#### save\_checkpoint

```python
def save_checkpoint()
```

Set up mesh, function spaces, state vector, etc

#### init\_bcs

```python
def init_bcs()
```

Initialize any boundary conditions for the PDE.

#### copy\_state

```python
def copy_state(deepcopy=True)
```

Return a copy of the flow state

#### render

```python
def render(**kwargs)
```

Plot the current PDE state (called by `gym.Env`)

## PseudoSpectralNavierStokes2D Objects

```python
class PseudoSpectralNavierStokes2D(IMEXEquation)
```

Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver.
We transform the 2D Navier-Stokes equation to a vorticity equation:
∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
ω = - ∇²φ ;
and solve in Fourier space

#### linear\_terms

```python
def linear_terms(omega_hat)
```

Computes the linear (viscous) term of the vorticity equation

#### implicit\_timestep

```python
def implicit_timestep(omega_hat, time_step)
```

Function that computes an implicit euler timestep,
y_n+1 = y_n / (1-∇tλ).

#### nonlinear\_terms

```python
def nonlinear_terms(omega_hat)
```

Computes the explicit (nonlinear) terms in the vorticity equation.
Uses the stream function to compute velocity components in Fourier space.

**Arguments**:

- `omega_hat` - fft of vorticity
  

**Returns**:

- `terms` - Nonlinear terms of the equation.

#### control\_term

```python
def control_term(omega_hat)
```

Computes the user-specified forcing term of the vorticity equation

**Arguments**:

- `omega_hat` - Fourier transformed vorticity term
- `forcing` - Forcing function as specified by environment or user

#### forcing\_term

```python
def forcing_term()
```

Computes the user-specified forcing term of the vorticity equation

**Arguments**:

- `omega_hat` - Fourier transformed vorticity term
- `forcing` - Forcing function as specified by environment or user

