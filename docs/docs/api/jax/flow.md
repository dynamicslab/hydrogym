---
sidebar_label: flow
title: hydrogym.jax.flow
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

#### evaluate\_objective

```python
def evaluate_objective()
```

Return a copy of the flow state

#### render

```python
def render(**kwargs)
```

Plot the current PDE state (called by `gym.Env`)

