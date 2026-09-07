---
sidebar_label: kolmogorov
title: hydrogym.jax.envs.kolmogorov
---

## FlowConfig Objects

```python
class FlowConfig(PDEBase)
```

Flow configuration for the 2D Kolmogorov flow on a periodic square domain.

Holds the forcing wavenumber ``k``, Reynolds number ``Re``, grid size,
domain extents, and observation size, and provides the real/Fourier meshes,
the divergence-free initial condition, the sinusoidal forcing, and the
observation extraction used by the environment. The state stored in
``self.vorticity`` (and ``self.state``) is the FFT of the vorticity field.

#### DEFAULT\_OBS\_SIZE

This correlates to a total observation size of 8x8 = 64.

#### \_\_init\_\_

```python
def __init__(**config)
```

Read the flow-configuration options and set up the control field.

**Arguments**:

- `**config` - Optional overrides: ``k`` (forcing wavenumber, default 4),
  ``Re`` (Reynolds number, default 200), ``grid_size`` ((ny, nx)
  tuple, default (64, 64)), ``domain_x``/``domain_y`` (extent
  tuples, default (0, 2π)), ``obs_size`` (observation grid side,
  default 8). Remaining keys are forwarded to
  :class:``3.

#### load\_mesh

```python
def load_mesh(name)
```

Create jax grid given the desired dimensions and spacing in real space

**Returns**:

  jax meshgrid

#### state

```python
def state() -> jnp.array
```

Return the current flow state (the FFT&#x27;d vorticity field).

#### get\_observations

```python
def get_observations() -> jnp.array
```

Compute velocity-magnitude observations over the whole stored trajectory.

For each saved vorticity snapshot in ``self.vorticity``, the velocity is
reconstructed in Fourier space and evaluated on a regularly subsampled
grid of ``obs_size x obs_size`` points.

**Returns**:

  Array of observations with a leading axis over the trajectory
  (shape (n_saved, obs_size * obs_size)).

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

#### set\_BCs

```python
def set_BCs()
```

Apply boundary conditions (no-op; the domain is fully periodic).

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

Evaluate the control objective (not implemented; returns None).

#### nu

```python
@property
def nu()
```

Kinematic viscosity ``1 / Re``.

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

#### load\_checkpoint

```python
def load_checkpoint(filename: str)
```

Load a saved flow state from disk (not implemented).

## PseudoSpectralNavierStokes2D Objects

```python
class PseudoSpectralNavierStokes2D(IMEXEquation)
```

Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver.
We transform the 2D Navier-Stokes equation to a vorticity equation:
∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
ω = - ∇²φ ;
and solve in Fourier space

#### \_\_init\_\_

```python
def __init__(flow: FlowConfig)
```

Store the flow configuration and cache its Fourier and real-space meshes.

**Arguments**:

- `flow` - The :class:`FlowConfig` providing the grid, viscosity, and
  forcing for the equation.

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
def nonlinear_terms(omega_hat, control_field=None)
```

Computes the explicit (nonlinear) terms in the vorticity equation.
Uses the stream function to compute velocity components in Fourier space.

**Arguments**:

- `omega_hat` - fft of vorticity
- `control_field` - tuple (cfx, cfy) of physical-space forcing arrays, or None
  

**Returns**:

- `terms` - Nonlinear terms of the equation.

#### control\_term

```python
def control_term(omega_hat, control_field=None)
```

Computes the user-specified forcing term of the vorticity equation

**Arguments**:

- `omega_hat` - Fourier transformed vorticity term
- `control_field` - tuple (cfx, cfy) of physical-space forcing arrays, or None

#### forcing\_term

```python
def forcing_term()
```

Compute the environmental forcing term of the vorticity equation.

Evaluates the flow&#x27;s ``forcing_function(k, x, y)`` in physical space,
transforms the (fx, fy) velocity forcing to Fourier space, and takes
its curl ``2i*pi * (fy_hat * kx - fx_hat * ky)`` to obtain the
vorticity-space forcing.

**Returns**:

  The spectral forcing term of the same shape as the state, or
  ``None`` if the flow defines no forcing function.

## KolmogorovFlowState Objects

```python
@struct.dataclass
class KolmogorovFlowState(environment.EnvState)
```

Kolmogorov-flow environment state.

**Attributes**:

- `trajectory` - Saved spectral vorticity snapshots of the last rollout
  (leading axis over time).
- `omega_hat` - Final spectral vorticity of the last rollout.
- `time` - Number of RL steps taken in the current episode.
- `terminal` - Whether the episode has terminated.

## KolmogorovFlowParams Objects

```python
@struct.dataclass
class KolmogorovFlowParams(EnvParams)
```

Gymnax parameters for the Kolmogorov-flow environment.

**Attributes**:

- `min_action` - Lower bound of each control amplitude.
- `max_action` - Upper bound of each control amplitude.
- `min_obs` - Lower bound of the observation space (unbounded).
- `max_obs` - Upper bound of the observation space (unbounded).
- `dt` - DNS timestep of the integrator.
- `action_time` - Physical time simulated per RL step.
- `save_time` - Time interval between saved trajectory states.
  k1, k2, k3, k4: Wavenumbers of the four sinusoidal control modes.
- `action_dim` - Number of control amplitudes.
- `obs_dim` - Flattened observation size.
- `max_episode_steps` - Episode length in RL steps.
- `max_action`0 - Weight of the TKE term in the reward (the action
  penalty is always weighted at 1).
- `max_action`1 - Whether gradient information is included (kept for
  interface compatibility).

## KolmogorovFlow Objects

```python
class KolmogorovFlow(JAXFlowEnvBase)
```

2D Kolmogorov-flow Gymnax environment.

Each RL step rolls the pseudo-spectral Navier-Stokes solver forward for
``action_time`` of physical time with the four sinusoidal control modes
derived from the action vector. The observation is the time-mean of the
velocity magnitude over the rollout, sampled on an ``obs_size x obs_size``
grid; the reward combines the mean turbulent kinetic energy with an
L1 penalty on the action.

#### \_\_init\_\_

```python
def __init__(env_config: Optional[Dict] = None,
             flow_config: Optional[Dict] = None)
```

Create the flow configuration, equation, and integrator.

**Arguments**:

- `env_config` - Environment options; supports ``dt`` to override the
  integrator timestep (default: the value in
  :class:`KolmogorovFlowParams`).
- `flow_config` - Options forwarded to :class:`FlowConfig` (``k``,
  ``Re``, ``grid_size``, ``domain_x``, ``domain_y``, ``obs_size``).

#### name

```python
@property
def name() -> str
```

Name of this environment.

#### default\_params

```python
@property
def default_params() -> KolmogorovFlowParams
```

Default parameters, with ``obs_dim`` from the flow&#x27;s ``obs_size`` and the optional ``dt`` override applied.

**Returns**:

  A :class:`KolmogorovFlowParams` instance.

#### action\_space

```python
def action_space(params: Optional[KolmogorovFlowParams] = None)
```

Return the (Box) action space bounded by the parameters&#x27; action limits.

**Arguments**:

- `params` - Environment parameters; defaults to ``self.default_params``.
  

**Returns**:

  Gymnax Box space of shape ``(params.action_dim,)`` with bounds
  ``[params.min_action, params.max_action]``.

#### observation\_space

```python
def observation_space(params: KolmogorovFlowParams)
```

Return the (Box) observation space bounded by the parameters&#x27; observation limits.

**Arguments**:

- `params` - Environment parameters.
  

**Returns**:

  Gymnax Box space of shape ``(params.obs_dim,)`` with bounds
  ``[params.min_obs, params.max_obs]``.

#### get\_obs

```python
def get_obs(state: KolmogorovFlowState,
            params: KolmogorovFlowParams,
            key: Optional[chex.PRNGKey] = None) -> chex.Array
```

Compute the observation as the time-mean velocity magnitude over the rollout.

**Arguments**:

- `state` - Current environment state.
- `params` - Environment parameters (unused).
- `key` - Unused; accepted for API compatibility.
  

**Returns**:

  Flattened array of shape ``(params.obs_dim,)`` (see
  :meth:`_trajectory_mean_obs`).

#### reset\_env

```python
def reset_env(key: chex.PRNGKey, params: KolmogorovFlowParams)
```

Reset the environment: initialize the flow and spin it up unactuated.

The initial divergence-free vorticity field is generated and rolled out
for one ``action_time`` window with no control; the resulting state
carries both the final spectral vorticity and the saved trajectory.

**Arguments**:

- `key` - PRNG key (unused; the reset is deterministic).
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, KolmogorovFlowState)`` at RL time 0.

#### step\_env

```python
def step_env(key: chex.PRNGKey, state: KolmogorovFlowState, action: chex.Array,
             params: KolmogorovFlowParams)
```

Advance the environment by one RL step under the given action.

The action is clipped to the parameter bounds, converted into a
physical-space control field of four sinusoidal modes, and the flow is
rolled out for one ``action_time`` window. The observation is the
time-mean velocity magnitude over the rollout and the reward is the
weighted mean TKE plus the L1 action penalty (negated).

**Arguments**:

- `key` - PRNG key (unused; the dynamics are deterministic).
- `state` - Current environment state.
- `action` - Control amplitudes for the four sinusoidal modes.
- `params` - Environment parameters.
  

**Returns**:

  Tuple ``(obs, next_state, reward, done, info)`` where ``info``
  contains ``&quot;discount&quot;`` and ``&quot;mean_tke&quot;``.

