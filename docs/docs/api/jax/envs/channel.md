---
sidebar_label: channel
title: hydrogym.jax.envs.channel
---

## ChannelEnvParams Objects

```python
@struct.dataclass
class ChannelEnvParams(BaseEnvParams)
```

Extends base EnvParams with channel-specific settings.

#### nsteps

DNS substeps per RL step

## ChannelEnvState Objects

```python
@struct.dataclass
class ChannelEnvState(environment.EnvState)
```

Channel-flow environment state (physical-space velocity fields).

**Attributes**:

- `time` - Simulation time (in DNS steps) elapsed in the episode.
- `U` - Streamwise velocity field, shape (Nx, Ny, Nz).
- `V` - Wall-normal velocity field, shape (Nx, Ny, Nz).
- `W` - Spanwise velocity field, shape (Nx, Ny, Nz).
- `dt` - Current DNS timestep.
- `terminal` - Whether the episode has terminated.

## SpectralState Objects

```python
class SpectralState(NamedTuple)
```

Three-component spectral (FFT&#x27;d) velocity state.

**Attributes**:

- `u_hat` - x-component spectrum, shape (Nx, Ny, Nz), complex.
- `v_hat` - y-component spectrum.
- `w_hat` - z-component spectrum.

#### u\_hat

(Nx,Ny,Nz), complex

#### make\_obs\_grid\_indices

```python
def make_obs_grid_indices(Nx: int, Ny: int, n: int)
```

Build the (x, y) index grid used to subsample the observation plane.

**Arguments**:

- `Nx` - Number of grid points in x.
- `Ny` - Number of grid points in y.
- `n` - Number of subsample points along each axis.
  

**Returns**:

  Tuple ``(Xi, Yi)`` of index arrays of shape (n, n) spanning the full
  x/y ranges, suitable for advanced indexing of a slice of the state.

#### get\_obs\_spectral\_channel

```python
def get_obs_spectral_channel(state: ChannelEnvState, params: ChannelEnvParams,
                             Xi: jnp.ndarray, Yi: jnp.ndarray) -> chex.Array
```

Extract the channel-flow observation vector from the current state.

Samples the streamwise (U) and spanwise (W) velocity at the subsampled
grid points on the wall-parallel plane ``z = params.k_det``.

**Arguments**:

- `state` - Current channel environment state.
- `params` - Channel environment parameters.
- `Xi` - Observation x-index grid (from :func:`make_obs_grid_indices`).
- `Yi` - Observation y-index grid.
  

**Returns**:

  1D array of the sampled U and W values stacked and flattened.

#### wss\_compute

```python
def wss_compute(k, U, nu=1.9e-3, z=None)
```

Compute the domain-mean wall shear stress from a finite-difference gradient.

Approximates the wall-normal velocity gradient at the wall by a one-sided
difference between wall-parallel slices ``k`` and ``0``, using the physical
z-coordinates in ``z``.

**Arguments**:

- `k` - Index of the near-wall slice used for the finite difference.
- `U` - Streamwise velocity field, shape (Nx, Ny, Nz).
- `nu` - Kinematic viscosity used to convert the gradient to stress.
- `z` - Physical z-coordinates of the grid points (1D array).
  

**Returns**:

  Mean wall shear stress over the (Nx, Ny) plane.

## PseudoSpectralNavierStokes3D Objects

```python
class PseudoSpectralNavierStokes3D(SplitEquation)
```

Pseudo-spectral 3D Navier-Stokes equation for a channel flow.

Horizontal (x, y) directions are treated spectrally with FFTs and the
wall-normal (z) direction with Chebyshev collocation. The state is the
triple of spectral velocity components on the (Nx, Ny, Nz) grid. The
incompressibility constraint is enforced by a pressure Poisson solve whose
per-horizontal-wavenumber Chebyshev matrices are pre-inverted at
construction, with the no-penetration wall condition folded in.

The pressure solver imposes the mean-pressure gauge by pinning one grid
point (the zero-wavenumber row), and the wall-normal pressure-gradient
boundary conditions (from the wall values of w) are applied in
:meth:`project`.

#### \_\_init\_\_

```python
def __init__(Nx, Ny, Nz, Lx, Ly, Lz, nu, dtype=jnp.float32)
```

Precompute wavenumbers, derivative operators, dealiasing mask, and pressure solver.

**Arguments**:

- `Nx` - Number of grid points in x.
- `Ny` - Number of grid points in y.
- `Nz` - Number of Chebyshev collocation points in z.
- `Lx` - Domain length in x.
- `Ly` - Domain length in y.
- `Lz` - Domain height in z.
- `nu` - Kinematic viscosity.
- `dtype` - Floating dtype for the precomputed operators.

#### fft\_xy

```python
def fft_xy(f)
```

Forward FFT of a field over the horizontal (x, y) axes.

**Arguments**:

- `f` - Physical-space field of shape (Nx, Ny, Nz).
  

**Returns**:

  Spectral field of the same shape.

#### ifft\_xy

```python
def ifft_xy(f_hat)
```

Inverse FFT of a field over the horizontal (x, y) axes, taking the real part.

**Arguments**:

- `f_hat` - Spectral field of shape (Nx, Ny, Nz).
  

**Returns**:

  Physical-space (real) field of the same shape.

#### to\_spectral

```python
def to_spectral(state: "VelocityState") -> "VelocityState"
```

Transform a physical-space velocity state to spectral space.

**Arguments**:

- `state` - Physical-space :class:`VelocityState`.
  

**Returns**:

  Spectral-space :class:`VelocityState`.

#### to\_physical

```python
def to_physical(state_hat: "VelocityState") -> "VelocityState"
```

Transform a spectral velocity state to physical space.

**Arguments**:

- `state_hat` - Spectral-space :class:`VelocityState`.
  

**Returns**:

  Physical-space :class:`VelocityState`.

#### dx\_hat

```python
def dx_hat(f_hat)
```

x-derivative of a spectral field (multiplication by i*kx).

**Arguments**:

- `f_hat` - Spectral field.
  

**Returns**:

  Spectral x-derivative.

#### dy\_hat

```python
def dy_hat(f_hat)
```

y-derivative of a spectral field (multiplication by i*ky).

**Arguments**:

- `f_hat` - Spectral field.
  

**Returns**:

  Spectral y-derivative.

#### dz\_phys

```python
def dz_phys(f_phys)
```

First z-derivative of a physical field via the Chebyshev matrix.

**Arguments**:

- `f_phys` - Physical-space field of shape (Nx, Ny, Nz).
  

**Returns**:

  z-derivative of the same shape.

#### dzz\_phys

```python
def dzz_phys(f_phys)
```

Second z-derivative of a physical field via the Chebyshev matrix.

**Arguments**:

- `f_phys` - Physical-space field of shape (Nx, Ny, Nz).
  

**Returns**:

  Second z-derivative of the same shape.

#### apply\_jets\_v

```python
def apply_jets_v(v,
                 Vjets,
                 z0=1,
                 jet_thickness=5,
                 slit_length_x=3.0,
                 slit_width_y=2.0,
                 x_span_frac=2 / 3,
                 nx_jets=6,
                 ny_jets=4)
```

Imprint a grid of wall-normal jet velocity profiles onto the v-field.

A periodic array of ``nx_jets x ny_jets`` Gaussian jets centered on the
bottom wall builds a velocity mask from the per-jet amplitudes in
``Vjets``; the mask is mean-subtracted (zero net blowing) and written
into the wall-normal component ``v`` over ``jet_thickness`` layers
starting at wall layer ``z0``. The wall layers of ``v`` are first set
to zero.

**Arguments**:

- ``2 - Wall-normal velocity field, shape (Nx, Ny, Nz).
- ``3 - Jet amplitudes, shape (nx_jets, ny_jets).
- ``4 - First wall-normal layer the jets penetrate.
- ``5 - Number of layers the jets span.
- ``6 - Gaussian width of each jet in x.
- ``7 - Gaussian width of each jet in y.
- ``8 - Fraction of the x domain covered by the jet array.
- ``9 - Number of jets along x.
- ``0 - Number of jets along y.
  

**Returns**:

  The modified wall-normal velocity field.

#### enforce\_noslip

```python
def enforce_noslip(u, v, w, action=None, action_time=50.0, t=0.0)
```

Apply the wall boundary conditions to a physical-space velocity state.

u and v are forced to zero at both walls. For w, an actuation ramp is
applied: if an ``action`` is given, the wall-normal jets
(:meth:`apply_jets_v`) are driven with amplitude proportional to the
action, scaled by a gain that ramps linearly up over the first 10 time
units and down over the last 10 before ``action_time``; without an
action, w is simply set to zero at the walls.

**Arguments**:

- `u` - Streamwise velocity field, shape (Nx, Ny, Nz).
- `v` - Wall-normal velocity field.
- `w` - Spanwise velocity field.
- `action` - Jet control amplitudes, or None for the unactuated case.
- `action_time` - Total actuation window over which the gain ramps up and down.
- ``0 - Current time, used for the gain ramp.
  

**Returns**:

  Tuple ``(u, v, w)`` with the boundary conditions applied.

#### nonlinear\_terms

```python
def nonlinear_terms(state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0)
```

Evaluate the explicitly treated (nonlinear advection + forcing) terms.

The state is transformed to physical space, wall boundary conditions
(and jets, if actuated) are applied, and the advective terms
``u · ∇u`` are computed with spectral x/y derivatives and Chebyshev
z-derivatives. The result is returned in spectral space with the 2/3
dealiasing mask applied and the body forcing added.

**Arguments**:

- `state_hat` - Spectral velocity state.
- `action` - Jet control amplitudes, or None.
- `t` - Current time (used for the actuation gain ramp).
- `fx` - x-direction body forcing (scalar or physical-space field).
- `fy` - y-direction body forcing.
- `fz` - z-direction body forcing.
  

**Returns**:

  Spectral :class:`VelocityState` of the nonlinear terms.

#### linear\_terms

```python
def linear_terms(state_hat, action=None, t=0.0)
```

Evaluate the implicitly treated (viscous diffusion) term.

Computes ``nu * (∂²u/∂z² - k²u)`` for each component, combining the
Chebyshev second z-derivative in physical space with the horizontal
Laplacian applied spectrally.

**Arguments**:

- `state_hat` - Spectral velocity state.
- `action` - Unused; present for interface compatibility.
- `t` - Unused; present for interface compatibility.
  

**Returns**:

  Spectral :class:`VelocityState` of the linear terms.

#### rhs

```python
def rhs(state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0)
```

Evaluate the full right-hand side as nonlinear + linear terms.

**Arguments**:

- `state_hat` - Spectral velocity state.
- `action` - Jet control amplitudes, or None.
- `t` - Current time.
- `fx` - x-direction body forcing.
- `fy` - y-direction body forcing.
- `fz` - z-direction body forcing.
  

**Returns**:

  Spectral :class:`VelocityState` of the full right-hand side.

#### project

```python
def project(state_hat, dt, action=None, t=0.0)
```

Project the state onto the divergence-free (incompressibility) constraint.

Solves the pressure Poisson problem for the divergence error over one
timestep using the precomputed per-wavenumber inverse Chebyshev
matrices, with the wall-normal pressure-gradient conditions derived
from the wall values of w, and the mean-pressure gauge pinned at the
zero-wavenumber mode. The velocity is corrected by the pressure
gradient, the boundary conditions (and jets, if actuated) are
re-applied, and the result is returned in spectral space.

**Arguments**:

- `state_hat` - Spectral velocity state to project.
- `dt` - Timestep used in the pressure correction.
- `action` - Jet control amplitudes, or None.
- `t` - Current time (used for the actuation gain ramp).
  

**Returns**:

  Projected spectral :class:`VelocityState`.

#### run\_channel\_pseudospectral

```python
def run_channel_pseudospectral(U0,
                               V0,
                               W0,
                               action,
                               dt: float,
                               equation: PseudoSpectralNavierStokes3D,
                               integrator: RungeKutta4,
                               nsteps: int = 50,
                               return_trajectory: bool = False,
                               checkpoint_steps: bool = True)
```

Run a channel-flow DNS rollout of ``nsteps`` RK4 steps.

The initial physical fields are transformed to spectral space, advanced
with ``integrator.rk4_step`` for ``nsteps`` substeps (constant body forcing
``fx = 2.0``, constant-mass-flux correction toward a bulk velocity of 8.0),
and returned either as a trajectory of physical-space snapshots or as the
final state only.

**Arguments**:

- `U0` - Initial streamwise velocity field, shape (Nx, Ny, Nz).
- `V0` - Initial wall-normal velocity field.
- ``0 - Initial spanwise velocity field.
- ``1 - Jet control amplitudes applied at every substep, or None.
- ``2 - DNS timestep.
- ``3 - The pseudo-spectral equation (provides transforms).
- ``4 - The RK4 integrator to step with.
- ``5 - Number of DNS substeps to run.
- ``6 - If True, return the full trajectory of physical
  velocity fields instead of just the final state.
- ``7 - If True, wrap the step function in ``jax.checkpoint``
  to trade compute for reduced memory during the scan.
  

**Returns**:

  If ``return_trajectory``: arrays ``(U, V, W)`` of trajectories with a
  leading time axis of length ``nsteps``; otherwise the final state&#x27;s
  physical ``(u, v, w)`` fields.

## ChannelFlowSpectralEnv Objects

```python
class ChannelFlowSpectralEnv(JAXFlowEnvBase)
```

3D turbulent channel flow environment using a pseudo-spectral DNS solver.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Build the equation, integrator, and initial fields for the channel.

Physical/spectral grid parameters are taken from ``env_config`` (with
defaults matching the previously hardcoded values). Initial velocity
fields are loaded either from a local directory or from Hugging Face
(see the source comments for the accepted overrides).

**Arguments**:

- `env_config` - Configuration dictionary; recognized keys include
  ``Lx``/``Ly``/``Lz`` (domain extents), ``nu`` (viscosity),
  ``Nx``/``Ny``/``Nz`` (grid resolution, note that non-default
  values change the JIT-compiled operator shapes and require
  matching initial-field shapes), ``dtype`` (&quot;float32&quot; or
  &quot;float64&quot;), ``initial_field_dir`` (local directory containing
  U/V/W.npy), and the HFDataManager forwarding options
  ``hf_repo_id``/``cache_dir``/``use_clean_cache``/
  ``local_fallback_dir``/``hf_token``/``hf_revision``.

#### default\_params

```python
@property
def default_params() -> ChannelEnvParams
```

Default :class:`ChannelEnvParams` for this environment.

#### name

```python
@property
def name() -> str
```

Name of this environment.

#### reset\_env

```python
def reset_env(key: chex.PRNGKey,
              params: ChannelEnvParams) -> Tuple[chex.Array, ChannelEnvState]
```

Reset the channel to the stored initial fields.

**Arguments**:

- `key` - PRNG key (forwarded to the observation function; the reset
  itself is deterministic).
- `params` - Channel environment parameters.
  

**Returns**:

  Tuple ``(obs, ChannelEnvState)`` at time 0 with ``terminal=False``.

#### get\_obs

```python
def get_obs(state: ChannelEnvState,
            params: ChannelEnvParams,
            key=None) -> chex.Array
```

Sample the subsampled U/W observation plane at ``k_det``.

**Arguments**:

- `state` - Current channel environment state.
- `params` - Channel environment parameters.
- `key` - Unused; accepted for API compatibility.
  

**Returns**:

  1D observation array (see :func:`get_obs_spectral_channel`).

#### is\_terminal

```python
def is_terminal(state: ChannelEnvState,
                params: ChannelEnvParams) -> jnp.ndarray
```

Whether the episode has ended (``terminal`` flag or step limit reached).

**Arguments**:

- `state` - Current channel environment state.
- `params` - Channel environment parameters.
  

**Returns**:

  Boolean array indicating termination.

#### step\_env

```python
def step_env(
    key: chex.PRNGKey, state: ChannelEnvState, action: jnp.ndarray,
    params: ChannelEnvParams
) -> Tuple[chex.Array, ChannelEnvState, jnp.ndarray, jnp.ndarray, Dict]
```

Advance the channel flow by ``params.nsteps`` DNS substeps under the given jet action.

The action is clipped to the parameter bounds and the wall shear stress
after the rollout is differentiated with respect to the action (via
``jax.value_and_grad``) to obtain the reward ``-wss`` together with its
gradient.

**Arguments**:

- `key` - PRNG key (forwarded to the observation function; the dynamics
  are deterministic).
- `state` - Current channel environment state.
- `action` - Jet control input (clipped to ``[min_action, max_action]``).
- ``1 - Channel environment parameters.
  

**Returns**:

  Tuple ``(obs, next_state, reward, done, info)`` where ``info``
  contains ``&quot;discount&quot;``. (The reward gradient w.r.t. the action is
  computed as ``grad_wss`` but is not currently returned.)

#### action\_space

```python
def action_space(params: Optional[ChannelEnvParams] = None) -> spaces.Box
```

Return the jet-control action space.

**Arguments**:

- `params` - Channel environment parameters; defaults to
  ``self.default_params``.
  

**Returns**:

  Gymnax Box of shape ``(params.action_dim,)`` with bounds
  ``[params.min_action, params.max_action]``.

#### observation\_space

```python
def observation_space(params: Optional[ChannelEnvParams] = None) -> spaces.Box
```

Return the (unbounded) observation space.

**Arguments**:

- `params` - Channel environment parameters; defaults to
  ``self.default_params``.
  

**Returns**:

  Gymnax Box of shape
  ``(params.obs_subsample**2 * params.obs_include_components,)`` with
  infinite bounds.

