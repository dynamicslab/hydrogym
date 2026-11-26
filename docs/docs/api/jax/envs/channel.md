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
- `U` - Streamwise (x) velocity field, shape (Nx, Ny, Nz).
- `V` - Spanwise (y) velocity field, shape (Nx, Ny, Nz).
- `W` - Wall-normal (z) velocity field, shape (Nx, Ny, Nz).
- `dt` - Current DNS timestep.
- `terminal` - Whether the episode has terminated.

## SpectralState Objects

```python
class SpectralState(NamedTuple)
```

Three-component spectral (FFT'd) velocity state.

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

  Tuple `(Xi, Yi)` of index arrays of shape (n, n) spanning the full
  x/y ranges, suitable for advanced indexing of a slice of the state.

#### get\_obs\_spectral\_channel

```python
def get_obs_spectral_channel(state: ChannelEnvState, params: ChannelEnvParams,
                             Xi: jnp.ndarray, Yi: jnp.ndarray) -> chex.Array
```

Extract the channel-flow observation vector from the current state.

Samples the streamwise (U) and wall-normal (W) velocity at the subsampled
grid points on the wall-parallel plane `z = params.k_det`.

**Arguments**:

- `state` - Current channel environment state.
- `params` - Channel environment parameters.
- `Xi` - Observation x-index grid (from `make_obs_grid_indices`).
- `Yi` - Observation y-index grid.
  

**Returns**:

  1D array of the sampled U and W values stacked and flattened.

#### clenshaw\_curtis\_weights

```python
def clenshaw_curtis_weights(n_points: int, length: float) -> np.ndarray
```

Clenshaw-Curtis weights of the Chebyshev-Gauss-Lobatto points of `cheb_D_matrices` on `length`.

#### wss\_compute

```python
def wss_compute(k, U, nu=1.9e-3, z=None)
```

Compute the domain-mean wall shear stress from a finite-difference gradient.

Approximates the wall-normal velocity gradient at the wall by a one-sided
difference between wall-parallel slices `k` and `0`, using the physical
z-coordinates in `z`.

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
`project`.

Jets (`jet_model`):

- `"wall"` (default): the 24 jets are a wall-normal transpiration
velocity `w(x, y, z_wall) = w_jet(x, y)` on the bottom wall (zero net
mass flux), imposed consistently in the pressure projection. At zero
action `w_jet = 0` and the equation is the uncontrolled channel.
- `"legacy"`: reproduces the jets of HydroGym &lt;= 1.0, which overwrite
`w` in the grid layers `z[1:6]` next to the bottom wall after every
stage (also at zero action) with the gain of `jet_gain`.

#### JET\_GAIN

peak jet velocity per unit action

#### \_\_init\_\_

```python
def __init__(Nx,
             Ny,
             Nz,
             Lx,
             Ly,
             Lz,
             nu,
             dtype=jnp.float32,
             jet_model="wall",
             action_time=50.0,
             ramp_time=10.0)
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
- `jet_model` - `"wall"` (transpiration on the bottom wall, default)
  or `"legacy"` (overwrite of the layers `z[1:6]`, HydroGym
  &lt;= 1.0); see the class docstring.
- `action_time` - Length of one action window in the time unit of the
  `t` passed to the boundary conditions (`"wall"` model; the
  legacy gain always uses 50).
- `ramp_time` - Duration of the linear up/down ramp of the jet gain at
  the start/end of each action window (`"wall"` model).

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

- `state` - Physical-space `VelocityState`.
  

**Returns**:

  Spectral-space `VelocityState`.

#### to\_physical

```python
def to_physical(state_hat: "VelocityState") -> "VelocityState"
```

Transform a spectral velocity state to physical space.

**Arguments**:

- `state_hat` - Spectral-space `VelocityState`.
  

**Returns**:

  Physical-space `VelocityState`.

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

Imprint a grid of wall-normal jet velocity profiles onto the wall-normal field `w`.

A periodic array of `nx_jets x ny_jets` Gaussian jets centered on the
bottom wall builds a velocity mask from the per-jet amplitudes in
`Vjets`; the mask is mean-subtracted (zero net blowing) and written
into the wall-normal component over `jet_thickness` layers
starting at wall layer `z0`. The wall layers are first set
to zero.

**Arguments**:

- `v` - Wall-normal velocity field `w` (positional argument), shape (Nx, Ny, Nz).
- `Vjets` - Jet amplitudes, shape (nx_jets, ny_jets).
- `z0` - First wall-normal layer the jets penetrate.
- `jet_thickness` - Number of layers the jets span.
- `slit_length_x` - Gaussian width of each jet in x.
- `slit_width_y` - Gaussian width of each jet in y.
- `x_span_frac` - Fraction of the x domain covered by the jet array.
- `nx_jets` - Number of jets along x.
- `ny_jets` - Number of jets along y.
  

**Returns**:

  The modified wall-normal velocity field.

#### jet\_mask

```python
def jet_mask(Vjets,
             slit_length_x=3.0,
             slit_width_y=2.0,
             x_span_frac=2 / 3,
             nx_jets=6,
             ny_jets=4)
```

Zero-mean wall-normal velocity profile of the jet array, shape (Nx, Ny).

The same Gaussian jets as `apply_jets_v`: `nx_jets x ny_jets`
Gaussians with amplitudes `Vjets`, minus their plane mean (zero net
mass flux).

#### jet\_gain

```python
def jet_gain(t, action_time=50.0)
```

Jet gain (peak jet velocity per unit action) at time `t` of the action window.

`"wall"` model: `JET_GAIN * clip(min(t, T - t) / ramp_time, 0, 1)`
with `T = self.action_time`: a linear ramp up over the first
`ramp_time` of each action window, constant, and a linear ramp down
to 0 over the last `ramp_time`. `t` is the time since the start
of the current action.

`"legacy"` model (reproduces HydroGym &lt;= 1.0): the same formula
without the clip, with `T = action_time` (default 50) and
`ramp_time = 10` in units of DNS steps, evaluated at the episode DNS
step count `state.time + n`.

#### jet\_wall\_velocity

```python
def jet_wall_velocity(action, t)
```

Transpiration velocity `w_jet(x, y)` the jets impose on the bottom wall at time `t` (`"wall"` model).

Positive action = suction (`w < 0` at the bottom wall, as in the
legacy model). Returns zeros for `action=None`.

#### enforce\_noslip

```python
def enforce_noslip(u, v, w, action=None, action_time=50.0, t=0.0)
```

Apply the wall boundary conditions to a physical-space velocity state.

u and v are forced to zero at both walls, and w at the top wall. On
the bottom wall:

- `"wall"` model: `w = w_jet` (`jet_wall_velocity`, zero for
zero action or `action=None`); the interior is not touched.
- `"legacy"` model: with an `action`, the jets
(`apply_jets_v`) overwrite w in the layers `z[1:6]` (also at
zero action) with the legacy gain; without an action, w is set to
zero at the walls.

**Arguments**:

- `u` - Streamwise velocity field, shape (Nx, Ny, Nz).
- `v` - Spanwise velocity field.
- `w` - Wall-normal velocity field.
- `action` - Jet control amplitudes, or None for the unactuated case.
- `action_time` - Legacy model only: actuation window of the gain ramp.
- `t` - Current time, used for the gain ramp (see `jet_gain`).
  

**Returns**:

  Tuple `(u, v, w)` with the boundary conditions applied.

#### nonlinear\_terms

```python
def nonlinear_terms(state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0)
```

Evaluate the explicitly treated (nonlinear advection + forcing) terms.

The state is transformed to physical space, wall boundary conditions
(and jets, if actuated) are applied, and the advective terms
`u · ∇u` are computed with spectral x/y derivatives and Chebyshev
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

  Spectral `VelocityState` of the nonlinear terms.

#### linear\_terms

```python
def linear_terms(state_hat, action=None, t=0.0)
```

Evaluate the implicitly treated (viscous diffusion) term.

Computes `nu * (∂²u/∂z² - k²u)` for each component, combining the
Chebyshev second z-derivative in physical space with the horizontal
Laplacian applied spectrally.

**Arguments**:

- `state_hat` - Spectral velocity state.
- `action` - Unused; present for interface compatibility.
- `t` - Unused; present for interface compatibility.
  

**Returns**:

  Spectral `VelocityState` of the linear terms.

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

  Spectral `VelocityState` of the full right-hand side.

#### project

```python
def project(state_hat, dt, action=None, t=0.0, return_pre_bc=False)
```

Project the state onto the divergence-free (incompressibility) constraint.

Solves the pressure Poisson problem for the divergence error over one
timestep using the precomputed per-wavenumber inverse Chebyshev
matrices, with the wall-normal pressure-gradient conditions derived
from the wall values of w, and the mean-pressure gauge pinned at the
zero-wavenumber mode. The velocity is corrected by the pressure
gradient, the boundary conditions (and jets, if actuated) are
re-applied, and the result is returned in spectral space.

In the `"wall"` jet model the Neumann condition on the bottom wall is
`dp/dz = (w - w_jet) / dt`, so that the corrected velocity has the
jet transpiration velocity at the wall (and the mass-conserving
divergence-free projection includes it).

**Arguments**:

- `state_hat` - Spectral velocity state to project.
- `dt` - Timestep used in the pressure correction.
- `action` - Jet control amplitudes, or None.
- `t` - Current time (used for the actuation gain ramp).
- `return_pre_bc` - If True, also return the power diagnostic: for the
  `"legacy"` model the physical wall-normal velocity after the
  pressure correction and before the boundary/jet overwrite; for
  the `"wall"` model the physical pressure on the bottom wall,
  shape (Nx, Ny).
  

**Returns**:

  Projected spectral `VelocityState` (and, with
  `return_pre_bc`, the power diagnostic).

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

Run a channel-flow DNS rollout of `nsteps` RK4 steps.

The initial physical fields are transformed to spectral space, advanced
with `integrator.rk4_step` for `nsteps` substeps (constant body forcing
`fx = 2.0`, constant-mass-flux correction toward a bulk velocity of 8.0),
and returned either as a trajectory of physical-space snapshots or as the
final state only.

**Arguments**:

- `U0` - Initial streamwise velocity field, shape (Nx, Ny, Nz).
- `V0` - Initial spanwise velocity field.
- `W0` - Initial wall-normal velocity field.
- `action` - Jet control amplitudes applied at every substep, or None.
- `dt` - DNS timestep.
- `equation` - The pseudo-spectral equation (provides transforms).
- `integrator` - The RK4 integrator to step with.
- `nsteps` - Number of DNS substeps to run.
- `return_trajectory` - If True, return the full trajectory of physical
  velocity fields instead of just the final state.
- `checkpoint_steps` - If True, wrap the step function in `jax.checkpoint`
  to trade compute for reduced memory during the scan.
  

**Returns**:

  If `return_trajectory`: arrays `(U, V, W)` of trajectories with a
  leading time axis of length `nsteps`; otherwise the final state's
  physical `(u, v, w)` fields.

## ChannelFlowSpectralEnv Objects

```python
class ChannelFlowSpectralEnv(JAXFlowEnvBase)
```

3D turbulent channel flow environment using a pseudo-spectral DNS solver.

x streamwise, y spanwise (periodic), z wall-normal (Chebyshev, walls at
`z = -Lz/2, Lz/2`); constant mass flux at bulk velocity `U_b = 8`.
The reward is `-tau_w`, the mean shear stress on the bottom wall.

Jets. The 24 jets (6 x 4 Gaussian patches, zero net mass flux) impose a
wall-normal transpiration velocity `w_jet(x, y)` on the bottom wall
(positive action = suction). Each action window of `nsteps` DNS steps
starts and ends with a linear ramp of the jet gain over 10 DNS steps
(peak gain 0.3 per unit action), the same for every action. At zero
action the environment is the uncontrolled channel.

Actuation power. The power the jets put into the fluid is the energy flux
through the bottom wall, `P = sum w_jet (p - p_ref + w_jet^2 / 2) dA`
(`p` = wall pressure of the projection step, `p_ref` = its plane mean,
`dA` = cell area; the viscous normal stress vanishes at a no-slip wall),
the transpiration form also used for the Firedrake jets. It is formed
point-wise (`positive`/`abs` forms) and attributed per jet to the
nearest jet centre.

Legacy (`env_config["legacy_jets"] = True`) reproduces the jets of
HydroGym &lt;= 1.0: the jet profile overwrites `w` in the layers `z[1:6]`
next to the bottom wall after every stage, also at zero action, with the
gain of `PseudoSpectralNavierStokes3D.jet_gain`. The power is then
the rate of the kinetic-energy change of that impulsive constraint,
`P = sum (w_jet^2 - w_free^2) / 2 dV / dt` (`w_free` = velocity after
the pressure correction, before the overwrite; `dV` = cell area x
Clenshaw-Curtis weight).

Normalisation: `C_P = P / (rho U_b^3 Lx Ly / 2)` and
`drag_coefficient = C_f = tau_w / (rho U_b^2 / 2)` (density 1), so that
`C_f` is the drag power per bottom-wall area in the same units. Sampling:
`"mean"` (default, all DNS substeps of the action) or `"end"`.
Power options in `env_config`: `power_weight` (default 0: reward
unchanged), `power_cost`, `power_sampling`.

#### JET\_Z0

legacy jets: first layer overwritten (apply_jets_v default)

#### JET\_THICKNESS

legacy jets: number of overwritten layers (apply_jets_v default)

#### JET\_RAMP\_STEPS

DNS steps of the gain ramp at the start and end of each action

#### U\_BULK

target bulk velocity of the constant-mass-flux correction

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Build the equation, integrator, and initial fields for the channel.

Physical/spectral grid parameters are taken from `env_config` (with
the defaults of the Re_tau 180 channel). Initial velocity
fields are loaded either from a local directory or from Hugging Face
(see the source comments for the accepted overrides).

**Arguments**:

- `env_config` - Configuration dictionary; recognized keys include
  `Lx`/`Ly`/`Lz` (domain extents), `nu` (viscosity),
  `Nx`/`Ny`/`Nz` (grid resolution, note that non-default
  values change the JIT-compiled operator shapes and require
  matching initial-field shapes), `dtype` ("float32" or
  "float64"), `initial_field_dir` (local directory containing
  U/V/W.npy), and the HFDataManager forwarding options
  `hf_repo_id`/`cache_dir`/`use_clean_cache`/
  `local_fallback_dir`/`hf_token`/`hf_revision`,
  `legacy_jets` (default False: wall transpiration jets with a
  per-action gain ramp; True: the HydroGym &lt;= 1.0 layer overwrite
  and gain, see the class docstring).

#### default\_params

```python
@property
def default_params() -> ChannelEnvParams
```

Default `ChannelEnvParams` for this environment.

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

  Tuple `(obs, ChannelEnvState)` at time 0 with `terminal=False`.

#### get\_obs

```python
def get_obs(state: ChannelEnvState,
            params: ChannelEnvParams,
            key=None) -> chex.Array
```

Sample the subsampled U/W observation plane at `k_det`.

**Arguments**:

- `state` - Current channel environment state.
- `params` - Channel environment parameters.
- `key` - Unused; accepted for API compatibility.
  

**Returns**:

  1D observation array (see `get_obs_spectral_channel`).

#### is\_terminal

```python
def is_terminal(state: ChannelEnvState,
                params: ChannelEnvParams) -> jnp.ndarray
```

Whether the episode has ended (`terminal` flag or step limit reached).

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

Advance the channel flow by `params.nsteps` DNS substeps under the given jet action.

The action is clipped to the parameter bounds; the reward is `-wss`,
the mean wall shear stress of the rollout (with the power term of
`power_config`). The step is differentiable with respect to the
action, e.g. `jax.grad` of the reward through `step_env`.

**Arguments**:

- `key` - PRNG key (forwarded to the observation function; the dynamics
  are deterministic).
- `state` - Current channel environment state.
- `action` - Jet control input (clipped to `[min_action, max_action]`).
- `params` - Channel environment parameters.
  

**Returns**:

  Tuple `(obs, next_state, reward, done, info)` where `info`
  contains `"discount"`.

#### action\_space

```python
def action_space(params: Optional[ChannelEnvParams] = None) -> spaces.Box
```

Return the jet-control action space.

**Arguments**:

- `params` - Channel environment parameters; defaults to
  `self.default_params`.
  

**Returns**:

  Gymnax Box of shape `(params.action_dim,)` with bounds
  `[params.min_action, params.max_action]`.

#### observation\_space

```python
def observation_space(params: Optional[ChannelEnvParams] = None) -> spaces.Box
```

Return the (unbounded) observation space.

**Arguments**:

- `params` - Channel environment parameters; defaults to
  `self.default_params`.
  

**Returns**:

  Gymnax Box of shape
  `(params.obs_subsample**2 * params.obs_include_components,)` with
  infinite bounds.

