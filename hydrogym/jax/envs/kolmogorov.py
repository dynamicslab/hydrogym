from typing import Callable, Dict, Iterable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import tree_math
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax

from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.env_core import EnvParams, JAXFlowEnvBase
from hydrogym.jax.equation import IMEXEquation
from hydrogym.jax.solvers.base import RungeKuttaCrankNicolson
from hydrogym.jax.utils.utils import compute_real_velocity_point, compute_tke, compute_velocity_fft, dealiasing

#######################################################################################
#                                                                                     #
#                             FLOW CONFIGURATION                                      #
#                                                                                     #
#######################################################################################


class FlowConfig(PDEBase):
    """Flow configuration for the 2D Kolmogorov flow on a periodic square domain.

    Holds the forcing wavenumber ``k``, Reynolds number ``Re``, grid size,
    domain extents, and observation size, and provides the real/Fourier meshes,
    the divergence-free initial condition, the sinusoidal forcing, and the
    observation extraction used by the environment. The state stored in
    ``self.vorticity`` (and ``self.state``) is the FFT of the vorticity field.
    """

    DEFAULT_REYNOLDS = 200
    DEFAULT_WAVENUMBER = 4
    DEFAULT_GRID_SIZE = (64, 64)
    DEFAULT_DOMAIN_X = (0, 2 * jnp.pi)
    DEFAULT_DOMAIN_Y = (0, 2 * jnp.pi)
    DEFAULT_OBS_SIZE = 8  # This correlates to a total observation size of 8x8 = 64.

    def __init__(self, **config):
        """Read the flow-configuration options and set up the control field.

        Args:
            **config: Optional overrides: ``k`` (forcing wavenumber, default 4),
                ``Re`` (Reynolds number, default 200), ``grid_size`` ((ny, nx)
                tuple, default (64, 64)), ``domain_x``/``domain_y`` (extent
                tuples, default (0, 2π)), ``obs_size`` (observation grid side,
                default 8). Remaining keys are forwarded to
                :class:`~hydrogym.core.PDEBase`.
        """
        # Keys are popped (not just read) so PDEBase's unknown-key warning
        # (audit Task 2.1) only fires on genuinely unknown options.
        self.k = config.pop("k", self.DEFAULT_WAVENUMBER)
        self.Re = config.pop("Re", self.DEFAULT_REYNOLDS)
        self.grid_size = config.pop("grid_size", self.DEFAULT_GRID_SIZE)
        self.domain_x = config.pop("domain_x", self.DEFAULT_DOMAIN_X)
        self.domain_y = config.pop("domain_y", self.DEFAULT_DOMAIN_Y)
        self.obs_size = config.pop("obs_size", self.DEFAULT_OBS_SIZE)
        self.control_function = (
            jnp.zeros_like(self.load_mesh("default")[0]),
            jnp.zeros_like(self.load_mesh("default")[1]),
        )

        super().__init__(**config)

    def load_mesh(self, name):
        """
        Create jax grid given the desired dimensions and spacing in real space

        Returns:
            jax meshgrid
        """
        x0, xn, nx = self.domain_x[0], self.domain_x[1], self.grid_size[1]
        y0, yn, ny = self.domain_y[0], self.domain_y[1], self.grid_size[0]
        x = jnp.linspace(x0, xn, nx)
        y = jnp.linspace(y0, yn, ny)
        return jnp.meshgrid(x, y, indexing="ij")

    def _calculate_velocity_point(self, state, k1, k2):
        """Evaluate the velocity at a single observation point from a spectral vorticity state.

        Reconstructs the velocity in Fourier space (via
        :func:`~hydrogym.jax.utils.utils.compute_velocity_fft`) and evaluates
        the real-space velocity at grid point ``(k1, k2)``.

        Args:
            state: Spectral vorticity field.
            k1: x-index of the evaluation point.
            k2: y-index of the evaluation point.

        Returns:
            Real-space velocity at that point (2-vector).
        """
        # Calculate velocity point
        kx, ky = self.load_fft_mesh()
        uhat, vhat = compute_velocity_fft(state, kx, ky)
        point_velocity = compute_real_velocity_point(uhat, vhat, k1, k2)

        return point_velocity

    def state(self) -> jnp.array:
        """Return the current flow state (the FFT'd vorticity field)."""
        return self.state

    def get_observations(self) -> jnp.array:
        """Compute velocity-magnitude observations over the whole stored trajectory.

        For each saved vorticity snapshot in ``self.vorticity``, the velocity is
        reconstructed in Fourier space and evaluated on a regularly subsampled
        grid of ``obs_size x obs_size`` points.

        Returns:
            Array of observations with a leading axis over the trajectory
            (shape (n_saved, obs_size * obs_size)).
        """
        n, m = self.grid_size
        divisor = n // self.obs_size

        def calculate_velocity(trajectory):
            """Evaluate the velocity at every subsampled observation point of one snapshot."""
            points = [
                self._calculate_velocity_point(trajectory, x, y)
                for x in range(0, n, int(n / divisor))
                for y in range(0, m, int(m / divisor))
            ]
            return jnp.array(points)

        def scan_fn(carry, state):
            """Observation for one trajectory snapshot (the carry is unused)."""
            obs_val = calculate_velocity(state)  # To use energy observation, swap this with calculate_energy
            return carry, obs_val

        _, all_obs = lax.scan(scan_fn, None, self.vorticity)
        return all_obs

    def load_fft_mesh(self):
        """Create jax grid given desired dimensions and spacing in real Fourier space

        Returns:
            jax meshgrid
        """
        N = self.grid_size[0]
        M = self.grid_size[1]
        dx = self.domain_x[1] / N
        dy = self.domain_y[1] / M
        kx = jnp.fft.fftfreq(N, dx)
        ky = jnp.fft.rfftfreq(M, dy)
        return jnp.meshgrid(kx, ky, indexing="ij")

    def initialize_state(self):
        """Generate a divergence free velocity field to initialize the state
        Initializing with divergence free field specified with the following stream function:

        φ(x,y) = sin(x)cos(y)

        Returns:
            fft vorticity field
        """
        X, Y = self.load_mesh("default")

        # Gradients of φ(x,y) #
        def dstream_func_dx(x, y):
            """x-derivative of the stream function, ``d(phi)/dx = cos(x)``."""
            return jnp.cos(x)

        def dstream_func_dy(x, y):
            """y-derivative of the stream function, ``d(phi)/dy = -sin(y)``."""
            return -jnp.sin(y)

        dudy = jax.grad(dstream_func_dy, argnums=1)
        dvdx = jax.grad(dstream_func_dx, argnums=0)
        du_dy = jnp.vectorize(dudy)(X, Y)
        dv_dx = jnp.vectorize(dvdx)(X, Y)
        vorticity = dv_dx - du_dy
        vorticity_0 = jnp.fft.rfftn(vorticity)

        self.vorticity = vorticity_0
        return self.vorticity

    def set_BCs(self):
        """Apply boundary conditions (no-op; the domain is fully periodic)."""
        # Set the boundary conditions
        pass

    def forcing_function(self, k, x, y):
        """Sinusoidal forcing function that drives the Kolmogorov flow.

        Args:
            k (int): forcing wavenumber
            x (jnp.array): spatial coordinates in x
            y (jnp.array): spatial coordinates in y

        Returns:
            tuple: forcing function in (x,y)
        """
        return (jnp.sin(k * y), jnp.zeros_like(y))

    def evaluate_objective(self):
        """Evaluate the control objective (not implemented; returns None)."""
        pass

    @property
    def nu(self):
        """Kinematic viscosity ``1 / Re``."""
        return 1 / self.Re

    @property
    def num_inputs(self) -> int:
        """Length of the control vector (number of actuators)"""
        return 2

    @property
    def num_outputs(self) -> int:
        """Number of scalar observed variables"""
        pass

    def save_checkpoint(self):
        """Set up mesh, function spaces, state vector, etc"""
        pass

    def init_bcs(self):
        """Initialize any boundary conditions for the PDE."""
        pass

    def copy_state(self, deepcopy=True):
        """Return a copy of the flow state"""
        pass

    def render(self, **kwargs):
        """Plot the current PDE state (called by `gym.Env`)"""
        pass

    def load_checkpoint(self, filename: str):
        """Load a saved flow state from disk (not implemented)."""
        pass


#######################################################################################
#                                                                                     #
#                             PSEUDOSPECTRAL EQUATION                                 #
#                                                                                     #
#######################################################################################


class PseudoSpectralNavierStokes2D(IMEXEquation):
    """
    Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver.
    We transform the 2D Navier-Stokes equation to a vorticity equation:
        ∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
        ω = - ∇²φ ;
    and solve in Fourier space
    """

    def __init__(self, flow: FlowConfig):
        """Store the flow configuration and cache its Fourier and real-space meshes.

        Args:
            flow: The :class:`FlowConfig` providing the grid, viscosity, and
                forcing for the equation.
        """
        self.flow = flow
        self.grid = flow.load_fft_mesh()
        self.real_grid = flow.load_mesh("name")
        self.kx, self.ky = self.grid
        self.x, self.y = self.real_grid

    def linear_terms(self, omega_hat):
        """Computes the linear (viscous) term of the vorticity equation"""
        return self.flow.nu * (2j * jnp.pi) ** 2 * (self.kx**2 + self.ky**2) * omega_hat

    def implicit_timestep(self, omega_hat, time_step):
        """
        Function that computes an implicit euler timestep,
          y_n+1 = y_n / (1-∇tλ).

        """
        double_derivative = (2j * jnp.pi) ** 2 * (self.kx**2 + self.ky**2)
        return 1 / (1 - time_step * self.flow.nu * double_derivative) * omega_hat

    def nonlinear_terms(self, omega_hat, control_field=None):
        """Computes the explicit (nonlinear) terms in the vorticity equation.
        Uses the stream function to compute velocity components in Fourier space.

        Args:
            omega_hat: fft of vorticity
            control_field: tuple (cfx, cfy) of physical-space forcing arrays, or None

        Returns:
            terms: Nonlinear terms of the equation.
        """

        kx, ky = self.kx, self.ky

        double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(self.kx) ** 2 + abs(ky) ** 2)
        double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

        psi_hat = -1 * omega_hat / double_derivative
        uhat = (2 * jnp.pi * 1j) * ky * psi_hat  # Get u,v from phi
        vhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat

        u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)

        grad_x_hat = 2j * jnp.pi * self.kx * omega_hat
        grad_y_hat = 2j * jnp.pi * self.ky * omega_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * u + grad_y * v)
        advection_hat = jnp.fft.rfftn(advection)

        forcing_hat = self.forcing_term()
        control_hat = self.control_term(omega_hat, control_field=control_field)
        advection_hat = dealiasing(advection_hat)  # 2/3 dealiasing rule

        terms = advection_hat + forcing_hat + control_hat
        return terms

    def control_term(self, omega_hat, control_field=None):
        """Computes the user-specified forcing term of the vorticity equation
        Args:
          omega_hat: Fourier transformed vorticity term
          control_field: tuple (cfx, cfy) of physical-space forcing arrays, or None
        """
        if control_field is None:
            return jnp.zeros_like(omega_hat)

        cfx, cfy = control_field
        cfx_hat = jnp.fft.rfftn(cfx)
        cfy_hat = jnp.fft.rfftn(cfy)

        return self.kx * cfy_hat - self.ky * cfx_hat

    def forcing_term(self):
        """Compute the environmental forcing term of the vorticity equation.

        Evaluates the flow's ``forcing_function(k, x, y)`` in physical space,
        transforms the (fx, fy) velocity forcing to Fourier space, and takes
        its curl ``2i*pi * (fy_hat * kx - fx_hat * ky)`` to obtain the
        vorticity-space forcing.

        Returns:
            The spectral forcing term of the same shape as the state, or
            ``None`` if the flow defines no forcing function.
        """
        forcing_func = self.flow.forcing_function
        if forcing_func is not None:
            kx, ky = self.grid
            x, y = self.real_grid
            fx, fy = forcing_func(k=self.flow.k, x=x, y=y)
            fx_hat, fy_hat = jnp.fft.rfft2(fx), jnp.fft.rfft2(fy)

            # Transform the velocity forcing into vorticity
            derivative_term = 2j * jnp.pi
            f_vorticity = derivative_term * (fy_hat * kx - fx_hat * ky)
            return f_vorticity
        else:
            return None


#######################################################################################
#                                                                                     #
#                             GYMNAX ENVIRONMENT                                      #
#                                                                                     #
#######################################################################################


@struct.dataclass
class KolmogorovFlowState(environment.EnvState):
    """Kolmogorov-flow environment state.

    Attributes:
        trajectory: Saved spectral vorticity snapshots of the last rollout
            (leading axis over time).
        omega_hat: Final spectral vorticity of the last rollout.
        time: Number of RL steps taken in the current episode.
        terminal: Whether the episode has terminated.
    """

    trajectory: jnp.ndarray
    omega_hat: jnp.ndarray
    time: jnp.ndarray
    terminal: jnp.ndarray


@struct.dataclass
class KolmogorovFlowParams(EnvParams):
    """Gymnax parameters for the Kolmogorov-flow environment.

    Attributes:
        min_action: Lower bound of each control amplitude.
        max_action: Upper bound of each control amplitude.
        min_obs: Lower bound of the observation space (unbounded).
        max_obs: Upper bound of the observation space (unbounded).
        dt: DNS timestep of the integrator.
        action_time: Physical time simulated per RL step.
        save_time: Time interval between saved trajectory states.
        k1, k2, k3, k4: Wavenumbers of the four sinusoidal control modes.
        action_dim: Number of control amplitudes.
        obs_dim: Flattened observation size.
        max_episode_steps: Episode length in RL steps.
        reward_alpha: Weight of the TKE term in the reward (the action
            penalty is always weighted at 1).
        include_grad: Whether gradient information is included (kept for
            interface compatibility).
    """

    min_action: float = -0.5
    max_action: float = 0.5
    min_obs: float = -jnp.inf
    max_obs: float = jnp.inf

    dt: float = 1e-3
    action_time: float = 10.0
    save_time: float = 1

    k1: int = 4
    k2: int = 5
    k3: int = 6
    k4: int = 7

    action_dim: int = 4
    obs_dim: int = 64
    max_episode_steps: int = 1000
    reward_alpha: float = 0.0

    include_grad: bool = True


class KolmogorovFlow(JAXFlowEnvBase):
    """2D Kolmogorov-flow Gymnax environment.

    Each RL step rolls the pseudo-spectral Navier-Stokes solver forward for
    ``action_time`` of physical time with the four sinusoidal control modes
    derived from the action vector. The observation is the time-mean of the
    velocity magnitude over the rollout, sampled on an ``obs_size x obs_size``
    grid; the reward combines the mean turbulent kinetic energy with an
    L1 penalty on the action.
    """

    def __init__(
        self,
        env_config: Optional[Dict] = None,
        flow_config: Optional[Dict] = None,
    ):
        """Create the flow configuration, equation, and integrator.

        Args:
            env_config: Environment options; supports ``dt`` to override the
                integrator timestep (default: the value in
                :class:`KolmogorovFlowParams`).
            flow_config: Options forwarded to :class:`FlowConfig` (``k``,
                ``Re``, ``grid_size``, ``domain_x``, ``domain_y``, ``obs_size``).
        """
        super().__init__(env_config)

        self.flow = FlowConfig(**(flow_config or {}))

        self.n, self.m = self.flow.grid_size
        self.x, self.y = self.flow.load_mesh("")
        self.kx, self.ky = self.flow.load_fft_mesh()

        self._dt_override = (env_config or {}).get("dt", None)

        default = self.default_params
        self.equation = PseudoSpectralNavierStokes2D(self.flow)
        self.integrator = RungeKuttaCrankNicolson(
            flow=self.flow,
            dt=float(default.dt),
            save_n=int(default.save_time),
            equation=self.equation,
        )

    @property
    def name(self) -> str:
        """Name of this environment."""
        return "KolmogorovFlow"

    @property
    def default_params(self) -> KolmogorovFlowParams:
        """Default parameters, with ``obs_dim`` from the flow's ``obs_size`` and the optional ``dt`` override applied.

        Returns:
            A :class:`KolmogorovFlowParams` instance.
        """
        dt_override = getattr(self, "_dt_override", None)
        kwargs = dict(action_dim=4, obs_dim=self.flow.obs_size**2)
        if dt_override is not None:
            kwargs["dt"] = float(dt_override)
        return KolmogorovFlowParams(**kwargs)

    def action_space(self, params: Optional[KolmogorovFlowParams] = None):
        """Return the (Box) action space bounded by the parameters' action limits.

        Args:
            params: Environment parameters; defaults to ``self.default_params``.

        Returns:
            Gymnax Box space of shape ``(params.action_dim,)`` with bounds
            ``[params.min_action, params.max_action]``.
        """
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: KolmogorovFlowParams):
        """Return the (Box) observation space bounded by the parameters' observation limits.

        Args:
            params: Environment parameters.

        Returns:
            Gymnax Box space of shape ``(params.obs_dim,)`` with bounds
            ``[params.min_obs, params.max_obs]``.
        """
        return spaces.Box(
            low=params.min_obs,
            high=params.max_obs,
            shape=(params.obs_dim,),
        )

    def _control_field(self, action: jnp.ndarray, params: KolmogorovFlowParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert the action vector into a physical-space forcing field.

        Builds the x-forcing as the sum of four sinusoidal modes in y with
        amplitudes ``a1..a4`` and wavenumbers ``params.k1..k4``; the y-forcing
        is zero.

        Args:
            action: Control amplitudes, shape (4,).
            params: Environment parameters.

        Returns:
            Tuple ``(forcing_x, forcing_y)`` of physical-space arrays.
        """
        a1, a2, a3, a4 = action
        forcing_x = (
            a1 * jnp.sin(params.k1 * self.y)
            + a2 * jnp.sin(params.k2 * self.y)
            + a3 * jnp.sin(params.k3 * self.y)
            + a4 * jnp.sin(params.k4 * self.y)
        )
        forcing_y = jnp.zeros_like(self.y)
        return forcing_x, forcing_y

    def _rollout(
        self,
        omega_hat0: jnp.ndarray,
        params: KolmogorovFlowParams,
        control_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Roll the pseudo-spectral solver forward for one ``action_time`` window.

        Uses the integrator built at construction with the default parameters'
        ``dt`` and ``save_time``.

        Args:
            omega_hat0: Initial spectral vorticity field.
            params: Environment parameters (unused; the rollout settings come
                from ``self.default_params``).
            control_field: Optional tuple ``(forcing_x, forcing_y)`` of
                physical-space control fields applied throughout the rollout.

        Returns:
            Tuple ``(final_state_hat, trajectory)``: the final spectral
            vorticity and the stacked saved states along the rollout.
        """
        default = self.default_params
        dt = float(default.dt)
        save_n = int(default.save_time)
        action_time = float(default.action_time)

        final_state, trajectory = self.integrator.solve(
            dt=dt,
            flow=self.flow,
            t_span=(0.0, action_time),
            save_n=save_n,
            initial_state=omega_hat0,
            control_field=control_field,
        )
        return final_state, trajectory

    def _calculate_velocity_point(self, omega_hat: jnp.ndarray, i: int, j: int):
        """Evaluate the real-space velocity at grid point ``(i, j)`` from a spectral vorticity field.

        Uses the precomputed Fourier mesh cached at construction.

        Args:
            omega_hat: Spectral vorticity field.
            i: x-index of the evaluation point.
            j: y-index of the evaluation point.

        Returns:
            Real-space velocity at that point (2-vector).
        """
        uhat, vhat = compute_velocity_fft(omega_hat, self.kx, self.ky)
        return compute_real_velocity_point(uhat, vhat, i, j)

    def _trajectory_mean_obs(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """Compute the time-mean velocity-magnitude observation over a rollout trajectory.

        For each spectral vorticity snapshot the velocity is reconstructed in
        Fourier space, transformed to physical space on the full grid once,
        subsampled on a regular ``obs_size x obs_size`` grid (stride
        ``grid_size // obs_size``, at least 1), converted to velocity
        magnitude, and finally averaged over the snapshots.

        Args:
            trajectory: Array of spectral vorticity snapshots with a leading
                time axis.

        Returns:
            Flattened array of shape ``(obs_size * obs_size,)`` with the
            trajectory-mean velocity magnitude at each observation point.
        """
        stride_x = max(1, self.n // self.flow.obs_size)
        stride_y = max(1, self.m // self.flow.obs_size)

        def obs_one_state(omega_hat):
            """Velocity-magnitude observation (subsampled, flattened) for one spectral state."""
            # 1. Compute velocity in Fourier space for the whole grid ONCE
            uhat, vhat = compute_velocity_fft(omega_hat, self.kx, self.ky)

            # 2. Inverse FFT for the whole grid ONCE
            ureal = jnp.fft.irfftn(uhat)
            vreal = jnp.fft.irfftn(vhat)

            # 3. Slice exactly at the stride indices to match the 64 grid points
            u_sampled = ureal[::stride_x, ::stride_y]
            v_sampled = vreal[::stride_x, ::stride_y]

            # 4. Compute magnitude and flatten to 1D array
            obs_matrix = jnp.sqrt(jnp.abs(u_sampled) ** 2 + jnp.abs(v_sampled) ** 2)
            return obs_matrix.flatten()

        return jnp.mean(jax.vmap(obs_one_state)(trajectory), axis=0)

    def get_obs(
        self,
        state: KolmogorovFlowState,
        params: KolmogorovFlowParams,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Compute the observation as the time-mean velocity magnitude over the rollout.

        Args:
            state: Current environment state.
            params: Environment parameters (unused).
            key: Unused; accepted for API compatibility.

        Returns:
            Flattened array of shape ``(params.obs_dim,)`` (see
            :meth:`_trajectory_mean_obs`).
        """
        return self._trajectory_mean_obs(state.trajectory)

    def _avg_tke(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """Average the turbulent kinetic energy over a trajectory of spectral vorticity states.

        Args:
            trajectory: Array of spectral vorticity snapshots with a leading
                time axis.

        Returns:
            Scalar mean TKE over all snapshots.
        """
        def one(omega_hat):
            """TKE of a single spectral vorticity snapshot."""
            return compute_tke(omega_hat, self.kx, self.ky, self.n)

        return jnp.mean(jax.vmap(one)(trajectory))

    def _reward(
        self,
        action: jnp.ndarray,
        trajectory: jnp.ndarray,
        params: KolmogorovFlowParams,
    ) -> jnp.ndarray:
        """Compute the reward for a rollout as negative weighted TKE plus an L1 action penalty.

        Args:
            action: Control amplitudes applied during the rollout.
            trajectory: Spectral vorticity snapshots produced by the rollout.
            params: Environment parameters (``reward_alpha`` weights the TKE
                term; the action penalty has weight 1).

        Returns:
            Scalar reward ``-(reward_alpha * mean_TKE + sum(|action|))``.
        """
        energy = self._avg_tke(trajectory)
        action_penalty = jnp.sum(jnp.abs(action))
        return -(params.reward_alpha * energy + action_penalty)

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: KolmogorovFlowParams,
    ):
        """Reset the environment: initialize the flow and spin it up unactuated.

        The initial divergence-free vorticity field is generated and rolled out
        for one ``action_time`` window with no control; the resulting state
        carries both the final spectral vorticity and the saved trajectory.

        Args:
            key: PRNG key (unused; the reset is deterministic).
            params: Environment parameters.

        Returns:
            Tuple ``(obs, KolmogorovFlowState)`` at RL time 0.
        """
        omega0 = self.flow.initialize_state()

        final_state, trajectory = self._rollout(
            omega_hat0=omega0,
            params=params,
            control_field=None,
        )

        state = KolmogorovFlowState(
            trajectory=trajectory,
            omega_hat=final_state,
            time=jnp.array(0),
            terminal=jnp.array(False),
        )
        obs = self.get_obs(state, params, key)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: KolmogorovFlowState,
        action: chex.Array,
        params: KolmogorovFlowParams,
    ):
        """Advance the environment by one RL step under the given action.

        The action is clipped to the parameter bounds, converted into a
        physical-space control field of four sinusoidal modes, and the flow is
        rolled out for one ``action_time`` window. The observation is the
        time-mean velocity magnitude over the rollout and the reward is the
        weighted mean TKE plus the L1 action penalty (negated).

        Args:
            key: PRNG key (unused; the dynamics are deterministic).
            state: Current environment state.
            action: Control amplitudes for the four sinusoidal modes.
            params: Environment parameters.

        Returns:
            Tuple ``(obs, next_state, reward, done, info)`` where ``info``
            contains ``"discount"`` and ``"mean_tke"``.
        """
        action = self._clip_action(action, params)
        control_field = self._control_field(action, params)

        final_state, trajectory = self._rollout(
            omega_hat0=state.omega_hat,
            params=params,
            control_field=control_field,
        )

        next_state = KolmogorovFlowState(
            trajectory=trajectory,
            omega_hat=final_state,
            time=state.time + 1,
            terminal=jnp.array(False),
        )

        obs = self.get_obs(next_state, params, key)
        reward = self._reward(action, trajectory, params)
        done = self.is_terminal(next_state, params)

        info = {
            "discount": self.discount(next_state, params),
            "mean_tke": self._avg_tke(trajectory),
        }
        return obs, next_state, reward, done, info
