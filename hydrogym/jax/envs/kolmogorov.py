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
    DEFAULT_REYNOLDS = 200
    DEFAULT_WAVENUMBER = 4
    DEFAULT_GRID_SIZE = (64, 64)
    DEFAULT_DOMAIN_X = (0, 2 * jnp.pi)
    DEFAULT_DOMAIN_Y = (0, 2 * jnp.pi)
    DEFAULT_OBS_SIZE = 8  # This correlates to a total observation size of 8x8 = 64.

    def __init__(self, **config):
        self.k = config.get("k", self.DEFAULT_WAVENUMBER)
        self.Re = config.get("Re", self.DEFAULT_REYNOLDS)
        self.grid_size = config.get("grid_size", self.DEFAULT_GRID_SIZE)
        self.domain_x = config.get("domain_x", self.DEFAULT_DOMAIN_X)
        self.domain_y = config.get("domain_y", self.DEFAULT_DOMAIN_Y)
        self.obs_size = config.get("obs_size", self.DEFAULT_OBS_SIZE)
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
        # Calculate velocity point
        kx, ky = self.load_fft_mesh()
        uhat, vhat = compute_velocity_fft(state, kx, ky)
        point_velocity = compute_real_velocity_point(uhat, vhat, k1, k2)

        return point_velocity

    def state(self) -> jnp.array:
        return self.state

    def get_observations(self) -> jnp.array:
        n, m = self.grid_size
        divisor = n // self.obs_size

        def calculate_velocity(trajectory):
            points = [
                self._calculate_velocity_point(trajectory, x, y)
                for x in range(0, n, int(n / divisor))
                for y in range(0, m, int(m / divisor))
            ]
            return jnp.array(points)

        def scan_fn(carry, state):
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
            return jnp.cos(x)

        def dstream_func_dy(x, y):
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
        """Return a copy of the flow state"""
        pass

    @property
    def nu(self):
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

    def nonlinear_terms(self, omega_hat):
        """Computes the explicit (nonlinear) terms in the vorticity equation.
        Uses the stream function to compute velocity components in Fourier space.

        Args:
            omega_hat: fft of vorticity

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
        control_hat = self.control_term(omega_hat)
        advection_hat = dealiasing(advection_hat)  # 2/3 dealiasing rule

        terms = advection_hat + forcing_hat + control_hat
        return terms

    def control_term(self, omega_hat):
        """Computes the user-specified forcing term of the vorticity equation
        Args:
          omega_hat: Fourier transformed vorticity term
          forcing: Forcing function as specified by environment or user
        """
        cfx, cfy = self.flow.control_function

        if cfx is None:
            return jnp.zeros_like(omega_hat)

        cfx_hat = jnp.fft.rfftn(cfx)
        cfy_hat = jnp.fft.rfftn(cfy)

        return self.kx * cfy_hat - self.ky * cfx_hat

    def forcing_term(self):
        """Computes the user-specified forcing term of the vorticity equation
        Args:
          omega_hat: Fourier transformed vorticity term
          forcing: Forcing function as specified by environment or user
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
    trajectory: jnp.ndarray
    omega_hat: jnp.ndarray
    time: jnp.ndarray
    terminal: jnp.ndarray


@struct.dataclass
class KolmogorovFlowParams(EnvParams):
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
    def __init__(
        self,
        env_config: Optional[Dict] = None,
        flow_config: Optional[Dict] = None,
    ):
        super().__init__(env_config)

        self.flow = FlowConfig(**(flow_config or {}))
        self.equation_cls = PseudoSpectralNavierStokes2D
        self.integrator_cls = RungeKuttaCrankNicolson

        self.n, self.m = self.flow.grid_size
        self.x, self.y = self.flow.load_mesh("")
        self.kx, self.ky = self.flow.load_fft_mesh()

    @property
    def name(self) -> str:
        return "KolmogorovFlow"

    @property
    def default_params(self) -> KolmogorovFlowParams:
        return KolmogorovFlowParams(
            action_dim=4,
            obs_dim=self.flow.obs_size**2,
        )

    def action_space(self, params: Optional[KolmogorovFlowParams] = None):
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: KolmogorovFlowParams):
        return spaces.Box(
            low=params.min_obs,
            high=params.max_obs,
            shape=(params.obs_dim,),
        )

    def _control_field(self, action: jnp.ndarray, params: KolmogorovFlowParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a1, a2, a3, a4 = action
        forcing_x = (
            a1 * jnp.sin(params.k1 * self.y)
            + a2 * jnp.sin(params.k2 * self.y)
            + a3 * jnp.sin(params.k3 * self.y)
            + a4 * jnp.sin(params.k4 * self.y)
        )
        forcing_y = jnp.zeros_like(self.y)
        return forcing_x, forcing_y

    def _make_equation(
        self,
        control_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        if control_field is None:
            self.flow.control_function = (None, None)
        else:
            self.flow.control_function = control_field
        return self.equation_cls(self.flow)

    def _rollout(
        self,
        omega_hat0: jnp.ndarray,
        params: KolmogorovFlowParams,
        control_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            final_state_hat, trajectory
        """
        equation = self._make_equation(control_field)

        save_n = params.save_time
        integrator = self.integrator_cls(
            flow=self.flow,
            dt=params.dt,
            save_n=save_n,
            equation=equation,
        )

        # important: solve() uses flow.initialize_state(), so patch flow state first
        self.flow.initialize_state = lambda: omega_hat0

        final_state, trajectory = integrator.solve(
            dt=params.dt,
            flow=self.flow,
            t_span=(0.0, params.action_time),
            save_n=params.save_time,
        )
        return final_state, trajectory

    def _calculate_velocity_point(self, omega_hat: jnp.ndarray, i: int, j: int):
        uhat, vhat = compute_velocity_fft(omega_hat, self.kx, self.ky)
        return compute_real_velocity_point(uhat, vhat, i, j)

    def _trajectory_mean_obs(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        stride_x = max(1, self.n // self.flow.obs_size)
        stride_y = max(1, self.m // self.flow.obs_size)

        def obs_one_state(omega_hat):
            pts = [
                self._calculate_velocity_point(omega_hat, i, j)
                for i in range(0, self.n, stride_x)
                for j in range(0, self.m, stride_y)
            ]
            return jnp.asarray(pts)

        return jnp.mean(jax.vmap(obs_one_state)(trajectory), axis=0)

    def get_obs(
        self,
        state: KolmogorovFlowState,
        params: KolmogorovFlowParams,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        return self._trajectory_mean_obs(state.trajectory)

    def _avg_tke(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        def one(omega_hat):
            return compute_tke(omega_hat, self.kx, self.ky, self.n)

        return jnp.mean(jax.vmap(one)(trajectory))

    def _reward(
        self,
        action: jnp.ndarray,
        trajectory: jnp.ndarray,
        params: KolmogorovFlowParams,
    ) -> jnp.ndarray:
        energy = self._avg_tke(trajectory)
        action_penalty = jnp.sum(jnp.abs(action))
        return -(params.reward_alpha * energy + action_penalty)

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: KolmogorovFlowParams,
    ):
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
