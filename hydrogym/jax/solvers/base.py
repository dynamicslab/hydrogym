from typing import Callable, Iterable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tree_math
from jax import lax

from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.equation import IMEXEquation

_alpha_RK4 = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
_beta_RK4 = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
_gammas_RK4 = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]


class VelocityState(NamedTuple):
    """Three-component velocity (or velocity-spectrum) state, one array per direction.

    Attributes:
        u: x-component; physical or spectral depending on usage.
        v: y-component.
        w: z-component.
    """

    u: jnp.ndarray  # physical or spectral depending on usage
    v: jnp.ndarray
    w: jnp.ndarray


class RungeKuttaCrankNicolson(TransientSolver):
    """IMEX Crank-Nicolson Runge-Kutta transient solver for a split equation.

    Advances an :class:`~hydrogym.jax.equation.IMEXEquation` with the nonlinear
    terms treated explicitly and the linear terms treated implicitly, using a
    low-storage 5-stage scheme; the whole rollout is expressed as a
    ``lax.scan`` so it can be JIT-compiled.
    """

    def __init__(self, flow: PDEBase, dt: float, save_n: int, equation: IMEXEquation, **kwargs):
        """Initialize the solver.

        Args:
            flow: Flow configuration providing the state (used by the base class).
            dt: Timestep size.
            save_n: Number of (inner) steps between saved states.
            equation: The split (IMEX) equation being integrated.
            **kwargs: Ignored; accepted for API compatibility.
        """
        self.save_n = save_n
        self.dt = dt
        self.flow = flow
        self.equation = equation
        super().__init__(flow, dt)

    def RK4_CN(self, control_field=None):
        """
        Crank-Nicolson RK4 implicit-explicit time stepping scheme.
        Low storage scheme inspired by [1]. Method described in [2].

        Implicit-Explicit timestepping for an ODE of the form:
          ∂u/∂t = g(u,t) + l(u,t)
        where g(u,t) is the nonlinear advection term and l(u,t) is the linear diffusion term.

        [1] Kochkov, D., et. al. (2021) https://doi.org/10.1073/pnas.2101784118
        [2] PK Sweby, (1984). SIAM journal on numerical analysis 21, Appendix D.
        """
        nonlinear_with_control = lambda u: self.equation.nonlinear_terms(u, control_field)
        unwrapped_nonlinear = tree_math.unwrap(nonlinear_with_control)
        unwrapped_linear = tree_math.unwrap(self.equation.linear_terms)
        y = tree_math.unwrap(self.equation.implicit_timestep, vector_argnums=0)

        @tree_math.wrap
        def time_step_fn(u):
            """Advance a (tree-wrapped) state by one full 5-stage IMEX-RK step.

            Each stage combines the explicit nonlinear term (with the
            low-storage recursion ``h``) with a partially implicit
            Crank-Nicolson solve of the linear term.

            Args:
                u: Current state (tree of arrays).

            Returns:
                The state after one timestep.
            """
            h = 0
            for k in range(5):
                h = unwrapped_nonlinear(u) + _beta_RK4[k] * h
                mu = 0.5 * self.dt * (_alpha_RK4[k + 1] - _alpha_RK4[k])
                yn = u + _gammas_RK4[k] * self.dt * h + mu * unwrapped_linear(u)
                u = y(yn, mu)
            return u

        return time_step_fn

    def step(self, flow: PDEBase, dt: float, save_n: int, callbacks: Callable, control_field=None):
        """Build a ``lax.scan`` function that advances a state by ``save_n`` timesteps.

        Note this returns the *scan function*, not a stepped state: the caller
        nests it inside an outer scan (see :meth:`solve`).

        Args:
            flow: Flow configuration (unused here; the equation carries the
                dynamics).
            dt: Timestep size (unused here; the one fixed at construction
                applies).
            save_n: Number of timesteps each inner scan performs.
            callbacks: Unused; per-step callback tracking is not possible
                inside compiled scans.
            control_field: Optional control input forwarded to the equation's
                nonlinear terms at every step.

        Returns:
            Callable mapping an initial state to the state ``save_n``
            timesteps later.
        """
        func = self.RK4_CN(control_field=control_field)

        def inner_scan(initialization):
            """Run ``save_n`` RK4-CN timesteps from ``initialization``, keeping only the final state."""
            f = lambda init, inputs: (func(init), init)
            final_state, outputs = lax.scan(f, initialization, xs=None, length=save_n)
            return final_state

        return inner_scan

    def solve(
        self,
        dt: float,
        flow: PDEBase,
        t_span: Tuple[float, float],
        callbacks: Iterable[CallbackBase] = [],
        controller: Callable = None,
        save_n: int = 1,
        initial_state=None,
        control_field=None,
    ) -> PDEBase:
        """Integrate the equation from t=0 to ``t_span[1]`` with nested lax scans.

        The rollout is run as an outer scan of inner scans, each of ``save_n // dt``
        timesteps, so the saved trajectory holds one state per ``save_n`` time
        units. Callbacks are invoked once at the end (per-iteration callback
        tracking is not possible through the compiled scans).

        Args:
            dt: Timestep size.
            flow: Flow configuration; also supplies the initial state when
                ``initial_state`` is None.
            t_span: ``(t0, t1)`` integration interval; ``t1`` must be at least 1.
            callbacks: Callbacks invoked on the flow after the rollout.
            controller: Unused; accepted for API compatibility with the
                hydrogym solver interface.
            save_n: Time interval between saved trajectory states.
            initial_state: Optional starting state (FFT vorticity field);
                defaults to ``flow.initialize_state()``.
            control_field: Optional control input forwarded to the equation's
                nonlinear terms at every step.

        Returns:
            Tuple ``(final_state, outputs)`` where ``outputs`` is the stacked
            trajectory of states at each outer-scan step (also stored on
            ``flow.vorticity``).

        Raises:
            ValueError: If the end time in ``t_span`` is less than 1.
        """
        end_time = t_span[1]
        if end_time < 1:
            raise ValueError(
                "This flow configuration requires the end time to be at least 1. Please adjust t_span and run again."
            )

        initialization = flow.initialize_state() if initial_state is None else initial_state
        step_to_save = int(save_n // dt)

        total_steps = int(end_time // dt)
        outer_steps = int(total_steps // step_to_save)

        inner_scan = self.step(flow, dt, step_to_save, callbacks, control_field=control_field)

        outer_scan = lambda init, inputs: (inner_scan(init), inner_scan(init))

        final_state, outputs = lax.scan(outer_scan, initialization, xs=None, length=outer_steps)
        flow.vorticity = outputs
        # Dummy values for iter, t for hydrogym api callback function.
        # Optimized iteration through JAX (with scan) is not the same as native python,
        # and the iterations can not easily be tracked.
        for cb in callbacks:
            cb(flow)
        return final_state, outputs


class RungeKutta4:
    """Explicit classical 4th-order Runge-Kutta stepper for a velocity state.

    Each step evaluates the equation's full right-hand side four times,
    projects the result back onto the constraint manifold
    (``equation.project``), and optionally applies a constant-mass-flux
    correction by rescaling the mean streamwise velocity in physical space
    before re-applying the boundary conditions.
    """

    def __init__(self, equation, dt: float, save_n: int, **kwargs):
        """Initialize the integrator.

        Args:
            equation: Equation object providing ``rhs``, ``project``,
                ``to_physical``, ``to_spectral`` and ``enforce_noslip``.
            dt: Default timestep used when ``rk4_step`` gets no explicit ``dt``.
            save_n: Number of steps between saves (stored; not used by
                ``rk4_step`` itself).
            **kwargs: Ignored; accepted for API compatibility.
        """
        self.save_n = int(save_n)
        self.dt = dt
        self.equation = equation

    def rk4_step(
        self,
        state_hat: "VelocityState",
        dt: float = None,
        action=None,
        t: float = 0.0,
        fx=0.0,
        fy=0.0,
        fz=0.0,
        enforce_const_massflux=True,
        target_bulk_u=8.0,
    ):
        """Advance the state by one RK4 step, then project and correct mass flux.

        Args:
            state_hat: Current state (spectral velocity components).
            dt: Timestep; defaults to the value given at construction.
            action: Control (actuation) input forwarded to the equation's rhs,
                projection, and boundary-condition enforcement.
            t: Current time, used for time-dependent BCs and forcing.
            fx: x-direction body forcing.
            fy: y-direction body forcing.
            fz: z-direction body forcing.
            enforce_const_massflux: If True, rescale the streamwise velocity in
                physical space so the bulk velocity equals ``target_bulk_u``,
                then re-apply the no-slip/jet boundary conditions.
            target_bulk_u: Target bulk (domain-mean) streamwise velocity for
                the mass-flux correction.

        Returns:
            The new state in spectral form.
        """
        eq = self.equation
        dt = self.dt if dt is None else dt

        def add_state(a, b, alpha=1.0):
            """Component-wise ``a + alpha * b`` for two :class:`VelocityState` values."""
            return VelocityState(
                a.u + alpha * b.u,
                a.v + alpha * b.v,
                a.w + alpha * b.w,
            )

        k1 = eq.rhs(state_hat, action=action, t=t, fx=fx, fy=fy, fz=fz)
        s2 = add_state(state_hat, k1, 0.5 * dt)

        k2 = eq.rhs(s2, action=action, t=t + 0.5 * dt, fx=fx, fy=fy, fz=fz)
        s3 = add_state(state_hat, k2, 0.5 * dt)

        k3 = eq.rhs(s3, action=action, t=t + 0.5 * dt, fx=fx, fy=fy, fz=fz)
        s4 = add_state(state_hat, k3, dt)

        k4 = eq.rhs(s4, action=action, t=t + dt, fx=fx, fy=fy, fz=fz)

        u_star = VelocityState(
            state_hat.u + (dt / 6.0) * (k1.u + 2 * k2.u + 2 * k3.u + k4.u),
            state_hat.v + (dt / 6.0) * (k1.v + 2 * k2.v + 2 * k3.v + k4.v),
            state_hat.w + (dt / 6.0) * (k1.w + 2 * k2.w + 2 * k3.w + k4.w),
        )

        new_state = eq.project(u_star, dt, action=action, t=t + dt)

        if enforce_const_massflux:
            state_phys = eq.to_physical(new_state)
            bulk_u = jnp.mean(state_phys.u)
            delta = target_bulk_u - bulk_u

            corrected_phys = VelocityState(
                state_phys.u + delta,
                state_phys.v,
                state_phys.w,
            )
            u, v, w = eq.enforce_noslip(
                corrected_phys.u,
                corrected_phys.v,
                corrected_phys.w,
                action=action,
                t=t + dt,
            )
            new_state = eq.to_spectral(VelocityState(u, v, w))

        return new_state
