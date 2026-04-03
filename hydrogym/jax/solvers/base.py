from typing import Callable, Iterable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tree_math
from jax import lax
from jax.experimental import checkify

from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.equation import IMEXEquation
from hydrogym.jax.flow import FlowConfig

_alpha_RK4 = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
_beta_RK4 = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
_gammas_RK4 = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]


class VelocityState(NamedTuple):
    u: jnp.ndarray  # physical or spectral depending on usage
    v: jnp.ndarray
    w: jnp.ndarray


class RungeKuttaCrankNicolson(TransientSolver):
    def __init__(self, flow: FlowConfig, dt: float, save_n: int, equation: IMEXEquation, **kwargs):
        self.save_n = save_n
        self.dt = dt
        self.flow = flow
        self.equation = equation
        super().__init__(flow, dt)

    def RK4_CN(self):
        """
        Crank-Nicolson RK4 implicit-explicit time stepping scheme.
        Low storage scheme inspired by [1]. Method described in [2].

        Implicit-Explicit timestepping for an ODE of the form:
          ∂u/∂t = g(u,t) + l(u,t)
        where g(u,t) is the nonlinear advection term and l(u,t) is the linear diffusion term.

        [1] Kochkov, D., et. al. (2021) https://doi.org/10.1073/pnas.2101784118
        [2] PK Sweby, (1984). SIAM journal on numerical analysis 21, Appendix D.
        """
        unwrapped_nonlinear = tree_math.unwrap(self.equation.nonlinear_terms)
        unwrapped_linear = tree_math.unwrap(self.equation.linear_terms)
        y = tree_math.unwrap(self.equation.implicit_timestep, vector_argnums=0)

        @tree_math.wrap
        def time_step_fn(u):
            h = 0
            for k in range(5):
                h = unwrapped_nonlinear(u) + _beta_RK4[k] * h
                mu = 0.5 * self.dt * (_alpha_RK4[k + 1] - _alpha_RK4[k])
                yn = u + _gammas_RK4[k] * self.dt * h + mu * unwrapped_linear(u)
                u = y(yn, mu)
            return u

        return time_step_fn

    def step(self, flow: FlowConfig, dt: float, save_n: int, callbacks: Callable):
        """
        Lax.scan to iteratively apply a function given an initial value

        Args:
            initialization(grid array): the initial fft vorticity field
            steps (int):  number of timesteps
            save_n (int): save every n steps
            ignore_intermediate_steps (bool): if saving every n steps, ignore intermediate steps.
                                              this drastically reduces the memory requirements.

        """
        func = self.RK4_CN()

        def inner_scan(initialization):
            f = lambda init, inputs: (func(init), init)
            final_state, outputs = lax.scan(f, initialization, xs=None, length=save_n)
            return final_state

        return inner_scan

    def solve(
        self,
        dt: float,
        flow: FlowConfig,
        t_span: Tuple[float, float],
        callbacks: Iterable[CallbackBase] = [],
        controller: Callable = None,
        save_n: int = 1,
    ) -> PDEBase:
        end_time = t_span[1]
        checkify.check(
            end_time < 1,
            "This flow configuration requires the end time to be at least 1. Please adjust t_span and run again.",
        )

        initialization = flow.initialize_state()
        step_to_save = int(save_n // dt)

        total_steps = int(end_time // dt)
        outer_steps = int(total_steps // step_to_save)

        inner_scan = self.step(flow, dt, step_to_save, callbacks)

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
    def __init__(self, equation, dt: float, save_n: int, **kwargs):
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
        eq = self.equation
        dt = self.dt if dt is None else dt

        def add_state(a, b, alpha=1.0):
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
