import firedrake as fd
import numpy as np
from firedrake import logging
from ufl import as_ufl, div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from hydrogym.core import TransientSolver
from hydrogym.firedrake import FlowConfig
from hydrogym.firedrake.utils import white_noise

__all__ = ["NewtonSolver"]


class NewtonSolver:
    def __init__(self, flow: FlowConfig, solver_parameters: dict = {}):
        self.flow = flow
        self.solver_parameters = solver_parameters

    def solve(self, q: fd.Function = None):
        """Solve the steady-state problem from initial guess `q`"""
        if q is None:
            q = self.flow.q

        self.flow.init_bcs(mixed=True)

        F = self.steady_form(q)  # Nonlinear variational form
        J = fd.derivative(F, q)  # Jacobian with automatic differentiation

        bcs = self.flow.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, q, bcs, J)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=self.solver_parameters
        )
        solver.solve()

        return q.copy(deepcopy=True)

    def steady_form(self, q: fd.Function):
        (u, p) = fd.split(q)
        (v, s) = fd.TestFunctions(self.flow.mixed_space)

        F = (
            inner(dot(u, nabla_grad(u)), v) * dx
            + inner(self.flow.sigma(u, p), self.flow.epsilon(v)) * dx
            + inner(div(u), s) * dx
        )
        return F


class NavierStokesTransientSolver(TransientSolver):
    def __init__(
        self,
        flow: FlowConfig,
        dt: float = None,
        eta: float = 0.0,
        debug: bool = False,
        max_noise_iter: int = int(1e8),
        noise_cutoff: float = None,
    ):
        super().__init__(flow, dt)
        self.debug = debug

        self.forcing_config = {
            "eta": eta,
            "n_samples": max_noise_iter,
            "cutoff": noise_cutoff or 0.01 / flow.TAU,
        }

        self.reset()

    def reset(self):
        super().reset()

        self.initialize_functions()

        # Set up random forcing (if applicable)
        self.initialize_forcing(**self.forcing_config)

        self.initialize_operators()

    def initialize_forcing(self, eta, n_samples, cutoff):
        logging.log(logging.INFO, f"Initializing forcing with amplitude {eta}")
        self.f = self.flow.body_force
        self.eta = fd.Constant(0.0)  # Current forcing amplitude

        if eta > 0:
            self.noise = eta * white_noise(
                n_samples=n_samples,
                fs=1 / self.dt,
                cutoff=cutoff,
            )
        else:
            self.noise = np.zeros(n_samples)
        self.noise_idx = 0

    def initialize_functions(self):
        pass

    def initialize_operators(self):
        pass
