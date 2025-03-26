import numpy as np

from hydrogym.core import TransientSolver
from hydrogym.firedrake import FlowConfig
from hydrogym.firedrake.solvers.stabilization import ns_stabilization
from hydrogym.firedrake.utils import white_noise
from hydrogym.utils import DependencyNotInstalled

try:
    import firedrake as fd
    from firedrake import logging
    from ufl import as_ufl, div, dot, ds, dx, inner, lhs, nabla_grad, rhs
except ImportError as e:
    raise DependencyNotInstalled(
        "Firedrake is not installed, consult `https://www.firedrakeproject.org/install.html` for installation instructions."  # noqa: E501
    ) from e

__all__ = ["NewtonSolver"]


class NewtonSolver:
    def __init__(
        self,
        flow: FlowConfig,
        stabilization: str = "none",
        solver_parameters: dict = {},
    ):
        self.flow = flow
        self.solver_parameters = solver_parameters

        if stabilization not in ns_stabilization:
            raise ValueError(
                f"Stabilization type {stabilization} not recognized. Available options: {ns_stabilization.keys()}"
            )
        self.stabilization_type = ns_stabilization[stabilization]

    def solve(self, q: fd.Function = None):
        """Solve the steady-state problem from initial guess `q`"""
        if q is None:
            q = self.flow.q

        self.flow.init_bcs()

        F = self.steady_form(fd.split(q))  # Nonlinear variational form
        J = fd.derivative(F, q)  # Jacobian with automatic differentiation

        bcs = self.flow.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, q, bcs, J)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters)
        solver.solve()

        return q.copy(deepcopy=True)

    def steady_form(self, q: fd.Function, q_test=None):
        (u, p) = q
        if q_test is None:
            (v, s) = fd.TestFunctions(self.flow.mixed_space)
        else:
            (v, s) = q_test

        F = self.flow.residual((u, p), q_test=(v, s))
        stab = self.stabilization_type(
            self.flow,
            q_trial=(u, p),
            q_test=(v, s),
            wind=u,
        )
        F = stab.stabilize(F)

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
        self.reset()

    def reset(self):
        super().reset()

        self.initialize_functions()

        self.initialize_operators()

    def initialize_functions(self):
        pass

    def initialize_operators(self):
        pass
