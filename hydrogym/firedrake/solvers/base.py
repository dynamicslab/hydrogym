import firedrake as fd
import numpy as np
from ufl import as_ufl, div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from hydrogym.firedrake import FlowConfig

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
