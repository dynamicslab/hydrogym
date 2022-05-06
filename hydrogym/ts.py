import numpy as np
import firedrake as fd
from firedrake import dx, ds, lhs, rhs
from ufl import inner, dot, nabla_grad, div

# Typing
from .core import FlowConfig
from typing import Optional, Iterable, Callable, Tuple


class TransientSolver:
    def __init__(self, flow: FlowConfig, dt: float):
        self.flow = flow
        self.dt = dt

    def solve(self, t_span: Tuple[float, float], callbacks:Optional[Iterable[Callable]] = []):
        for iter, t in enumerate(np.arange(*t_span, self.dt)):
            flow = self.step(iter)
            for cb in callbacks:
                cb(iter, t, flow)

class IPCS(TransientSolver):
    def __init__(self, flow: FlowConfig, dt: float, **kwargs):
        super().__init__(flow, dt)
        self.initialize_operators()

    def initialize_operators(self):
        # Setup forms
        flow = self.flow
        k = fd.Constant(self.dt)
        nu = fd.Constant(1/flow.Re)

        flow.init_bcs()
        V, Q = flow.function_spaces(mixed=False)

        # Boundary conditions
        self.bcu = flow.collect_bcu()
        self.bcp = flow.collect_bcp()

        # Trial/test functions for linear problems
        u = fd.TrialFunction(V)
        p = fd.TrialFunction(Q)
        v = fd.TestFunction(V)
        s = fd.TestFunction(Q)

        # Actual solution (references the underlying Flow object)
        self.u, self.p = flow.u, flow.p

        # Previous solution for multistep scheme
        self.u_n = self.u.copy(deepcopy=True)
        self.p_n = self.p.copy(deepcopy=True)

        # Combinations of functions for form construction
        U = 0.5*(self.u_n + u)  # Average for semi-implicit
        u_t = (u - self.u_n)/k  # Time derivative

        # Velocity predictor
        F1 = dot(u_t, v)*dx \
            + dot(dot(self.u_n, nabla_grad(self.u_n)), v)*dx \
            + inner(flow.sigma(U, self.p_n), flow.epsilon(v))*dx \
            + dot(self.p_n*flow.n, v)*ds - dot(nu*nabla_grad(U)*flow.n, v)*ds
            # - dot(f, v)*self.dx
        vel_prob = fd.LinearVariationalProblem(lhs(F1), rhs(F1), self.u, bcs=flow.collect_bcu())
        self.predictor = fd.LinearVariationalSolver(vel_prob, solver_parameters={
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg"
        })

        # Poisson equation
        a2 = dot(nabla_grad(p), nabla_grad(s))*dx
        L2 = dot(nabla_grad(self.p_n), nabla_grad(s))*dx - (1/k)*div(self.u)*s*dx
        poisson_prob = fd.LinearVariationalProblem(a2, L2, self.p, bcs=flow.collect_bcp())
        self.poisson = fd.LinearVariationalSolver(poisson_prob, solver_parameters = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg"
        })

        # Projection step (pressure correction)
        a3 = dot(u, v)*dx
        L3 = dot(self.u, v)*dx - k*dot(nabla_grad(self.p - self.p_n), v)*dx
        proj_prob = fd.LinearVariationalProblem(a3, L3, self.u)
        self.projection = fd.LinearVariationalSolver(proj_prob, solver_parameters = {
            'ksp_type': 'cg',
            'pc_type': 'sor'
        })

    def step(self, iter):
        # Step 1: Tentative velocity step
        self.predictor.solve()
        self.poisson.solve()
        self.projection.solve()

        # Update previous solution
        self.u_n.assign(self.u)
        self.p_n.assign(self.p)

        return self.flow

METHODS = {'IPCS': IPCS}

def integrate(flow, t_span, dt, method='IPCS', 
        callbacks=[], **options):
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")
    
    method = METHODS[method]
    solver = method(flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks)