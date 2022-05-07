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

        return flow

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

class IPCS_diff(TransientSolver):
    def __init__(self, flow: FlowConfig, dt: float,
            callbacks: Optional[Iterable[Callable]] = []):
        """
        Modified form of IPCS solver that is differentiable with respect to the control parameters

        This is slightly slower because the SNES objects are reinitialized every time step
            (might be able to get speed and differentiability with low-level access to the KSP objects?)
        """
        super().__init__(flow, dt)
        self.callbacks = callbacks
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
        self.a1 = fd.lhs(F1)
        self.L1 = fd.rhs(F1)

        # Poisson equation
        a2 = dot(nabla_grad(p), nabla_grad(s))*dx
        self.L2 = dot(nabla_grad(self.p_n), nabla_grad(s))*dx - (1/k)*div(self.u)*s*dx

        # Projection step (pressure correction)
        a3 = dot(u, v)*dx
        self.L3 = dot(self.u, v)*dx - k*dot(nabla_grad(self.p - self.p_n), v)*dx

        # Assemble matrices
        self.A1 = fd.assemble(self.a1, bcs=self.bcu)
        self.A2 = fd.assemble(a2, bcs=self.bcp)
        self.A3 = fd.assemble(a3)

    def step(self, iter):
        # Step 1: Tentative velocity step
        self.bcu = self.flow.collect_bcu()
        self.A1 = fd.assemble(self.a1, bcs=self.bcu)
        b1 = fd.assemble(self.L1, bcs=self.bcu)
        fd.solve(self.A1, self.u.vector(), b1, solver_parameters={
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg"
        })

        # Step 2: Pressure correction step
        b2 = fd.assemble(self.L2, bcs=self.bcp)
        fd.solve(self.A2, self.p.vector(), b2, solver_parameters={
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg"
        })

        # Step 3: Velocity correction step
        b3 = fd.assemble(self.L3)
        fd.solve(self.A3, self.u.vector(), b3, solver_parameters={
            "ksp_type": "cg",
            "pc_type": "sor"
        })

        # Update previous solution
        self.u_n.assign(self.u)
        self.p_n.assign(self.p)

        return self.flow

METHODS = {'IPCS': IPCS, 'IPCS_diff': IPCS_diff}
def integrate(flow, t_span, dt, method='IPCS', 
        callbacks=[], **options):
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")
    
    method = METHODS[method]
    solver = method(flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks)