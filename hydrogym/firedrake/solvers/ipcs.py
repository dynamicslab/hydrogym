import firedrake as fd
import numpy as np
from firedrake import logging
from ufl import div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from ..utils import get_array, set_from_array
from .base import NavierStokesTransientSolver

__all__ = ["IPCS"]


class IPCS(NavierStokesTransientSolver):
    def initialize_functions(self):
        flow = self.flow
        self.f = flow.body_force

        if flow.velocity_order < 2:
            raise ValueError("IPCS requires at least second-order velocity elements")

        # Trial/test functions for linear sub-problems
        V, Q = flow.function_spaces(mixed=False)
        u = fd.TrialFunction(V)
        p = fd.TrialFunction(Q)
        v = fd.TestFunction(V)
        s = fd.TestFunction(Q)

        self.q_trial = (u, p)
        self.q_test = (v, s)

        # Actual solution (references the underlying Flow object)
        self.u, self.p = flow.u, flow.p

        # Previous solution for multistep scheme
        self.u_n = self.u.copy(deepcopy=True)
        self.p_n = self.p.copy(deepcopy=True)

    def initialize_operators(self):
        # Setup forms
        flow = self.flow
        k = fd.Constant(self.dt)

        flow.init_bcs(mixed=False)

        # Boundary conditions
        self.bcu = flow.collect_bcu()
        self.bcp = flow.collect_bcp()

        (u, p) = self.q_trial
        (v, s) = self.q_test

        # Combinations of functions for form construction
        U = 0.5 * (self.u_n + u)  # Average for semi-implicit
        u_t = (u - self.u_n) / k  # Time derivative

        # Velocity predictor
        F1 = (
            dot(u_t, v) * dx
            + dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx
            + inner(flow.sigma(U, self.p_n), flow.epsilon(v)) * dx
            + dot(self.p_n * flow.n, v) * ds
            - dot(flow.nu * nabla_grad(U) * flow.n, v) * ds
            - dot(self.eta * self.f, v) * dx
        )
        vel_prob = fd.LinearVariationalProblem(
            lhs(F1), rhs(F1), self.u, bcs=flow.collect_bcu()
        )
        solver_parameters = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }
        if self.debug:
            solver_parameters["ksp_monitor_true_residual"] = None
        self.predictor = fd.LinearVariationalSolver(
            vel_prob, solver_parameters=solver_parameters
        )

        # Poisson equation
        a2 = inner(nabla_grad(p), nabla_grad(s)) * dx
        L2 = (
            inner(nabla_grad(self.p_n), nabla_grad(s)) * dx
            - (1 / k) * div(self.u) * s * dx
        )
        poisson_prob = fd.LinearVariationalProblem(
            a2, L2, self.p, bcs=flow.collect_bcp()
        )
        self.poisson = fd.LinearVariationalSolver(
            poisson_prob, solver_parameters=solver_parameters
        )

        # Projection step (pressure correction)
        a3 = inner(u, v) * dx
        L3 = inner(self.u, v) * dx - k * inner(nabla_grad(self.p - self.p_n), v) * dx
        proj_prob = fd.LinearVariationalProblem(a3, L3, self.u)
        self.projection = fd.LinearVariationalSolver(
            proj_prob, solver_parameters={"ksp_type": "cg", "pc_type": "sor"}
        )

    def step(self, iter, control=None):
        bc_scale = self.flow.advance_time(self.dt, control)
        self.flow.set_control(bc_scale)

        # Step 1: Velocity predictor step
        logging.log(logging.DEBUG, f"iter: {iter}, solving velocity predictor")
        self.predictor.solve()

        # Step 2: Pressure Poisson equation
        logging.log(logging.DEBUG, "Velocity predictor done, solving Poisson")
        self.poisson.solve()

        # Step 3: Projection step (pressure correction)
        logging.log(logging.DEBUG, "Poisson done, solving projection step")
        self.projection.solve()

        logging.log(logging.DEBUG, "IPCS step finished")

        # Update previous solution
        self.u_n.assign(self.u)
        self.p_n.assign(self.p)

        return self.flow

    def linearize(self, qB=None, return_operators=True):
        """
        Return a LinearOperator that can act on numpy arrays (pulled from utils.get_array)
        """
        from scipy.sparse.linalg import LinearOperator

        from ..utils import linalg

        flow = self.flow
        k = fd.Constant(self.dt)

        if qB is None:
            uB = flow.u.copy(deepcopy=True)
        else:
            uB = qB.subfunctions[0].copy(deepcopy=True)

        flow.linearize_bcs(mixed=False)
        (u, p) = self.q_trial
        (v, s) = self.q_test

        U = 0.5 * (self.u_n + u)  # Average for semi-implicit
        u_t = (u - self.u_n) / k  # Time derivative

        F1 = (
            dot(u_t, v) * dx
            + dot(dot(uB, nabla_grad(self.u_n)), v) * dx
            + dot(dot(self.u_n, nabla_grad(uB)), v) * dx
            + inner(flow.sigma(U, self.p_n), flow.epsilon(v)) * dx
            + dot(self.p_n * flow.n, v) * ds
            - dot(flow.nu * nabla_grad(U) * flow.n, v) * ds
        )
        a1 = lhs(F1)
        L1 = rhs(F1)
        vel_prob = fd.LinearVariationalProblem(a1, L1, self.u, bcs=flow.collect_bcu())
        solver_parameters = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }
        self.predictor = fd.LinearVariationalSolver(
            vel_prob, solver_parameters=solver_parameters
        )

        # Adjoint operator
        adj_vel_prob = fd.LinearVariationalProblem(
            linalg.adjoint(a1), L1, self.u, bcs=flow.collect_bcu()
        )
        self.adj_predictor = fd.LinearVariationalSolver(
            adj_vel_prob, solver_parameters=solver_parameters
        )

        # Poisson equation
        a2 = inner(nabla_grad(p), nabla_grad(s)) * dx
        L2 = (
            inner(nabla_grad(self.p_n), nabla_grad(s)) * dx
            - (1 / k) * div(self.u) * s * dx
        )
        poisson_prob = fd.LinearVariationalProblem(
            a2, L2, self.p, bcs=flow.collect_bcp()
        )
        self.poisson = fd.LinearVariationalSolver(
            poisson_prob, solver_parameters=solver_parameters
        )

        # Projection step (pressure correction)
        a3 = inner(u, v) * dx
        L3 = inner(self.u, v) * dx - k * inner(nabla_grad(self.p - self.p_n), v) * dx
        proj_prob = fd.LinearVariationalProblem(a3, L3, self.u)
        self.projection = fd.LinearVariationalSolver(
            proj_prob, solver_parameters={"ksp_type": "cg", "pc_type": "sor"}
        )

        if return_operators:
            assert fd.COMM_WORLD.size == 1, "Must run in serial for scipy operators"
            self.B = flow.control_vec()

            q = flow.q.copy(deepcopy=True)

            def matvec(q_vec, mode):
                set_from_array(q, q_vec)
                u, p = q.subfunctions
                self.u_n.assign(q.sub(0))
                self.p_n.assign(q.sub(1))

                self.step(
                    0, mode=mode
                )  # Updates self.u, self.p, and by extension flow.q = [self.u, self.p]
                return get_array(flow.q)

            def lmatvec(q_vec):
                return matvec(q_vec, mode="direct")

            def rmatvec(q_vec):
                return matvec(q_vec, mode="adjoint")

            N = q.vector().size()
            A = LinearOperator(shape=(N, N), matvec=lmatvec, rmatvec=rmatvec)
            B = np.zeros((N, len(self.B)))
            for i, Bi in enumerate(self.B):
                B[:, i] = get_array(Bi)
            return A, B
