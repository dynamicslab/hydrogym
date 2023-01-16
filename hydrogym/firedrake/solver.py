import firedrake as fd
import numpy as np
from firedrake import logging
from ufl import div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from hydrogym.core import TransientSolver

from .flow import FlowConfig
from .utils import get_array, set_from_array, white_noise


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
            + inner(p * self.flow.n, v) * ds
            - inner(self.flow.nu * nabla_grad(u) * self.flow.n, v) * ds
            + inner(div(u), s) * dx
        )
        return F


class IPCS(TransientSolver):
    def __init__(
        self,
        flow: FlowConfig,
        dt: float = None,
        eta: float = 0.0,
        debug=False,
        **kwargs,
    ):
        super().__init__(flow, dt)
        self.debug = debug

        self.forcing_config = {
            "eta": eta,
            "n_samples": kwargs.get("max_iter", int(1e8)),
            "cutoff": kwargs.get("noise_cutoff", 0.01 / flow.TAU),
        }

        self.reset()

    def reset(self):
        self.initialize_functions()

        # Set up random forcing (if applicable)
        self.initialize_forcing(**self.forcing_config)

        self.initialize_operators()

        self.B = self.flow.control_vec()

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
        flow = self.flow

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
        # Update perturbations (if applicable)
        self.eta.assign(self.noise[self.noise_idx])

        # Step 1: Tentative velocity step
        logging.log(logging.DEBUG, f"iter: {iter}, solving velocity predictor")
        self.predictor.solve()
        if control is not None:
            control = self.flow.update_actuators(control, self.dt)
            for (B, ctrl) in zip(self.B, control):
                Bu, _ = B.split()
                self.u += Bu * fd.Constant(ctrl)

        logging.log(logging.DEBUG, "Velocity predictor done, solving Poisson")
        self.poisson.solve()
        logging.log(logging.DEBUG, "Poisson done, solving projection step")
        self.projection.solve()
        logging.log(logging.DEBUG, "IPCS step finished")

        # Update previous solution
        self.u_n.assign(self.u)
        self.p_n.assign(self.p)

        self.t += self.dt
        self.noise_idx += 1
        assert self.noise_idx < len(self.noise), "Not enough noise samples generated"

        return self.flow

    def linearize(self, qB=None, return_operators=True):
        """
        Return a LinearOperator that can act on numpy arrays (pulled from utils.get_array)
        """
        from scipy.sparse.linalg import LinearOperator

        from .utils import linalg

        flow = self.flow
        k = fd.Constant(self.dt)

        if qB is None:
            uB = flow.u.copy(deepcopy=True)
        else:
            uB = qB.split()[0].copy(deepcopy=True)

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
                u, p = q.split()
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
            for (i, Bi) in enumerate(self.B):
                B[:, i] = get_array(Bi)
            return A, B


METHODS = {"IPCS": IPCS}


def integrate(
    flow, t_span, dt, method="IPCS", callbacks=[], controller=None, **options
):
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")

    solver = METHODS[method](flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks, controller=controller)
