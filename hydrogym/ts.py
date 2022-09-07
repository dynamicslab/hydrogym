from typing import Callable, Iterable, Optional, Tuple

import firedrake as fd
import numpy as np
import ufl
from firedrake import ds, dx, lhs, logging, rhs
from ufl import div, dot, inner, nabla_grad

# Typing
from .core import FlowConfig
from .utils import white_noise


class TransientSolver:
    def __init__(self, flow: FlowConfig, dt: float):
        self.flow = flow
        self.dt = dt

        self.t = 0.0

    def solve(
        self, t_span: Tuple[float, float], callbacks: Optional[Iterable[Callable]] = []
    ):
        for iter, t in enumerate(np.arange(*t_span, self.dt)):
            flow = self.step(iter)
            for cb in callbacks:
                cb(iter, t, flow)

        for cb in callbacks:
            cb.close()

        return flow

    def step(self, iter, control=None, mode="direct"):
        pass

    def enlist_controls(self, control):
        # Check if scalar
        if ufl.checks.is_ufl_scalar(control):
            control = [control]
        elif isinstance(control, (list, tuple, np.ndarray)):
            return control
        else:
            try:
                # TODO: better test for whether or not it is a number
                int(control)
                control = [control]  # Make into a list of one if so
                return control
            except TypeError:
                raise TypeError("Control type not recognized")


class IPCS(TransientSolver):
    def __init__(
        self, flow: FlowConfig, dt: float, eta: float = 0.0, debug=False, **kwargs
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

        self.B = self.flow.initialize_control()
        self.control = self.flow.get_control()

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

    def update_controls(self, control):
        """Add a damping factor to the controller response

        If actual control is u and input is v, effectively
            du/dt = (1/tau)*(v - u)
        """
        logging.log(logging.DEBUG, type(control))
        logging.log(logging.DEBUG, control)
        assert len(self.control) == len(control)
        # for i, (u, v) in enumerate(zip(self.control, control)):
        #     logging.log(logging.DEBUG, f"Input control: {v} / current value: {u}")
        #     self.control[i] += (self.dt/self.flow.TAU)*(v - u)
        for (u, v) in zip(self.control, control):
            logging.log(logging.DEBUG, f"Input control: {v} / current value: {u}")
            u += (self.dt / self.flow.TAU) * (v - u)  # F
        #     logging.log(logging.DEBUG, f"New value: {u}")
        #     # u = fd.interpolate(u + (self.dt/self.flow.TAU)*(v - u), u.function_space())  # F

        #     # u.assign(u + (self.dt/self.flow.TAU)*(v - u), annotate=True)  # RL
        #     # u.assign(v)  # RL
        #     # u = u + (self.dt / self.flow.TAU) * (v - u)  #

    def step(self, iter, control=None):
        # Update perturbations (if applicable)
        self.eta.assign(self.noise[self.noise_idx])

        # Step 1: Tentative velocity step
        logging.log(logging.DEBUG, f"iter: {iter}, solving velocity predictor")
        self.predictor.solve()
        if control is not None:
            control = self.enlist_controls(control)
            self.update_controls(control)
            for (B, ctrl) in zip(self.B, self.control):
                Bu, _ = B.split()
                self.u += Bu * ctrl

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

        from .utils import get_array, linalg, set_from_array

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
            self.B = flow.initialize_control()

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


class LinearizedIPCS(IPCS):
    def __init__(
        self, flow: FlowConfig, dt: float, base_flow: fd.Function, debug=False, **kwargs
    ):
        self.qB = base_flow
        super().__init__(flow, dt, debug=debug, **kwargs)

    def initialize_operators(self):
        from .utils import linalg

        flow = self.flow
        k = fd.Constant(self.dt)

        uB = self.qB.split()[0].copy(deepcopy=True)

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
            - inner(self.f, v) * dx
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

    def get_operators(self):
        """
        Return a LinearOperator that can act on numpy arrays (pulled from utils.get_array)

        Acts as linearized Navier-Stokes in discrete time... note that this is probably not
        the best way to actually do controller design. See Barbagallo (2009) for details
        """
        from scipy.sparse.linalg import LinearOperator

        from .utils import get_array, set_from_array

        assert fd.COMM_WORLD.size == 1, "Must run in serial for scipy operators"

        flow = self.flow
        self.B = flow.initialize_control()

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

    def step(self, iter, control=None, mode="direct"):
        if mode == "direct":
            return super().step(iter, control=control)
        elif mode == "adjoint":
            logging.log(
                logging.WARNING,
                "Adjoint time-stepping has not yet been implemented correctly",
            )

            assert hasattr(self, "adj_predictor")
            # Step 1: Tentative velocity step
            logging.log(
                logging.DEBUG, f"iter: {iter}, solving adjoint velocity predictor"
            )
            self.adj_predictor.solve()
            if control is not None:
                logging.log(logging.WARNING, "Ignoring control applied to adjoint step")

            # Update previous solution
            self.u_n.assign(self.u)
            self.p_n.assign(self.p)
            return self.flow


class IPCS_diff(TransientSolver):
    def __init__(
        self, flow: FlowConfig, dt: float, callbacks: Optional[Iterable[Callable]] = []
    ):
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

        flow.init_bcs(mixed=False)
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
        U = 0.5 * (self.u_n + u)  # Average for semi-implicit
        u_t = (u - self.u_n) / k  # Time derivative

        # Velocity predictor
        F1 = (
            dot(u_t, v) * dx
            + dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx
            + inner(flow.sigma(U, self.p_n), flow.epsilon(v)) * dx
            + dot(self.p_n * flow.n, v) * ds
            - dot(flow.nu * nabla_grad(U) * flow.n, v) * ds
        )
        self.a1 = fd.lhs(F1)
        self.L1 = fd.rhs(F1)

        # Poisson equation
        a2 = dot(nabla_grad(p), nabla_grad(s)) * dx
        self.L2 = (
            dot(nabla_grad(self.p_n), nabla_grad(s)) * dx
            - (1 / k) * div(self.u) * s * dx
        )

        # Projection step (pressure correction)
        a3 = dot(u, v) * dx
        self.L3 = dot(self.u, v) * dx - k * dot(nabla_grad(self.p - self.p_n), v) * dx

        # Assemble matrices
        self.A1 = fd.assemble(self.a1, bcs=self.bcu)
        self.A2 = fd.assemble(a2, bcs=self.bcp)
        self.A3 = fd.assemble(a3)

        self.B = flow.initialize_control()

    def step(self, iter, control=None):
        # Step 1: Tentative velocity step
        self.bcu = self.flow.collect_bcu()
        self.A1 = fd.assemble(self.a1, bcs=self.bcu)
        b1 = fd.assemble(self.L1, bcs=self.bcu)
        fd.solve(
            self.A1,
            self.u.vector(),
            b1,
            solver_parameters={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )
        if control is not None:
            control = self.enlist_controls(control)
            for (B, ctrl) in zip(self.B, control):
                Bu, _ = B.split()
                self.u += ctrl * Bu

        # Step 2: Pressure correction step
        b2 = fd.assemble(self.L2, bcs=self.bcp)
        fd.solve(
            self.A2,
            self.p.vector(),
            b2,
            solver_parameters={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )

        # Step 3: Velocity correction step
        b3 = fd.assemble(self.L3)
        fd.solve(
            self.A3,
            self.u.vector(),
            b3,
            solver_parameters={"ksp_type": "cg", "pc_type": "sor"},
        )

        # Update previous solution
        self.u_n.assign(self.u)
        self.p_n.assign(self.p)

        self.t += self.dt

        return self.flow


METHODS = {"IPCS": IPCS, "IPCS_diff": IPCS_diff}


def integrate(flow, t_span, dt, method="IPCS", callbacks=[], **options):
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")

    method = METHODS[method]
    solver = method(flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks)
