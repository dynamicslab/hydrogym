import firedrake as fd
from ufl import div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from ..flow import FlowConfig
from .base import NavierStokesTransientSolver

_alpha_BDF = [1.0, 3.0 / 2.0, 11.0 / 6.0]
_beta_BDF = [
    [1.0],
    [2.0, -1.0 / 2.0],
    [3.0, -3.0 / 2.0, 1.0 / 3.0],
]
_beta_EXT = [
    [1.0],
    [2.0, -1.0],
    [3.0, -3.0, 1.0],
]


class SemiImplicitBDF(NavierStokesTransientSolver):
    def __init__(
        self,
        flow: FlowConfig,
        dt: float = None,
        order: int = 3,
        **kwargs,
    ):
        self.k = order  # Order of the BDF/EXT scheme

        super().__init__(flow, dt, **kwargs)
        self.reset()

    def initialize_functions(self):
        flow = self.flow

        # Trial/test functions for linear sub-problem
        W = flow.mixed_space
        u, p = fd.TrialFunctions(W)
        w, s = fd.TestFunctions(W)

        self.q_trial = (u, p)
        self.q_test = (w, s)

        # Previous solutions for BDF and extrapolation
        q_prev = [fd.Function(W) for _ in range(self.k)]

        self.u_prev = [q.subfunctions[0] for q in q_prev]

        # Assign the current solution to all `u_prev`
        for u in self.u_prev:
            u.assign(flow.q.subfunctions[0])

    def _make_order_k_solver(self, k):
        # Setup functions and spaces
        flow = self.flow
        h = fd.Constant(self.dt)

        (u, p) = self.q_trial
        (w, s) = self.q_test

        # Combinations of functions for form construction
        k_idx = k - 1
        u_EXT = sum(beta * u_n for beta, u_n in zip(_beta_EXT[k_idx], self.u_prev))
        u_BDF = sum(beta * u_n for beta, u_n in zip(_beta_BDF[k_idx], self.u_prev))
        alpha_k = _alpha_BDF[k_idx]
        u_t = (alpha_k * u - u_BDF) / h  # BDF estimate of time derivative

        # Semi-implicit weak form
        weak_form = (
            dot(u_t, w) * dx
            + dot(dot(u_EXT, nabla_grad(u)), w) * dx
            + inner(flow.sigma(u, p), flow.epsilon(w)) * dx
            + dot(p * flow.n, w) * ds
            - dot(flow.nu * nabla_grad(u) * flow.n, w) * ds
            - dot(self.eta * self.f, w) * dx
            + dot(div(u), s) * dx
        )

        # Construct variational problem and PETSc solver
        q = self.flow.q
        a = lhs(weak_form)
        L = rhs(weak_form)
        bcs = self.flow.collect_bcs()
        bdf_prob = fd.LinearVariationalProblem(a, L, q, bcs=bcs)

        # Schur complement preconditioner. See:
        # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
        solver_parameters = {
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-6,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_schur_precondition": "selfp",
            #
            # Default preconditioner for inv(A)
            #   (ilu in serial, bjacobi in parallel)
            "fieldsplit_0_ksp_type": "preonly",
            #
            # Single multigrid cycle preconditioner for inv(S)
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "hypre",
        }

        petsc_solver = fd.LinearVariationalSolver(
            bdf_prob, solver_parameters=solver_parameters
        )
        return petsc_solver

    def initialize_operators(self):
        self.flow.init_bcs(mixed=True)
        self.petsc_solver = self._make_order_k_solver(self.k)

        # Start-up solvers for BDF/EXT schemes
        self.startup_solvers = []
        if self.k > 1:
            for i in range(self.k - 1):
                self.startup_solvers.append(self._make_order_k_solver(i + 1))

    def step(self, iter, control=None):
        # Update perturbations (if applicable)
        self.eta.assign(self.noise[self.noise_idx])

        # Pass input through actuator "dynamics" or filter
        if control is not None:
            bc_scale = self.flow.update_actuators(control, self.dt)
            self.flow.set_control(bc_scale)

        # Solve the linear problem
        if iter > self.k - 1:
            self.petsc_solver.solve()
        else:
            self.startup_solvers[iter - 1].solve()

        # Store the historical solutions for BDF/EXT estimates
        for i in range(self.k - 1):
            self.u_prev[-(i + 1)].assign(self.u_prev[-(i + 2)])

        self.u_prev[0].assign(self.flow.q.subfunctions[0])

        self.t += self.dt
        self.noise_idx += 1
        assert self.noise_idx < len(self.noise), "Not enough noise samples generated"

        return self.flow
