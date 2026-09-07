import firedrake as fd
from ufl import div, dot, dx, inner, lhs, nabla_grad, rhs

from ..flow import FlowConfig
from .base import NavierStokesTransientSolver
from .stabilization import ns_stabilization

__all__ = [
    "SemiImplicitBDF",
    "LinearizedBDF",
]

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
    """Semi-implicit BDF transient solver for the incompressible Navier-Stokes equations.

    Uses a backward-differentiation formula of order ``k`` for the time
    derivative (treated implicitly, together with the viscous and pressure
    terms) and a k-th-order extrapolation of the velocity for the convective
    term, which is thereby treated explicitly. The result is a sequence of
    linear variational problems, one per BDF order, solved each timestep with
    Firedrake's ``LinearVariationalSolver``.

    For ``k > 1`` the first ``k - 1`` timesteps are taken with lower-order
    startup solvers (order 1 .. k-1, each with a matching extrapolation),
    since the full-order scheme needs ``k`` previous solutions.

    Attributes:
        k: Order of the BDF/extrapolation scheme (1, 2, or 3).
        ksp_rtol: Relative tolerance for the Krylov solver.
        custom_solver_parameters: User-supplied PETSc solver parameters,
            overriding the built-in defaults (see ``_make_petsc_solver``).
        stabilization: Name of the stabilization scheme (a key of
            :data:`~hydrogym.firedrake.solvers.stabilization.ns_stabilization`);
            ``"default"`` resolves to ``flow.DEFAULT_STABILIZATION``.
        u_prev: List of the ``k`` most recent velocity solutions (oldest
            first) used to build the BDF and extrapolation combinations.
    """

    def __init__(
        self,
        flow: FlowConfig,
        dt: float = None,
        order: int = 3,
        stabilization: str = "default",
        rtol=1e-6,
        solver_parameters: dict = None,
        **kwargs,
    ):
        """Initialize the solver and build the BDF/startup operators.

        Args:
            flow: Flow configuration (mesh, mixed space, BCs, forcing).
            dt: Timestep size. Optional; defaults to the flow's
                ``DEFAULT_DT`` if not given (see
                ``NavierStokesTransientSolver``/``TransientSolver``, whose
                contract this class previously broke by requiring ``dt``
                positionally).
            order: Order of the BDF/extrapolation scheme (1-3).
            stabilization: Stabilization type; ``"default"`` resolves to the
                flow's ``DEFAULT_STABILIZATION``. See ``ns_stabilization`` for
                the available keys (e.g. "none", "supg", "gls", and their
                "linearized_" variants).
            rtol: Relative tolerance for the Krylov (KSP) solver.
            solver_parameters: Optional PETSc solver parameters replacing the
                built-in defaults chosen in ``_make_petsc_solver``.
            **kwargs: Forwarded to
                :class:`~hydrogym.firedrake.solvers.base.NavierStokesTransientSolver`.

        Raises:
            ValueError: If ``stabilization`` is not a recognized type.
        """
        self.k = order  # Order of the BDF/EXT scheme
        self.ksp_rtol = rtol  # Krylov solver tolerance
        self.custom_solver_parameters = solver_parameters  # Custom solver parameters

        if stabilization == "default":
            stabilization = flow.DEFAULT_STABILIZATION

        self.stabilization = stabilization
        if stabilization not in ns_stabilization:
            raise ValueError(
                f"Stabilization type {stabilization} not recognized. Available options: {ns_stabilization.keys()}"
            )
        self.StabilizationType = ns_stabilization[stabilization]

        super().__init__(flow, dt, **kwargs)
        self.reset()

    def initialize_functions(self):
        """Allocate trial/test functions and the BDF history of previous solutions.

        Sets ``self.q_trial``/``self.q_test`` (velocity-pressure pairs on the
        mixed space), aliases the body force ``self.f`` from the flow
        configuration, and creates ``k`` copies of the current velocity in
        ``self.u_prev`` so the first (startup) steps have valid history.
        """
        flow = self.flow
        self.f = flow.body_force

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

    def _make_petsc_solver(self, weak_form):
        """Wrap a weak form in a Firedrake linear variational problem and PETSc solver.

        The problem is defined on the flow's current state ``self.flow.q``
        with the flow's boundary conditions applied.

        Args:
            weak_form: UFL form (zero if written ``lhs - rhs = 0``) for the
                order-``k`` BDF step.

        Returns:
            fd.LinearVariationalSolver: Solver with either the user-supplied
            parameters or built-in defaults: a Schur-complement fieldsplit
            preconditioner for the unstabilized saddle-point system, monolithic
            hypre AMG for SUPG-type stabilization, and a direct LU/MUMPS solve
            for GLS-type stabilization (which is incompatible with the Schur
            complement approach).
        """
        # Construct variational problem and PETSc solver
        q = self.flow.q
        a = lhs(weak_form)
        L = rhs(weak_form)
        bcs = self.flow.collect_bcs()
        bdf_prob = fd.LinearVariationalProblem(a, L, q, bcs=bcs)

        # Use custom solver parameters if provided, otherwise use defaults
        if self.custom_solver_parameters is not None:
            solver_parameters = self.custom_solver_parameters
        else:
            # Schur complement preconditioner. See:
            # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
            solver_parameters = {
                "ksp_type": "fgmres",
                "ksp_rtol": self.ksp_rtol,
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

            # Stabilization-specific solver parameters
            # Both SUPG and GLS are incompatible with Schur complement preconditioners
            # due to the additional coupling terms in the stabilization.
            if self.stabilization in ["supg", "linearized_supg"]:
                # SUPG: Use monolithic AMG (works, much faster than direct solver)
                solver_parameters = {
                    "ksp_type": "gmres",
                    "ksp_rtol": self.ksp_rtol,
                    "ksp_max_it": 200,
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                }
            elif self.stabilization in ["gls", "linearized_gls"]:
                # GLS: Use direct solver (monolithic AMG fails for GLS)
                from firedrake import logging

                logging.warning(
                    "GLS stabilization detected: using direct solver (LU/MUMPS). "
                    "This is slower than SUPG with monolithic AMG."
                )
                solver_parameters = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }

        petsc_solver = fd.LinearVariationalSolver(bdf_prob, solver_parameters=solver_parameters)
        return petsc_solver

    def _stabilize_weak_form(self, weak_form, u_t, wind, f=None):
        """Append the configured stabilization terms (SUPG, GLS, etc.) to a weak form.

        Args:
            weak_form: The unstabilized UFL weak form.
            u_t: UFL expression for the BDF estimate of the time derivative.
            wind: Velocity field used as the "wind" in the convective and
                stabilization terms (the extrapolated velocity, or the base
                flow for the linearized solvers).
            f: Body forcing entering the residual-based stabilization terms.

        Returns:
            The weak form with the stabilization term added (unchanged for
            the "none" stabilization types).
        """
        # Stabilization (SUPG, GLS, etc.)
        stab = self.StabilizationType(
            self.flow,
            self.q_trial,
            self.q_test,
            wind=wind,
            dt=self.dt,
            u_t=u_t,
            f=f,
        )
        return stab.stabilize(weak_form)

    def _make_order_k_solver(self, k):
        """Build the (order-``k`` BDF + order-``k`` extrapolation) solver.

        Assembles the semi-implicit weak form

            ``(alpha_k u - sum beta_BDF u_n) / dt + w . grad(u) + sigma(u,p) : eps(v)
            + div(u) s - f . v``

        where ``w`` is the order-``k`` extrapolation of the previous
        velocities (the explicit convective velocity) and ``alpha_k``/``beta_BDF``
        are the BDF-``k`` coefficients, then adds stabilization and wraps the
        form in a PETSc solver.

        Args:
            k: BDF order to build (``1..self.k``); orders below ``self.k``
                are the startup schemes used for the first timesteps.

        Returns:
            fd.LinearVariationalSolver: Solver for the order-``k`` step.
        """
        # Setup functions and spaces
        flow = self.flow
        h = fd.Constant(self.dt)

        (u, p) = self.q_trial
        (v, s) = self.q_test

        # Combinations of functions for form construction
        k_idx = k - 1
        # The "wind" w is the extrapolation estimate of u[n+1]
        w = sum(beta * u_n for beta, u_n in zip(_beta_EXT[k_idx], self.u_prev))
        u_BDF = sum(beta * u_n for beta, u_n in zip(_beta_BDF[k_idx], self.u_prev))
        alpha_k = _alpha_BDF[k_idx]
        u_t = (alpha_k * u - u_BDF) / h  # BDF estimate of time derivative

        # Semi-implicit weak form
        weak_form = (
            dot(u_t, v) * dx
            + dot(dot(w, nabla_grad(u)), v) * dx
            + inner(flow.sigma(u, p), flow.epsilon(v)) * dx
            + dot(div(u), s) * dx
            - dot(self.f, v) * dx
        )

        weak_form = self._stabilize_weak_form(weak_form, u_t, wind=w, f=self.f)
        return self._make_petsc_solver(weak_form)

    def initialize_operators(self):
        """Build the main order-``k`` solver and the lower-order startup solvers.

        Initializes the flow's boundary conditions first, then constructs the
        full-order BDF solver and, if ``k > 1``, one solver per order
        ``1 .. k-1`` for the startup timesteps (``self.startup_solvers``).
        """
        self.flow.init_bcs()
        self.petsc_solver = self._make_order_k_solver(self.k)

        # Start-up solvers for BDF/EXT schemes
        self.startup_solvers = []
        if self.k > 1:
            for i in range(self.k - 1):
                self.startup_solvers.append(self._make_order_k_solver(i + 1))

    def step(self, iter, control=None):
        """Advance the flow by one timestep.

        The flow's time is advanced (which also applies any actuation
        scaling), the appropriate linear problem is solved — the full-order
        BDF solver once ``iter`` exceeds ``k - 1``, otherwise the matching
        lower-order startup solver — and the velocity history is shifted so
        the newest solution enters ``u_prev[0]``.

        Args:
            iter: Timestep index within the solve (0-based); the first
                ``k - 1`` iterations use the startup solvers.
            control: Optional actuation value(s) forwarded to
                ``flow.advance_time`` to scale the actuation BCs.

        Returns:
            FlowConfig: The updated flow configuration.
        """
        # Update the time of the flow
        # TODO: Test with actuation
        bc_scale = self.flow.advance_time(self.dt, control)
        self.flow.set_control(bc_scale)

        # Solve the linear problem
        if (self.k == 1) or (iter > self.k - 1):
            self.petsc_solver.solve()
        else:
            self.startup_solvers[iter - 1].solve()

        # Store the historical solutions for BDF/EXT estimates
        for i in range(self.k - 1):
            self.u_prev[-(i + 1)].assign(self.u_prev[-(i + 2)])

        self.u_prev[0].assign(self.flow.q.subfunctions[0])

        return self.flow


class LinearizedBDF(SemiImplicitBDF):
    """Semi-implicit BDF solver for the Navier-Stokes equations linearized about a base flow.

    Instead of the full convective term, solves the linearized form

        ``du/dt + uB . grad(u) + u . grad(uB) - div(sigma(u,p)) = f``

    around the base flow ``qB`` supplied at construction, with the flow's
    boundary conditions linearized (``flow.linearize_bcs()``) and the
    base-flow velocity ``uB`` acting as the "wind" in the stabilization
    terms. Intended for adjoint/transient-growth-type analyses about a known
    (typically steady) state.

    Attributes:
        qB: Base flow (mixed velocity-pressure ``fd.Function``) to
            linearize about.
    """

    def __init__(self, *args, qB: fd.Function, **kwargs):
        """Initialize the linearized solver.

        Args:
            *args: Positional arguments forwarded to
                :class:`SemiImplicitBDF` (``flow``, ``dt``, ...).
            qB: Base flow (mixed velocity-pressure function) to linearize
                the equations about.
            **kwargs: Keyword arguments forwarded to :class:`SemiImplicitBDF`.
                ``stabilization`` defaults to ``"none"`` (resolved to
                ``"linearized_none"``) rather than the unlinearized default;
                a plain name (e.g. ``"supg"``) is automatically prefixed with
                ``"linearized_"``.
        """
        self.qB = qB
        stabilization = kwargs.pop("stabilization", "none").split("_")
        if stabilization[0] != "linearized":
            stabilization = ["linearized", stabilization[0]]
        stabilization = "_".join(stabilization)
        super().__init__(*args, stabilization=stabilization, **kwargs)

    def _make_order_k_solver(self, k):
        """Build the order-``k`` solver for the base-flow-linearized weak form.

        Like :meth:`SemiImplicitBDF._make_order_k_solver`, but the convective
        terms are the linearized pair ``uB . grad(u) + u . grad(uB)``, the
        flow's boundary conditions are linearized (``flow.linearize_bcs()``),
        and the base-flow velocity ``uB`` is used as the wind in the
        stabilization terms. The base-flow forcing contributions on the RHS
        vanish for a steady base flow and are folded into ``self.f``.

        Args:
            k: BDF order to build (``1..self.k``).

        Returns:
            fd.LinearVariationalSolver: Solver for the order-``k`` step.
        """
        # Setup functions and spaces
        flow = self.flow
        sigma, epsilon = flow.sigma, flow.epsilon

        h = fd.Constant(self.dt)

        flow.linearize_bcs()

        (uB, pB) = self.qB.subfunctions
        (u, p) = self.q_trial
        (v, s) = self.q_test

        # Combinations of functions for form construction
        k_idx = k - 1
        u_BDF = sum(beta * u_n for beta, u_n in zip(_beta_BDF[k_idx], self.u_prev))
        alpha_k = _alpha_BDF[k_idx]
        u_t = (alpha_k * u - u_BDF) / h  # BDF estimate of time derivative

        # Semi-implicit weak form
        # Note that the base flow terms are added to the RHS in `self.f`
        weak_form = (
            dot(u_t, v) * dx
            + dot(dot(uB, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(uB)), v) * dx
            + inner(sigma(u, p), epsilon(v)) * dx
            + dot(div(u), s) * dx
            - dot(self.f, v) * dx
            # # Base flow forcing (will be zero if base flow is steady solution)
            # + inner(dot(uB, nabla_grad(uB)), v) * dx
            # + inner(sigma(uB, pB), epsilon(v)) * dx
        )

        # RHS forcing term for stabilization (also zero if d(qB)/dt = 0)
        f_stab = self.f  # - dot(uB, nabla_grad(uB)) + div(sigma(uB, pB))

        weak_form = self._stabilize_weak_form(weak_form, u_t, wind=uB, f=f_stab)
        return self._make_petsc_solver(weak_form)

    # def initialize_functions(self):
    #     # Add to the RHS forcing with base flow terms
    #     super().initialize_functions()

    #     sigma, epsilon = self.flow.sigma, self.flow.epsilon
    #     (uB, pB) = self.qB.subfunctions
    #     (v, _) = self.q_test

    #     self.f += (
    #         inner(sigma(uB, pB), epsilon(v)) * dx
    #         - inner(dot(uB, nabla_grad(uB)), v) * dx
    #     )
