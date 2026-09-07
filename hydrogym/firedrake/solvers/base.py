import firedrake as fd
from firedrake import logging
from ufl import as_ufl, div, dot, ds, dx, inner, lhs, nabla_grad, rhs

from hydrogym.core import TransientSolver
from hydrogym.firedrake import FlowConfig
from hydrogym.firedrake.solvers.stabilization import ns_stabilization

__all__ = ["NewtonSolver"]


class NewtonSolver:
    """Newton solver for the steady-state Navier-Stokes equations.

    Assembles the nonlinear steady residual of ``flow.steady_form`` (with
    optional SUPG/GLS-type stabilization) and solves the resulting
    nonlinear variational problem with Firedrake's Newton solver.
    """

    def __init__(
        self,
        flow: FlowConfig,
        stabilization: str = "none",
        solver_parameters: dict = {},
    ):
        """Configure the steady solver.

        Args:
            flow (FlowConfig): Flow configuration defining the mesh,
                function spaces, and boundary conditions.
            stabilization (str, optional): Stabilization method, one of
                the keys of ``ns_stabilization``. Default "none".
            solver_parameters (dict, optional): Parameters passed through
                to ``fd.NonlinearVariationalSolver``.

        Raises:
            ValueError: If ``stabilization`` is not a recognized type.
        """
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
        """Assemble the nonlinear steady-state Navier-Stokes variational form.

        Builds ``F(u, p; v, s)`` with the sign convention used for
        steady solves (advection and stress terms positive), plus any
        stabilization terms. This differs from ``FlowConfig.residual``,
        whose signs are written for the transient problem.

        Args:
            q (fd.Function): Mixed trial state (u, p).
            q_test (optional): Pair of test functions (v, s); defaults to
                the test functions of the flow's mixed space.

        Returns:
            ufl.Form: The nonlinear residual form.
        """
        (u, p) = q
        if q_test is None:
            (v, s) = fd.TestFunctions(self.flow.mixed_space)
        else:
            (v, s) = q_test

        # BUGFIX: flow.residual() was introduced in commit b6ac668 (March 2024) with
        # negative signs designed for transient solvers (du/dt = residual). This broke
        # GLS stabilization which was added in commit 044d2ac (Feb 2024) with positive signs.
        # Restoring the original formulation that works with GLS/SUPG stabilization.
        F = (
            inner(dot(u, nabla_grad(u)), v) * dx
            + inner(self.flow.sigma(u, p), self.flow.epsilon(v)) * dx
            + inner(div(u), s) * dx
        )

        stab = self.stabilization_type(
            self.flow,
            q_trial=(u, p),
            q_test=(v, s),
            wind=u,
        )
        F = stab.stabilize(F)

        return F


class NavierStokesTransientSolver(TransientSolver):
    """Base class for transient Navier-Stokes solvers.

    Extends ``TransientSolver`` with a hook-based reset: subclasses
    allocate their fields and variational forms in
    ``initialize_functions`` and ``initialize_operators``, both of which
    are invoked on construction and on every ``reset``.
    """

    def __init__(self, flow: FlowConfig, dt: float = None, debug: bool = False):
        """Initialize the transient solver.

        Args:
            flow (FlowConfig): Flow configuration to advance in time.
            dt (float, optional): Time step. Defaults to ``TransientSolver``'s
                handling (typically the flow's ``DEFAULT_DT``).
            debug (bool, optional): Whether to enable debug output.
                Default False.

        Note:
            This class previously accepted ``eta``/``max_noise_iter``/
            ``noise_cutoff`` for a random white-noise body forcing, but the
            forcing was removed upstream (a067781, 2024-03) and those
            kwargs were silently ignored ever since; they were removed in
            this repo's audit Task 2.3.
        """
        # NOTE: this class previously accepted eta/max_noise_iter/noise_cutoff
        # for a random white-noise body forcing, but the forcing itself was
        # removed upstream in a067781 ("Clean up old forcing code", 2024-03)
        # and the kwargs were silently ignored ever since -- they were removed
        # in this repo's audit Task 2.3 for exactly that reason.
        super().__init__(flow, dt)
        self.debug = debug
        self.reset()

    def reset(self):
        """Reset the solver to its initial condition and rebuild state.

        Calls the parent reset, then re-runs ``initialize_functions`` and
        ``initialize_operators``.
        """
        super().reset()

        self.initialize_functions()

        self.initialize_operators()

    def initialize_functions(self):
        """Allocate solver-specific state fields.

        No-op in the base class; subclasses should override this to create
        the functions they need.
        """
        pass

    def initialize_operators(self):
        """Allocate solver-specific variational forms and operators.

        No-op in the base class; subclasses should override this to set up
        their forms/operators.
        """
        pass
