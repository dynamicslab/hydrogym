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

  def __init__(
      self,
      flow: FlowConfig,
      dt: float,
      order: int = 3,
      stabilization: str = "default",
      rtol=1e-6,
      **kwargs,
  ):
    self.k = order  # Order of the BDF/EXT scheme
    self.ksp_rtol = rtol  # Krylov solver tolerance

    if stabilization == "default":
      stabilization = flow.DEFAULT_STABILIZATION

    self.stabilization = stabilization
    if stabilization not in ns_stabilization:
      raise ValueError(f"Stabilization type {stabilization} not recognized. "
                       f"Available options: {ns_stabilization.keys()}")
    self.StabilizationType = ns_stabilization[stabilization]

    super().__init__(flow, dt, **kwargs)
    self.reset()

  def initialize_functions(self):
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

    petsc_solver = fd.LinearVariationalSolver(
        bdf_prob, solver_parameters=solver_parameters)
    return petsc_solver

  def _stabilize_weak_form(self, weak_form, u_t, wind, f=None):
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
        dot(u_t, v) * dx + dot(dot(w, nabla_grad(u)), v) * dx +
        inner(flow.sigma(u, p), flow.epsilon(v)) * dx + dot(div(u), s) * dx -
        dot(self.f, v) * dx)

    weak_form = self._stabilize_weak_form(weak_form, u_t, wind=w, f=self.f)
    return self._make_petsc_solver(weak_form)

  def initialize_operators(self):
    self.flow.init_bcs()
    self.petsc_solver = self._make_order_k_solver(self.k)

    # Start-up solvers for BDF/EXT schemes
    self.startup_solvers = []
    if self.k > 1:
      for i in range(self.k - 1):
        self.startup_solvers.append(self._make_order_k_solver(i + 1))

  def step(self, iter, control=None):
    # Update the time of the flow
    # TODO: Test with actuation
    bc_scale = self.flow.advance_time(self.dt, control)
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

    return self.flow


class LinearizedBDF(SemiImplicitBDF):

  def __init__(self, *args, qB: fd.Function, **kwargs):
    self.qB = qB
    stabilization = kwargs.pop("stabilization", "none").split("_")
    if stabilization[0] != "linearized":
      stabilization = ["linearized", stabilization[0]]
    stabilization = "_".join(stabilization)
    super().__init__(*args, stabilization=stabilization, **kwargs)

  def _make_order_k_solver(self, k):
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
        dot(u_t, v) * dx + dot(dot(uB, nabla_grad(u)), v) * dx +
        dot(dot(u, nabla_grad(uB)), v) * dx +
        inner(sigma(u, p), epsilon(v)) * dx + dot(div(u), s) * dx -
        dot(self.f, v) * dx
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
