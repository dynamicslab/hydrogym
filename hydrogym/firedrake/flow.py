from functools import partial
from typing import Callable, Iterable, NamedTuple

import firedrake as fd
import numpy as np
import pyadjoint
import ufl
from firedrake import dx, logging
from firedrake.__future__ import interpolate
from mpi4py import MPI
from numpy.typing import ArrayLike
from ufl import curl, div, dot, inner, nabla_grad, sqrt, sym

from ..core import ActuatorBase, PDEBase
from .actuator import DampedActuator
from .utils.linalg import DirectOperator, InverseOperator


class ScaledDirichletBC(fd.DirichletBC):

  def __init__(self, V, g, sub_domain, method=None):
    self.unscaled_function_arg = g
    self._scale = fd.Constant(1.0)
    super().__init__(V, self._scale * g, sub_domain, method)

  def set_scale(self, c):
    self._scale.assign(c)


class ObservationFunction(NamedTuple):
  func: Callable
  num_outputs: int

  def __call__(self, q: fd.Function) -> np.ndarray:
    return self.func(q)


class FlowConfig(PDEBase):
  DEFAULT_REYNOLDS = 1
  DEFAULT_VELOCITY_ORDER = 2  # Taylor-Hood elements
  DEFAULT_STABILIZATION = "none"
  MESH_DIR = ""

  FUNCTIONS = ("q",)  # tuple of functions necessary for the flow

  def __init__(self, velocity_order=None, **config):
    self.Re = fd.Constant(ufl.real(config.get("Re", self.DEFAULT_REYNOLDS)))

    if velocity_order is None:
      velocity_order = self.DEFAULT_VELOCITY_ORDER
    self.velocity_order = velocity_order

    obs_type = config.pop("observation_type", None)
    if obs_type == "velocity_probes":
      probes = config.pop("probes", self.DEFAULT_VEL_PROBES)
    elif obs_type == "pressure_probes":
      probes = config.pop("probes", self.DEFAULT_PRES_PROBES)
    elif obs_type == "vorticity_probes":
      probes = config.pop("probes", self.DEFAULT_VORT_PROBES)
    else:
      probes = config.pop("probes", None)

    # print("Probes are", probes, self.DEFAULT_PRES_PROBES, flush=True)
    if probes is None:
      probes = []

    probe_obs_types = {
        "velocity_probes":
            ObservationFunction(
                partial(self.velocity_probe, probes),
                num_outputs=2 * len(probes)),
        "pressure_probes":
            ObservationFunction(
                partial(self.pressure_probe, probes), num_outputs=len(probes)),
        "vorticity_probes":
            ObservationFunction(
                partial(self.vorticity_probe, probes), num_outputs=len(probes)),
    }

    self.obs_fun = self.configure_observations(
        obs_type=obs_type,
        probe_obs_types=probe_obs_types,
    )

    super().__init__(**config)

  def load_mesh(self, name: str) -> ufl.Mesh:
    return fd.Mesh(f"{self.MESH_DIR}/{name}.msh", name="mesh")

  def save_checkpoint(self, filename: str, write_mesh=True, idx=None):
    with fd.CheckpointFile(filename, "w") as chk:
      if write_mesh:
        chk.save_mesh(self.mesh)  # optional
      for f_name in self.FUNCTIONS:
        chk.save_function(getattr(self, f_name), idx=idx)

      act_state = np.array([act.state for act in self.actuators])
      chk.set_attr("/", "act_state", act_state)

  def load_checkpoint(self, filename: str, idx=None, read_mesh=True):
    with fd.CheckpointFile(filename, "r") as chk:
      if read_mesh:
        self.mesh = chk.load_mesh("mesh")
        self.initialize_state()
      else:
        assert hasattr(self, "mesh")
      for f_name in self.FUNCTIONS:
        try:
          f_load = chk.load_function(self.mesh, f_name, idx=idx)
          f_self = getattr(self, f_name)

          # If the checkpoint saved on a different function space,
          # approximate the same field on the current function space
          # by projecting the checkpoint field onto the current space
          V_chk = f_load.function_space()
          V_self = f_self.function_space()
          if V_chk.ufl_element() != V_self.ufl_element():
            f_load = fd.project(f_load, V_self)

          f_self.assign(f_load)
        except RuntimeError:
          logging.log(
              logging.WARN,
              f"Function {f_name} not found in checkpoint, defaulting to zero.",
          )

      if chk.has_attr("/", "act_state"):
        act_state = chk.get_attr("/", "act_state")
        for i in range(self.num_inputs):
          self.actuators[i].state = act_state[i]

    self.split_solution(
    )  # Reset functions so self.u, self.p point to the new solution

  def configure_observations(self,
                             obs_type=None,
                             probe_obs_types={}) -> ObservationFunction:
    raise NotImplementedError

  def get_observations(self) -> np.ndarray:
    return self.obs_fun(self.q)

  @property
  def num_outputs(self) -> int:
    # This may be lift/drag, a stress "sensor", or a set of probe locations
    return self.obs_fun.num_outputs

  def initialize_state(self):
    # Set up UFL objects referring to the mesh
    self.n = fd.FacetNormal(self.mesh)
    self.x, self.y = fd.SpatialCoordinate(self.mesh)

    # Set up Taylor-Hood elements
    self.velocity_space = fd.VectorFunctionSpace(self.mesh, "CG",
                                                 self.velocity_order)
    self.pressure_space = fd.FunctionSpace(self.mesh, "CG", 1)
    self.mixed_space = fd.MixedFunctionSpace(
        [self.velocity_space, self.pressure_space])
    for f_name in self.FUNCTIONS:
      setattr(self, f_name, fd.Function(self.mixed_space, name=f_name))

    self.split_solution()  # Break out and rename main solution

    self._vorticity = fd.Function(self.pressure_space, name="vort")

  def set_state(self, q: fd.Function):
    """Set the current state fields

        Args:
            q (fd.Function): State to be assigned
        """
    self.q.assign(q)

  def copy_state(self, deepcopy: bool = True) -> fd.Function:
    """Return a copy of the current state fields

        Returns:
            q (fd.Function): copy of the flow state
        """
    return self.q.copy(deepcopy=deepcopy)

  def create_actuator(self, tau=None) -> ActuatorBase:
    """Create a single actuator for this flow"""
    if tau is None:
      tau = self.TAU
    return DampedActuator(1 / tau)

  def reset_controls(self, function_spaces=None):
    """Reset the controls to a zero state

        Note that this is broken out from `reset` because
        the two are not necessarily called together (e.g.
        for linearization or deriving the control vector)

        TODO: Allow for different kinds of actuators
        """
    self.actuators = [self.create_actuator() for _ in range(self.num_inputs)]
    self.init_bcs(function_spaces=function_spaces)

  @property
  def nu(self):
    return fd.Constant(1 / ufl.real(self.Re))

  def split_solution(self):
    self.u, self.p = self.q.subfunctions
    self.u.rename("u")
    self.p.rename("p")

  def vorticity(self, u: fd.Function = None) -> fd.Function:
    """Compute the vorticity field `curl(u)` of the flow

        Args:
            u (fd.Function, optional):
                If given, compute the vorticity of this velocity
                field rather than the current state.

        Returns:
            fd.Function: vorticity field
        """
    if u is None:
      u = self.u
    self._vorticity.project(curl(u))
    return self._vorticity

  def function_spaces(self, mixed: bool = True):
    """Function spaces for velocity and pressure

        Args:
            mixed (bool, optional):
                If True (default), return subspaces of the mixed velocity/pressure
                space. Otherwise return the segregated velocity and pressure spaces.

        Returns:
            Tuple[fd.FunctionSpace, fd.FunctionSpace]: Velocity and pressure spaces
        """
    if mixed:
      V = self.mixed_space.sub(0)
      Q = self.mixed_space.sub(1)
    else:
      V = self.velocity_space
      Q = self.pressure_space
    return V, Q

  def collect_bcu(self) -> Iterable[fd.DirichletBC]:
    """List of velocity boundary conditions"""
    return []

  def collect_bcp(self) -> Iterable[fd.DirichletBC]:
    """List of pressure boundary conditions"""
    return []

  def collect_bcs(self) -> Iterable[fd.DirichletBC]:
    """List of all boundary conditions"""
    return self.collect_bcu() + self.collect_bcp()

  def epsilon(self, u) -> ufl.Form:
    """Symmetric gradient (strain) tensor"""
    return sym(nabla_grad(u))

  def sigma(self, u, p) -> ufl.Form:
    """Newtonian stress tensor"""
    return 2 * self.nu * self.epsilon(u) - p * fd.Identity(len(u))

  def residual(self, q, q_test=None):
    """Nonlinear residual for the incompressible Navier-Stokes equations.

        Returns a UFL form F(u, p, v, s) = 0, where (u, p) is the trial function
        and (v, s) is the test function.  This residual is also the right-hand side
        of the unsteady equations.

        A linearized form can be constructed by calling:
        ```
        F = flow.residual((uB, pB), (v, s))
        J = fd.derivative(F, qB, q_trial)
        ```
        """
    (u, p) = q
    if q_test is None:
      (v, s) = fd.TestFunctions(self.mixed_space)
    else:
      (v, s) = q_test

    sigma, epsilon = self.sigma, self.epsilon
    F = (-inner(dot(u, nabla_grad(u)), v) * dx -
         inner(sigma(u, p), epsilon(v)) * dx + inner(div(u), s) * dx)
    return F

  @pyadjoint.no_annotations
  def max_cfl(self, dt) -> float:
    """Estimate of maximum CFL number"""
    h = fd.CellSize(self.mesh)
    CFL = fd.assemble(
        interpolate(dt * sqrt(dot(self.u, self.u)) / h, self.pressure_space))
    return self.mesh.comm.allreduce(CFL.vector().max(), op=MPI.MAX)

  @property
  def body_force(self):
    return fd.Function(self.velocity_space).assign(fd.Constant((0.0, 0.0)))

  def linearize_bcs(self):
    """Sets the boundary conditions appropriately for linearized flow"""
    raise NotImplementedError

  def set_control(self, act: ArrayLike = None):
    """
        Directly sets the control state

        Note that for time-varying controls it will be better to adjust the controls
        in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
        to change control for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
    if act is not None:
      super().set_control(act)

      if hasattr(self, "bcu_actuation"):
        for i in range(self.num_inputs):
          u = np.clip(self.actuators[i].state, self.MAX_CONTROL_LOW,
                      self.MAX_CONTROL_UP) * self.CONTROL_SCALING
          self.bcu_actuation[i].set_scale(u)

  def inner_product(
      self,
      q1: fd.Function,
      q2: fd.Function,
      assemble=True,
      augmented=False,
  ):
    """Energy inner product for the Navier-Stokes equations.

        `augmented` is used to specify whether the function space is
        extended to represent complex numbers. In this case the inner
        product is the L2 norm of the real and imaginary parts.
        """
    if not augmented:
      (u, _) = fd.split(q1)
      (v, _) = fd.split(q2)
      M = inner(u, v) * dx

    else:
      (u_re, _, u_im, _) = fd.split(q1)
      (v_re, _, v_im, _) = fd.split(q2)
      M = (inner(u_re, v_re) + inner(u_im, v_im)) * dx

    if not assemble:
      return M
    return fd.assemble(M)

  def velocity_probe(self, probes, q: fd.Function = None) -> list[float]:
    """Probe velocity in the wake.

        Returns a list of velocities at the probe locations, ordered as
        (u1, u2, ..., uN, v1, v2, ..., vN) where N is the number of probes.
        """
    if q is None:
      q = self.q
    u = q.subfunctions[0]

    # probe_values = np.stack([u.at(probe) for probe in probes]).flatten("F")
    probe_values = np.stack(u.at(probes)).flatten("F")
    return probe_values

  def pressure_probe(self, probes, q: fd.Function = None) -> list[float]:
    """Probe pressure around the cylinder"""
    # print("pressure probing at:", probes, flush=True)
    if q is None:
      q = self.q
    p = q.subfunctions[1]
    # probe_values = np.stack([p.at(probe) for probe in probes])
    probe_values = np.stack(p.at(probes))
    return probe_values

  def vorticity_probe(self, probes, q: fd.Function = None) -> list[float]:
    """Probe vorticity in the wake."""
    if q is None:
      u = None
    else:
      u = q.subfunctions[0]
    vort = self.vorticity(u=u)
    # probe_values = np.stack([vort.at(probe) for probe in probes])
    probe_values = np.stack(vort.at(probes))
    return probe_values

  def linearize(self,
                qB=None,
                adjoint=False,
                sigma=0.0,
                inverse=False,
                solver_parameters=None):
    if sigma != 0.0 and not inverse:
      raise ValueError("Must use `inverse=True` with spectral shift.")

    if qB is None:
      qB = self.q

    if not inverse:
      return self._jacobian_operator(qB, adjoint=adjoint)

    if sigma.imag == 0.0:
      return self._real_shift_inv_operator(
          qB, sigma, adjoint=adjoint, solver_parameters=solver_parameters)

    return self._complex_shift_inv_operator(
        qB, sigma, adjoint=adjoint, solver_parameters=solver_parameters)

  # TODO: Test this. This is just here as an extension of the work done with
  # shift-inverse-type operators, but hasn't been directly tested yet. Really
  # it should be used for linearized time-stepping.
  def _jacobian_operator(self, qB, adjoint=False):
    """Construct the Jacobian operator for the Navier-Stokes equations.

        A matrix-vector product for this operator is equivalent to evaluating
        the Jacobian of the Navier-Stokes equations at a given state.
        """
    # Set up function spaces
    fn_space = self.mixed_space
    (uB, pB) = fd.split(qB)
    q_trial = fd.TrialFunction(fn_space)
    q_test = fd.TestFunction(fn_space)
    (v, s) = fd.split(q_test)

    # Collect boundary conditions
    self.linearize_bcs()
    bcs = self.collect_bcs()

    # Linear form expressing the RHS of the Navier-Stokes without time derivative
    # For a steady solution this is F(qB) = 0.
    F = self.residual((uB, pB), q_test=(v, s))

    # The Jacobian of F is the bilinear form J(qB, q_test) = dF/dq(qB) @ q_test
    J = fd.derivative(F, qB, q_trial)

    A = DirectOperator(J, bcs, fn_space)

    if adjoint:
      A = A.T

    return A

  def _real_shift_inv_operator(self,
                               qB,
                               sigma,
                               adjoint=False,
                               solver_parameters=None):
    """Construct a shift-inverse Arnoldi iterator with real (or zero) shift.

        The shift-inverse iteration solves the matrix pencil
        (J - sigma * M) @ v1 = M @ v0 for v1, where J is the Jacobian of the
        Navier-Stokes equations, and M is the mass matrix.
        """
    fn_space = self.mixed_space
    (uB, pB) = fd.split(qB)
    q_trial = fd.TrialFunction(fn_space)
    q_test = fd.TestFunction(fn_space)
    (v, s) = fd.split(q_test)

    # Collect boundary conditions
    self.linearize_bcs()
    bcs = self.collect_bcs()

    def M(q):
      """Mass matrix for the Navier-Stokes equations"""
      return self.inner_product(q, q_test, assemble=False)

    # Linear form expressing the RHS of the Navier-Stokes without time derivative
    # For a steady solution this is F(qB) = 0.
    F = self.residual((uB, pB), q_test=(v, s))

    if sigma != 0.0:
      F -= sigma * M(qB)

    # The Jacobian of F is the bilinear form J(qB, q_test) = dF/dq(qB) @ q_test
    J = fd.derivative(F, qB, q_trial)

    A = InverseOperator(J, M, bcs, fn_space, solver_parameters)

    if adjoint:
      A = A.T

    return A

  def _complex_shift_inv_operator(self,
                                  qB,
                                  sigma,
                                  adjoint=False,
                                  solver_parameters=None):
    """Construct a shift-inverse Arnoldi iterator with complex-valued shift.

        The shifted operator is `A = (J - sigma * M)`

        For sigma = (sr, si), the real and imaginary parts of A are
        A = (J - sr * M, -si * M)
        The system solve is A @ v1 = M @ v0, so for complex vectors v = (vr, vi):
        (Ar + 1j * Ai) @ (v1r + 1j * v1i) = M @ (v0r + 1j * v0i)
        Since we can't do complex analysis without re-building PETSc, instead we treat this
        as a block system:
        ```
        [Ar,   Ai]  [v1r]  = [M 0]  [v0r]
        [-Ai,  Ar]  [v1i]    [0 M]  [v0i]
        ```

        Note that this will be more expensive than real- or zero-shifted iteration,
        since there are twice as many degrees of freedom.  However, it will tend to
        converge faster for the eigenvalues of interest.

        The shift-inverse iteration solves the matrix pencil
        (J - sigma * M) @ v1 = M @ v0 for v1, where J is the Jacobian of the
        Navier-Stokes equations, and M is the mass matrix.
        """
    fn_space = self.mixed_space
    W = fn_space * fn_space
    V1, Q1, V2, Q2 = W

    # Set the boundary conditions for each function space
    # These will be identical
    self.linearize_bcs(function_spaces=(V1, Q1))
    bcs1 = self.collect_bcs()

    self.linearize_bcs(function_spaces=(V2, Q2))
    bcs2 = self.collect_bcs()

    bcs = [*bcs1, *bcs2]

    # Since the base flow is used to construct the Navier-Stokes Jacobian
    # which is used on the diagonal block for both real and imaginary components,
    # we have to duplicate the base flow for both components.  This does NOT
    # mean that the base flow literally has an imaginary component
    qB_aug = fd.Function(W)
    uB_re, pB_re, uB_im, pB_im = fd.split(qB_aug)
    qB_aug.subfunctions[0].interpolate(qB.subfunctions[0])
    qB_aug.subfunctions[1].interpolate(qB.subfunctions[1])
    qB_aug.subfunctions[2].interpolate(qB.subfunctions[0])
    qB_aug.subfunctions[3].interpolate(qB.subfunctions[1])

    # Create trial and test functions
    q_trial = fd.TrialFunction(W)
    q_test = fd.TestFunction(W)
    (v_re, s_re, v_im, s_im) = fd.split(q_test)

    # Construct the nonlinear residual for the Navier-Stokes equations
    F_re = self.residual((uB_re, pB_re), q_test=(v_re, s_re))
    F_im = self.residual((uB_im, pB_im), q_test=(v_im, s_im))

    def _inner(u, v):
      return ufl.inner(u, v) * ufl.dx

    # Shift each block of the linear form appropriately
    F11 = F_re - sigma.real * _inner(uB_re, v_re)
    F22 = F_im - sigma.real * _inner(uB_im, v_im)
    F12 = sigma.imag * _inner(uB_im, v_re)
    F21 = -sigma.imag * _inner(uB_re, v_im)
    F = F11 + F22 + F12 + F21

    # Differentiate to get the bilinear form for the Jacobian
    J = fd.derivative(F, qB_aug, q_trial)

    def M(q):
      """Mass matrix for the Navier-Stokes equations"""
      return self.inner_product(q, q_test, assemble=False, augmented=True)

    A = InverseOperator(J, M, bcs, W, solver_parameters)

    if adjoint:
      A = A.T

    return A
