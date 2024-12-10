import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake import ds
from firedrake.pyplot import tricontourf
from ufl import as_vector, atan2, cos, dot, sign, sin, sqrt

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC

# # Velocity probes
# xp = np.linspace(1.0, 10.0, 16)
# yp = np.linspace(-2.0, 2.0, 4)
# X, Y = np.meshgrid(xp, yp)
# DEFAULT_VEL_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

# # Pressure probes (spaced equally around the cylinder)
# xp = np.linspace(1.0, 4.0, 4)
# yp = np.linspace(-0.66, 0.66, 3)
# X, Y = np.meshgrid(xp, yp)
# DEFAULT_PRES_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

# RADIUS = 0.5
# DEFAULT_PRES_PROBES = [
#     (RADIUS * np.cos(theta), RADIUS * np.sin(theta))
#     for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False)
# ]

RADIUS = 0.5


class CylinderBase(FlowConfig):
  # Velocity probes
  xp = np.linspace(1.0, 4.0, 4)
  yp = np.linspace(-0.66, 0.66, 3)
  X, Y = np.meshgrid(xp, yp)
  DEFAULT_VEL_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

  # Pressure probes (spaced equally around the cylinder)
  xp = np.linspace(1.0, 4.0, 4)
  yp = np.linspace(-0.66, 0.66, 3)
  X, Y = np.meshgrid(xp, yp)
  DEFAULT_PRES_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

  # Vorticity probes (spaced equally around the cylinder)
  xp = np.linspace(1.0, 4.0, 4)
  yp = np.linspace(-0.66, 0.66, 3)
  X, Y = np.meshgrid(xp, yp)
  DEFAULT_VORT_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

  DEFAULT_REYNOLDS = 100
  DEFAULT_MESH = "medium"
  DEFAULT_DT = 1e-2

  # MAX_CONTROL = 0.5 * np.pi
  # TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
  # TAU = 0.278  # Time constant for controller damping (0.05*vortex shedding period)
  TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)

  # Domain labels
  FLUID = 1
  INLET = 2
  FREESTREAM = 3
  OUTLET = 4
  CYLINDER = 5

  MESH_DIR = os.path.abspath(f"{__file__}/..")

  @property
  def num_inputs(self) -> int:
    return 1  # Rotary control

  def configure_observations(self,
                             obs_type=None,
                             probe_obs_types={}) -> ObservationFunction:
    if obs_type is None:
      obs_type = "lift_drag"

    supported_obs_types = {
        **probe_obs_types,
        "lift_drag":
            ObservationFunction(self.compute_forces, num_outputs=2),
    }

    if obs_type not in supported_obs_types:
      raise ValueError(f"Invalid observation type {obs_type}")

    return supported_obs_types[obs_type]

  def init_bcs(self, function_spaces=None):
    if function_spaces is None:
      V, Q = self.function_spaces(mixed=True)
    else:
      V, Q = function_spaces

    # Define the static boundary conditions
    self.U_inf = fd.Constant((1.0, 0.0))
    self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
    self.bcu_freestream = fd.DirichletBC(
        V.sub(1), fd.Constant(0.0), self.FREESTREAM)  # Symmetry BCs
    self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    # Define time-varying boundary conditions for the actuation
    u_bc = self.cyl_velocity_field
    self.bcu_actuation = [ScaledDirichletBC(V, u_bc, self.CYLINDER)]

    # Reset the control with the current mixed (or not) function spaces
    self.set_control(self.control_state)

  @property
  def cyl_velocity_field(self):
    """Velocity vector for boundary condition"""
    raise NotImplementedError

  def collect_bcu(self) -> list[fd.DirichletBC]:
    return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

  def collect_bcp(self) -> list[fd.DirichletBC]:
    return [self.bcp_outflow]

  def compute_forces(self, q: fd.Function = None) -> tuple[float]:
    """Compute dimensionless lift/drag coefficients on cylinder

        Args:
            q (fd.Function, optional):
                Flow state to compute shear force from, if not the current state.

        Returns:
            Iterable[float]: Tuple of (lift, drag) coefficients
        """
    if q is None:
      q = self.q
    (u, p) = fd.split(q)
    # Lift/drag on cylinder
    force = -dot(self.sigma(u, p), self.n)
    CL = fd.assemble(2 * force[1] * ds(self.CYLINDER))
    CD = fd.assemble(2 * force[0] * ds(self.CYLINDER))
    return CL, CD

  # get net shear force acting tangential to the surface of the cylinder
  def shear_force(self, q: fd.Function = None) -> float:
    """Net shear force acting tangentially to the cylinder surface

        Implements the general case of the article below:
        http://www.homepages.ucl.ac.uk/~uceseug/Fluids2/Notes_Viscosity.pdf

        Args:
            q (fd.Function, optional):
                Flow state to compute shear force from, if not the current state.

        Returns:
            float: Tangential shear force
        """
    if q is None:
      q = self.q
    (u, p) = fd.split(q)
    (v, s) = fd.TestFunctions(self.mixed_space)

    # der of velocity wrt to the unit normal at the surface of the cylinder
    # equivalent to directional derivative along normal:
    du_dn = dot(self.epsilon(u), self.n)

    # Get unit tangent vector
    # pulled from https://fenics-shells.readthedocs.io/_/downloads/en/stable/pdf/
    t = as_vector((-self.n[1], self.n[0]))

    du_dn_t = (dot(du_dn, t)) * t

    # get the sign from the tangential cmpnt
    direction = sign(dot(du_dn, t))

    return fd.assemble(
        (direction / self.Re * sqrt(du_dn_t[0]**2 + du_dn_t[1]**2)) *
        ds(self.CYLINDER))

  # TODO: Add back in when linearization is fixed
  # def linearize_control(self, qB=None):
  #     """
  #     Solve linear problem with nonzero Dirichlet BCs to derive forcing term for unsteady DNS
  #     """
  #     if qB is None:
  #         qB = self.solve_steady()

  #     A = self.linearize_dynamics(qB, adjoint=False)
  #     # M = self.mass_matrix()
  #     self.linearize_bcs()  # Linearize BCs first (sets freestream to zero)
  #     self.set_control([1.0])  # Now change the cylinder rotation  TODO: FIX

  #     (v, _) = fd.TestFunctions(self.mixed_space)
  #     zero = fd.inner(fd.Constant((0, 0)), v) * fd.dx  # Zero RHS for linear form

  #     f = fd.Function(self.mixed_space)
  #     fd.solve(A == zero, f, bcs=self.collect_bcs())
  #     return f

  def linearize_bcs(self, function_spaces=None):
    self.reset_controls(function_spaces=function_spaces)
    self.bcu_inflow.set_value(fd.Constant((0, 0)))
    self.bcu_freestream.set_value(fd.Constant(0.0))

  def evaluate_objective(self, q: fd.Function = None, averaged_objective_values=None, lambda_value=0.2, return_objective_values=False) -> float:
    """The objective function for this flow is the drag coefficient"""
    if averaged_objective_values is None:
        CL, CD = self.compute_forces(q=q)
        if return_objective_values:
          return CL, CD
    else:
        CL, CD = averaged_objective_values
    # return np.square(CD) + lambda_value * np.square(CL)
    return np.abs(CD) + lambda_value * np.abs(CL)

  def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
    if clim is None:
      clim = (-2, 2)
    if levels is None:
      levels = np.linspace(*clim, 10)
    vort = fd.project(fd.curl(self.u), self.pressure_space)
    im = tricontourf(
        vort,
        cmap=cmap,
        levels=levels,
        vmin=clim[0],
        vmax=clim[1],
        extend="both",
        **kwargs,
    )

    cyl = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
    im.axes.add_artist(cyl)


class RotaryCylinder(CylinderBase):
  MAX_CONTROL_LOW = -1.0
  MAX_CONTROL_UP = 1.0
  CONTROL_SCALING = 5.0
  DEFAULT_DT = 1e-2

  @property
  def cyl_velocity_field(self):
    # Set up tangential boundaries to cylinder
    theta = atan2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
    self.rad = fd.Constant(RADIUS)
    # Tangential velocity
    return ufl.as_tensor((-self.rad * sin(theta), self.rad * cos(theta)))


class Cylinder(CylinderBase):
  MAX_CONTROL_LOW = -0.1
  MAX_CONTROL_UP = 0.1
  CONTROL_SCALING = 1.0
  DEFAULT_DT = 1e-2

  @property
  def cyl_velocity_field(self):
    """Velocity vector for boundary condition

        Blowing/suction actuation on the cylinder wall, following Rabault, et al (2018)
        https://arxiv.org/abs/1808.07664
        """

    # Set up tangential boundaries to cylinder
    theta = atan2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
    pi = ufl.pi
    self.rad = fd.Constant(RADIUS)

    omega = pi / 18  # 10 degree jet width

    theta_up = 0.5 * pi
    A_up = ufl.conditional(
        abs(theta - theta_up) < omega / 2,
        pi / (2 * omega * self.rad**2) * ufl.cos(
            (pi / omega) * (theta - theta_up)),
        0.0,
    )

    theta_lo = -0.5 * pi
    A_lo = ufl.conditional(
        abs(theta - theta_lo) < omega / 2,
        -pi / (2 * omega * self.rad**2) * ufl.cos(
            (pi / omega) * (theta - theta_lo)),
        0.0,
    )

    # Normal velocity (blowing/suction) at the cylinder wall
    return ufl.as_tensor((self.x, self.y)) * (A_up + A_lo)
