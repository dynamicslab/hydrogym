import os
from typing import Iterable

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake import ds
from ufl import as_vector, atan_2, cos, dot, sign, sin, sqrt

from hydrogym.firedrake import DampedActuator, FlowConfig


class Cylinder(FlowConfig):
    DEFAULT_REYNOLDS = 100
    DEFAULT_MESH = "medium"
    DEFAULT_DT = 1e-2

    OBS_DIM = 2
    MAX_CONTROL = 0.5 * np.pi
    TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
    # TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)
    I_CM = 0.0115846  # Moment of inertia  (TODO: Test and switch to this value)
    # I_CM = 1.0  # Moment of inertia

    # Domain labels
    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    CYLINDER = 5

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    def initialize_state(self):
        super().initialize_state()
        self.U_inf = fd.Constant((1.0, 0.0))

        # Set up tangential boundaries to cylinder
        theta = atan_2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        self.rad = fd.Constant(0.5)
        self.u_ctrl = [
            ufl.as_tensor((self.rad * sin(theta), self.rad * cos(theta)))
        ]  # Tangential velocity

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs
        self.bcu_actuation = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CYLINDER)
        ]
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        # Reset the control with the current mixed (or not) function spaces
        self.set_control(self.control_state)

    def create_actuator(self) -> DampedActuator:
        return DampedActuator(damping=1 / self.TAU, inertia=self.I_CM)

    def collect_bcu(self) -> Iterable[fd.DirichletBC]:
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

    def collect_bcp(self) -> Iterable[fd.DirichletBC]:
        return [self.bcp_outflow]

    def compute_forces(self, q: fd.Function = None) -> Iterable[float]:
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
        # https://math.libretexts.org/Courses/University_of_California_Davis/UCD_Mat_21C%3A_Multivariate_Calculus/13%3A_Partial_Derivatives/13.5%3A_Directional_Derivatives_and_Gradient_Vectors#mjx-eqn-DD2v
        du_dn = dot(self.epsilon(u), self.n)

        # Get unit tangent vector
        # pulled from https://fenics-shells.readthedocs.io/_/downloads/en/stable/pdf/
        t = as_vector((-self.n[1], self.n[0]))

        du_dn_t = (dot(du_dn, t)) * t

        # get the sign from the tangential cmpnt
        direction = sign(dot(du_dn, t))

        return fd.assemble(
            (direction / self.Re * sqrt(du_dn_t[0] ** 2 + du_dn_t[1] ** 2))
            * ds(self.CYLINDER)
        )

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

    def linearize_bcs(self, mixed=True):
        self.reset_controls(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def get_observations(self) -> Iterable[FlowConfig.ObsType]:
        return self.compute_forces()

    def evaluate_objective(self, q: fd.Function = None) -> float:
        """The objective function for this flow is the drag coefficient"""
        CL, CD = self.compute_forces(q=q)
        return CD

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.u), self.pressure_space)
        im = fd.tricontourf(
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
