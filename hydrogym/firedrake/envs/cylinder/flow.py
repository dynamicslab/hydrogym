import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake import ds
from firedrake.pyplot import tricontourf
from ufl import as_vector, atan2, cos, dot, sign, sin, sqrt

from hydrogym.firedrake import FlowConfig, ScaledDirichletBC

# Velocity probes
xp = np.linspace(1.0, 10.0, 16)
yp = np.linspace(-2.0, 2.0, 4)
X, Y = np.meshgrid(xp, yp)
probes = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]


class CylinderBase(FlowConfig):
    DEFAULT_REYNOLDS = 100
    DEFAULT_MESH = "medium"
    DEFAULT_DT = 1e-2

    MAX_CONTROL = 0.5 * np.pi
    # TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
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

    @property
    def num_outputs(self) -> int:
        # This may be lift/drag or a grid of velocity probes
        return self._num_outputs

    def __init__(self, **config):
        obs_type = config.pop("observation_type", "lift_drag")
        supported_obs_types = {
            "lift_drag": (2, self.compute_forces),
            "velocity_probes": (2 * len(probes), self.velocity_probe),
        }

        if obs_type not in supported_obs_types:
            raise ValueError(f"Invalid observation type {obs_type}")

        self._num_outputs, self._get_observations = supported_obs_types[obs_type]

        super().__init__(**config)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define the static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs
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

    def velocity_probe(self, q: fd.Function = None) -> float:
        """Probe velocity in the wake"""
        pass

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

    def get_observations(self) -> tuple[float]:
        return self._get_observations()

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
    MAX_CONTROL = 0.5 * np.pi
    DEFAULT_DT = 1e-2

    @property
    def cyl_velocity_field(self):
        # Set up tangential boundaries to cylinder
        theta = atan2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        self.rad = fd.Constant(0.5)
        # Tangential velocity
        return ufl.as_tensor((self.rad * sin(theta), self.rad * cos(theta)))


class Cylinder(CylinderBase):
    MAX_CONTROL = 0.1
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
        self.rad = fd.Constant(0.5)

        omega = pi / 18  # 10 degree jet width

        theta_up = 0.5 * pi
        A_up = ufl.conditional(
            abs(theta - theta_up) < omega / 2,
            pi
            / (2 * omega * self.rad**2)
            * ufl.cos((pi / omega) * (theta - theta_up)),
            0.0,
        )

        theta_lo = -0.5 * pi
        A_lo = ufl.conditional(
            abs(theta - theta_lo) < omega / 2,
            pi
            / (2 * omega * self.rad**2)
            * ufl.cos((pi / omega) * (theta - theta_lo)),
            0.0,
        )

        # Normal velocity (blowing/suction) at the cylinder wall
        return ufl.as_tensor((self.x, self.y)) * (A_up + A_lo)
