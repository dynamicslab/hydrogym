import firedrake as fd
import numpy as np
import ufl
from firedrake import ds
from ufl import as_vector, atan_2, cos, dot, sign, sin, sqrt

from .base import FlowConfigBase


class Cylinder(FlowConfigBase):
    from .mesh.cylinder import CYLINDER, FREESTREAM, INLET, OUTLET

    DEFAULT_MESH = "medium"
    DEFAULT_REYNOLDS = 100
    ACT_DIM = 1
    MAX_CONTROL = 0.5 * np.pi
    TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
    # TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)

    def __init__(
        self, account_for_skin_friction=False, control_method="direct", **kwargs
    ):
        self.control_method = control_method
        self.account_for_skin_friction = account_for_skin_friction
        super().__init__(**kwargs)

    def get_mesh_loader(self):
        from .mesh.cylinder import load_mesh

        return load_mesh

    def initialize_state(self):
        super().initialize_state()
        self.U_inf = fd.Constant((1.0, 0.0))

        # Set up tangential boundaries to cylinder
        theta = atan_2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        rad = fd.Constant(0.5)
        self.u_ctrl = [
            ufl.as_tensor((rad * sin(theta), rad * cos(theta)))
        ]  # Tangential velocity

        # I_cm = 1/2 M R**2
        # taking I_cm for a plexiglass cylinder with R=0.05m & length = 1m
        self.I_cm = 0.0115846
        self.k_damp = 1 / self.TAU

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

        self.set_control(self.control)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def compute_forces(self, q=None):
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = fd.assemble(2 * force[1] * ds(self.CYLINDER))
        CD = fd.assemble(2 * force[0] * ds(self.CYLINDER))
        return CL, CD

    # get net shear force acting tangential to the surface of the cylinder
    # Implementing the general case of the article below:
    # http://www.homepages.ucl.ac.uk/~uceseug/Fluids2/Notes_Viscosity.pdf
    def shear_force(self, q=None):
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

    def linearize_control(self, qB=None):
        """
        Solve linear problem with nonzero Dirichlet BCs to derive forcing term for unsteady DNS
        """
        if qB is None:
            qB = self.solve_steady()

        A = self.linearize_dynamics(qB, adjoint=False)
        # M = self.mass_matrix()
        self.linearize_bcs()  # Linearize BCs first (sets freestream to zero)
        self.set_control([1.0])  # Now change the cylinder rotation  TODO: FIX

        (v, _) = fd.TestFunctions(self.mixed_space)
        zero = fd.inner(fd.Constant((0, 0)), v) * fd.dx  # Zero RHS for linear form

        f = fd.Function(self.mixed_space)
        fd.solve(A == zero, f, bcs=self.collect_bcs())
        return f

    def linearize_bcs(self, mixed=True):
        self.reset_control(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def num_controls(self):
        return 1

    def get_observations(self):
        return self.compute_forces()

    def set_damping(self, k_new):
        if not isinstance(k_new, list):
            k_new = [k_new]
        self.k_damp = k_new
        return

    def get_inertia(self):
        return self.I_cm

    def get_damping(self):
        return self.k_damp

    def get_ctrl_state(self):
        return self.control

    def update_controls(self, act, dt):
        if self.control_method == "indirect":
            """
            Update BCS of cylinder to new angular velocity using implicit euler solver according to diff eqn:

            omega_t[i+1] = omega_t[i] + (d_omega/dt)_t[i+1] * dt
                        = omega_t[i] + (control_t[i+1] - k_damping*omega_t[i+1] + shear_torque)/I_cm * dt

            omega_t[i+1] can be solved for directly in order to avoid using a costly root solver

            TODO: Generalize for other flows and add to FlowConfigBase
            """
            if self.account_for_skin_friction:
                F_s = self.shear_force()
                tau_s = F_s * float(self.rad)
            else:
                tau_s = 0

            self.control[0] = [
                (self.control[0] + (act[0] + tau_s) * dt / self.I_cm)
                / (1 + self.k_damp * dt / self.I_cm)
            ]
            return self.control
        else:
            return super().update_controls(act, dt)

    def evaluate_objective(self, q=None):
        CL, CD = self.compute_forces(q=q)
        return CD
