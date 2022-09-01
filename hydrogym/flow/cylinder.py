import firedrake as fd
import numpy as np
import ufl
from firedrake import ds, dx
from ufl import atan_2, cos, dot, inner, sin, sqrt, sign, as_vector

from ..core import FlowConfig


class Cylinder(FlowConfig):
    from .mesh.cylinder import CYLINDER, FREESTREAM, INLET, OUTLET

    MAX_CONTROL = 0.5 * np.pi
    # TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
    TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)

    def __init__(
        self,
        Re=100,
        mesh="medium",
        h5_file=None,
        control_method="direct",
        account_for_skin_friction=False,
    ):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        self.account_for_skin_friction = account_for_skin_friction
        self.control_method = control_method
        from .mesh.cylinder import load_mesh

        mesh = load_mesh(name=mesh)

        self.Re = fd.Constant(ufl.real(Re))
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh, h5_file=h5_file)

        # First set up tangential boundaries to cylinder
        self.omega = fd.Constant(0.0)
        theta = atan_2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        self.rad = fd.Constant(0.5)
        self.u_tan = ufl.as_tensor(
            (self.rad * sin(theta), self.rad * cos(theta))
        )  # Tangential velocity

        self.reset_control()

        # I_cm = MR**2 / 2

        # Is everything normalized wrt mass and D or are we just kind of choosing that?
        # Is there a reason we are normalizing by the diameter instead of by the radius? (R is what's used in the equation)
        # Should I just normalize by D**2
        self.ctrl_state = [float(self.omega)]
        self.I_cm = [0.006172]
        # ^^^ Moment of inertia about CoM of a Plexiglass Cylinder with a 2 inch radius and spanning a half meter test section of a wind tunnel
        self.controller_damping_coeff = [1 / self.TAU]

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs
        self.bcu_cylinder = fd.DirichletBC(
            V, fd.interpolate(fd.Constant((0, 0)), V), self.CYLINDER
        )
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        self.update_rotation()

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]

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

    def update_rotation(self):
        # If the boundary condition has already been defined, update it
        #   otherwise, the control will be applied with self.init_bcs()
        #
        # TODO: Is this necessary, or could it be combined with `set_control()`?
        if hasattr(self, "bcu_cylinder"):
            self.bcu_cylinder._function_arg.assign(
                fd.project(self.omega * self.u_tan, self.velocity_space)
            )

    def clamp(self, u):
        return max(-self.MAX_CONTROL, min(self.MAX_CONTROL, u))

    def linearize_control(self, qB=None):
        """
        Solve linear problem with nonzero Dirichlet BCs to derive forcing term for unsteady DNS
        """
        if qB is None:
            qB = self.solve_steady()

        A = self.linearize_dynamics(qB, adjoint=False)
        # M = self.mass_matrix()
        self.linearize_bcs()  # Linearize BCs first (sets freestream to zero)
        self.set_control(1.0)  # Now change the cylinder rotation

        (v, _) = fd.TestFunctions(self.mixed_space)
        zero = fd.inner(fd.Constant((0, 0)), v) * fd.dx  # Zero RHS for linear form

        f = fd.Function(self.mixed_space)
        fd.solve(A == zero, f, bcs=self.collect_bcs())
        return f

    def set_control(self, omega=None):
        """
        Sets the rotation rate of the cylinder

        Note that for time-varying controls it will be better to adjust the rotation rate
        in the timestepper with `solver.step(iter, control=omega)`.  This method could be used
        to change rotation rate for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        if omega is None:
            omega = 0.0
        self.omega.assign(omega)

        # TODO: Limit max control in a differentiable way
        # self.omega.assign(
        #     self.clamp( omega )
        # )

        self.update_rotation()

    def get_control(self):
        return [self.omega]

    def reset_control(self, mixed=False):
        self.set_control(0.0)
        self.init_bcs(mixed=mixed)

    def linearize_bcs(self, mixed=True):
        self.reset_control(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def initialize_control(self, act_idx=0):
        (v, _) = fd.TestFunctions(self.mixed_space)
        self.linearize_bcs()

        # self.linearize_bcs() should have reset control, need to perturb it now
        eps = fd.Constant(1.0)
        self.set_control(eps)
        B = fd.assemble(
            inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs()
        )  # As fd.Function

        self.reset_control()
        return [B]

    def num_controls(self):
        return 1

    def collect_observations(self):
        return self.compute_forces()

    def set_damping(self, k_new):
        if not isinstance(k_new, list):
            k_new = [k_new]
        self.controller_damping_coeff = k_new
        return

    def get_inertia(self):
        return self.I_cm

    def get_damping(self):
        return self.controller_damping_coeff

    def get_ctrl_state(self):
        return self.ctrl_state

    def update_state(self, control, dt):
        if self.control_method == "indirect":
            """
            Update BCS of cylinder to new angular velocity using implicit euler solver according to diff eqn:

            omega_t[i+1] = omega_t[i] + (d_omega/dt)_t[i+1] * dt
                        = omega_t[i] + (control_t[i+1] - k_damping*omega_t[i+1] + shear_torque)/I_cm * dt

            omega_t[i+1] can be solved for directly in order to avoid using a costly root solver
            """
            if self.account_for_skin_friction:
                F_s = self.shear_force()
                tau_s = F_s * float(self.rad)
            else:
                tau_s = 0

            for (state, ctrl, I_cm, k_damp) in zip(
                self.ctrl_state, control, self.I_cm, self.controller_damping_coeff
            ):
                self.ctrl_state = [
                    (state + (ctrl + tau_s) * dt / I_cm) / (1 + k_damp * dt / I_cm)
                ]
        else:
            """Add a damping factor to the controller response

            If actual control is u and input is v, effectively
                du/dt = (1/tau)*(v - u)
            """
            for (u, v) in zip(self.ctrl_state, control):
                u = u + (dt / self.TAU) * (v - u)
