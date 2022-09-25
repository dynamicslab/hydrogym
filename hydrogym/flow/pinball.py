import firedrake as fd
import numpy as np
import ufl
from firedrake import ds, dx
from ufl import atan_2, cos, dot, inner, sin

from .base import FlowConfigBase


class Pinball(FlowConfigBase):
    from .mesh.pinball import CYLINDER, FREESTREAM, INLET, OUTLET, rad, x0, y0

    NUM_ACT = len(CYLINDER)
    MAX_CONTROL = 0.5 * np.pi
    TAU = 1.0

    def __init__(self, Re=30, mesh="fine", restart=None):
        """ """
        from .mesh.pinball import load_mesh

        mesh = load_mesh(name=mesh)

        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh, Re, restart=restart)

        # self.omega = [fd.Constant(0.0) for _ in range(len(self.CYLINDER))]
        # self.omega = pyadjoint.create_overloaded_object(np.zeros(self.CYLINDER))
        # self.reset_control()

    def initialize_state(self):
        super().initialize_state()
        # Set up tangential boundaries for each cylinder
        self.rad = fd.Constant(self.rad)
        self.u_tan = []
        for cyl_idx in range(len(self.CYLINDER)):
            theta = atan_2(
                ufl.real(self.y - self.y0[cyl_idx]), ufl.real(self.x - self.x0[cyl_idx])
            )  # Angle from center of cylinder

            self.u_tan.append(
                ufl.as_tensor((self.rad * sin(theta), self.rad * cos(theta)))
            )  # Tangential velocity

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs
        self.bcu_cylinder = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), cyl)
            for cyl in self.CYLINDER
        ]
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        self.set_control(self.control)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_cylinder]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def linearize_bcs(self, mixed=True):
        self.reset_control(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def compute_forces(self, q=None):
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = [fd.assemble(2 * force[1] * ds(cyl)) for cyl in self.CYLINDER]
        CD = [fd.assemble(2 * force[0] * ds(cyl)) for cyl in self.CYLINDER]
        return CL, CD

    def set_control(self, omega=None):
        """
        Sets the rotation rate of the cylinder

        Note that for time-varying controls it will be better to adjust the rotation rate
        in the timestepper, e.g. with `solver.step(iter, control=omega)`.  This could be used
        to change rotation rate for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        if omega is None:
            omega = 0.0, 0.0, 0.0
        # for i in range(len(self.CYLINDER)):
        #     # self.omega[i].assign(omega[i])
        #     self.omega[i] = omega[i]
        self.control = self.enlist_controls(omega)

        if hasattr(self, "bcu_cylinder"):
            for cyl_idx in range(len(self.CYLINDER)):
                c = fd.Constant(self.control[cyl_idx])
                self.bcu_cylinder[cyl_idx]._function_arg.assign(
                    fd.interpolate(
                        c * self.u_tan[cyl_idx], self.velocity_space
                    )
                )

    def get_control(self):
        return self.omega

    def reset_control(self, mixed=False):
        self.set_control(omega=None)
        self.init_bcs(mixed=mixed)

    def control_vec(self, mixed=False):
        (v, _) = fd.TestFunctions(self.mixed_space)
        self.linearize_bcs()
        # self.linearize_bcs() should have reset control, need to perturb it now

        eps = fd.Constant(1.0)
        omega = [0.0, 0.0, 0.0]

        fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())

        B = []
        for i in range(self.num_controls()):
            omega[i] = eps  # Perturb the ith control
            self.set_control(omega)

            # Control as fd.Function
            B.append(
                fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())
            )

            self.reset_control(
                mixed=True
            )  # Have to have mixed function space for computing B functions

        # At the end the BC function spaces could be mixed or not
        self.reset_control(mixed=mixed)
        return B

    def num_controls(self):
        return 3

    def get_observations(self):
        CL, CD = self.compute_forces()
        return [*CL, *CD]

    def evaluate_objective(self, q=None):
        CL, CD = self.compute_forces(q=q)
        return sum(CD)
