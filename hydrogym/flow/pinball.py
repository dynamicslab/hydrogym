import firedrake as fd
import numpy as np
import ufl
from firedrake import ds
from ufl import atan_2, cos, dot, sin

from .base import FlowConfigBase


class Pinball(FlowConfigBase):
    from .mesh.pinball import CYLINDER, FREESTREAM, INLET, OUTLET, rad, x0, y0

    DEFAULT_MESH = "fine"
    DEFAULT_REYNOLDS = 30
    ACT_DIM = len(CYLINDER)
    MAX_CONTROL = 0.5 * np.pi
    TAU = 1.0

    def get_mesh_loader(self):
        from .mesh.pinball import load_mesh

        return load_mesh

    def initialize_state(self):
        super().initialize_state()
        self.U_inf = fd.Constant((1.0, 0.0))

        # Set up tangential boundaries for each cylinder
        self.rad = fd.Constant(self.rad)
        self.u_ctrl = []
        for cyl_idx in range(len(self.CYLINDER)):
            theta = atan_2(
                ufl.real(self.y - self.y0[cyl_idx]), ufl.real(self.x - self.x0[cyl_idx])
            )  # Angle from center of cylinder

            self.u_ctrl.append(
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
        self.bcu_actuation = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), cyl)
            for cyl in self.CYLINDER
        ]
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        self.set_control(self.control)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

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

    def reset_control(self, mixed=False):
        self.set_control(control=None)
        self.init_bcs(mixed=mixed)

    def num_controls(self):
        return 3

    # def update_state(self, control, dt):
    #     if self.control_method == "indirect":
    #         raise Exception("Indirect Control not yet implemented for this environment")
    #     else:
    #         """Add a damping factor to the controller response

    #         If actual control is u and input is v, effectively
    #             du/dt = (1/tau)*(v - u)
    #         """
    #         for (u, v) in zip(self.ctrl_state, control):
    #             u = u + (dt / self.TAU) * (v - u)

    def get_observations(self):
        CL, CD = self.compute_forces()
        return [*CL, *CD]

    def evaluate_objective(self, q=None):
        CL, CD = self.compute_forces(q=q)
        return sum(CD)
