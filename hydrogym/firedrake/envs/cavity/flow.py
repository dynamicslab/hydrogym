import os

import firedrake as fd
import ufl
from ufl import dot, ds, grad

from hydrogym.firedrake import FlowConfig


class Cavity(FlowConfig):
    DEFAULT_REYNOLDS = 7500
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-4

    MAX_CONTROL = 0.1
    TAU = 0.075  # Time constant for controller damping (0.01*instability frequency)

    # Domain labels
    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    SLIP = 5
    WALL = (6, 8)
    CONTROL = 7
    SENSOR = 8

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    def initialize_state(self):
        super().initialize_state()

        self.U_inf = fd.Constant((1.0, 0.0))
        self.u_ctrl = [
            ufl.as_tensor((0.0 * self.x, -self.x * (1600 * self.x + 560) / 147))
        ]  # Blowing/suction

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )
        self.bcu_noslip = fd.DirichletBC(
            # V, fd.interpolate(fd.Constant((0, 0)), V), (self.WALL, self.SENSOR)
            V,
            fd.interpolate(fd.Constant((0, 0)), V),
            self.WALL,
        )
        self.bcu_slip = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.SLIP
        )  # Free-slip
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)
        self.bcu_actuation = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CONTROL)
        ]

        self.set_control(self.control_state)

    def collect_bcu(self):
        return [
            self.bcu_inflow,
            self.bcu_freestream,
            self.bcu_noslip,
            self.bcu_slip,
            *self.bcu_actuation,
        ]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def linearize_bcs(self, mixed=True):
        self.reset_controls(mixed=mixed)
        self.init_bcs(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))

    def get_observations(self, q=None):
        """Integral of wall-normal shear stress (see Barbagallo et al, 2009)"""
        if q is None:
            q = self.q
        (u, p) = q.split()
        m = fd.assemble(-dot(grad(u[0]), self.n) * ds(self.SENSOR))
        return (m,)

    def evaluate_objective(self, q=None):
        # TODO: This should be *fluctuation* kinetic energy
        if q is None:
            q = self.q
        (u, p) = q.split()
        KE = 0.5 * fd.assemble(fd.inner(u, u) * fd.dx)
        return KE
        # m, = self.get_observations(q=q)
        # return m
