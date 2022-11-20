import os

import firedrake as fd
import ufl
from ufl import dot, ds, exp, grad

from hydrogym.firedrake import FlowConfig


class Step(FlowConfig):
    DEFAULT_REYNOLDS = 500
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-4

    MAX_CONTROL = 0.1  # Arbitrary... should tune this... TODO:  DEPRECATED??
    TAU = 0.005  # Time constant for controller damping (0.01*instability frequency)

    FLUID = 1
    INLET = 2
    OUTLET = 3
    WALL = 4
    CONTROL = 5
    SENSOR = 6

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    def initialize_state(self):
        super().initialize_state()
        self.U_inf = ufl.as_tensor((1.0 - ((self.y - 0.25) / 0.25) ** 2, 0.0 * self.y))
        self.u_ctrl = [
            ufl.as_tensor((0.0 * self.x, -self.x * (1600 * self.x + 560) / 147))
        ]  # Blowing/suction

    @property
    def nu(self):
        return fd.Constant(0.5 / ufl.real(self.Re))

    @property
    def body_force(self):
        delta = 0.1
        x0, y0 = 1.0, 0.25
        return ufl.as_tensor(
            (
                exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / delta**2),
                exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / delta**2),
            )
        )

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_noslip = fd.DirichletBC(
            V, fd.interpolate(fd.Constant((0, 0)), V), (self.WALL, self.SENSOR)
        )
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)
        self.bcu_actuation = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CONTROL)
        ]

        self.set_control(self.control_state)

    def linearize_bcs(self, mixed=True):
        self.reset_controls(mixed=mixed)
        self.init_bcs(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))

    def collect_bcu(self):
        return [
            self.bcu_inflow,
            self.bcu_noslip,
            *self.bcu_actuation,
        ]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def get_observations(self, q=None):
        """Integral of wall-normal shear stress (see Barbagallo et al, 2009)"""
        if q is None:
            q = self.q
        (u, p) = q.split()
        m = fd.assemble(-dot(grad(u[0]), self.n) * ds(self.SENSOR))
        return (m,)

    def evaluate_objective(self, q=None):
        if q is None:
            q = self.q
        (u, p) = q.split()
        KE = 0.5 * fd.assemble(fd.inner(u, u) * fd.dx)
        return KE
