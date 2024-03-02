import os

import firedrake as fd
import ufl
from ufl import dot, ds, grad

from hydrogym.firedrake import FlowConfig, ScaledDirichletBC


class Cavity(FlowConfig):
    DEFAULT_REYNOLDS = 7500
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-4
    DEFAULT_VELOCITY_ORDER = 1
    DEFAULT_STABILIZATION = "gls"

    FUNCTIONS = ("q", "qB")  # This flow needs a base flow to compute fluctuation KE

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

    @property
    def num_inputs(self) -> int:
        return 1  # Blowing/suction on leading edge

    @property
    def num_outputs(self) -> int:
        return 1  # Shear stress on trailing edge

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )
        self.bcu_noslip = fd.DirichletBC(V, fd.Constant((0, 0)), self.WALL)
        # Free-slip on top boundary
        self.bcu_slip = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.SLIP)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        # Define time-varying boundary conditions for actuation
        # This matches Barbagallo et al (2009), "Closed-loop control of an open cavity
        # flow using reduced-order models" https://doi.org/10.1017/S0022112009991418
        u_bc = ufl.as_tensor((0.0 * self.x, -self.x * (1600 * self.x + 560) / 147))
        self.bcu_actuation = [ScaledDirichletBC(V, u_bc, self.CONTROL)]

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
        u = q.subfunctions[0]
        m = fd.assemble(-dot(grad(u[0]), self.n) * ds(self.SENSOR))
        return (m,)

    def evaluate_objective(self, q=None, qB=None):
        if q is None:
            q = self.q
        if qB is None:
            qB = self.qB
        u = q.subfunctions[0]
        uB = qB.subfunctions[0]
        KE = 0.5 * fd.assemble(fd.inner(u - uB, u - uB) * fd.dx)
        return KE
