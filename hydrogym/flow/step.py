import firedrake as fd
import ufl
from firedrake import ds
from ufl import dot, exp, grad

from .base import FlowConfigBase


class Step(FlowConfigBase):
    ACT_DIM = 1
    MAX_CONTROL = 0.1  # Arbitrary... should tune this... TODO:  DEPRECATED??
    TAU = 0.005  # Time constant for controller damping (0.01*instability frequency)
    DEFAULT_MESH = "fine"
    DEFAULT_REYNOLDS = 500

    from .mesh.step import CONTROL, INLET, OUTLET, SENSOR, WALL

    def get_mesh_loader(self):
        from .mesh.step import load_mesh

        return load_mesh

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
            V,
            fd.interpolate(fd.Constant((0, 0)), V),
            (self.WALL, self.SENSOR)
            # V, fd.interpolate(fd.Constant((0, 0)), V), self.WALL
        )
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)
        self.bcu_actuation = [
            fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CONTROL)
        ]

        self.set_control(self.control)

    def linearize_bcs(self, mixed=True):
        self.reset_control(mixed=mixed)
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

    # def set_control(self, control=None):
    #     """
    #     Sets the blowing/suction at the leading edge
    #     """
    #     if control is None:
    #         control = 0.0
    #     self.control = self.enlist_controls(control)

    #     c = fd.Constant(self.control[0])
    #     if hasattr(self, "bcu_actuation"):
    #         self.bcu_actuation._function_arg.assign(
    #             fd.project(c * self.u_ctrl, self.velocity_space)
    #         )

    # def control_vec(self, act_idx=0):
    #     (v, _) = fd.TestFunctions(self.mixed_space)
    #     self.linearize_bcs()

    #     # self.linearize_bcs() should have reset control, need to perturb it now
    #     self.set_control(1.0)
    #     B = fd.assemble(
    #         inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs()
    #     )  # As fd.Function

    #     self.reset_control()
    #     return [B]

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
