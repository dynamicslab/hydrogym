import firedrake as fd
import ufl
from firedrake import ds, dx
from ufl import dot, grad, inner

from ..core import FlowConfig


class Cavity(FlowConfig):
    MAX_CONTROL = 0.1  # Arbitrary... should tune this
    TAU = 0.075  # Time constant for controller damping (0.01*instability frequency)

    from .mesh.cavity import CONTROL, FREESTREAM, INLET, OUTLET, SENSOR, SLIP, WALL

    def __init__(self, h5_file=None, Re=7500, mesh="fine"):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        from .mesh.cavity import load_mesh

        mesh = load_mesh(name=mesh)

        self.Re = fd.Constant(ufl.real(Re))
        self.U_inf = fd.Constant((1.0, 0.0))

        super().__init__(mesh, h5_file=h5_file)

        self.control = fd.Constant(0.0)
        self.u_ctrl = ufl.as_tensor(
            (0.0 * self.x, -self.x * (1600 * self.x + 560) / 147)
        )  # Blowing/suction

        self.reset_control()

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )
        self.bcu_noslip = fd.DirichletBC(
            V, fd.interpolate(fd.Constant((0, 0)), V), self.WALL
        )
        self.bcu_slip = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.SLIP
        )  # Free-slip
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)
        self.bcu_actuation = fd.DirichletBC(
            V, fd.interpolate(fd.Constant((0, 0)), V), self.CONTROL
        )

        self.set_control(self.control)

    def linearize_bcs(self, mixed=True):
        self.reset_control(mixed=mixed)
        self.init_bcs(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))

    def collect_bcu(self):
        return [
            self.bcu_inflow,
            self.bcu_freestream,
            self.bcu_noslip,
            self.bcu_slip,
            self.bcu_actuation,
        ]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def set_control(self, control=None):
        """
        Sets the blowing/suction at the leading edge
        """
        if control is None:
            control = 0.0
        self.control.assign(control)

        if hasattr(self, "bcu_actuation"):
            self.bcu_actuation._function_arg.assign(
                fd.project(self.control * self.u_ctrl, self.velocity_space)
            )

    def get_control(self):
        return [self.control]

    def reset_control(self, mixed=False):

        self.set_control(0.0)
        self.init_bcs(mixed=mixed)

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

    def collect_observations(self, q=None):
        """Integral of wall-normal shear stress (see Barbagallo et al, 2009)"""
        if q is None:
            q = self.q
        (u, p) = q.split()
        return fd.assemble(-dot(grad(u[0]), self.n) * ds(self.SENSOR))

    # def solve_steady(self, **kwargs):
    #     if self.Re > 500:
    #         Re_final = self.Re.values()[0]
    #         logging.log(logging.INFO, "Re > 500, will need to ramp up steady solve")

    #         Re_init = []
    #         Re = Re_final/2
    #         while Re > 500:
    #             Re_init.insert(0, Re)
    #             Re = Re/2

    #         for Re, in Re_init:
    #             self.Re.assign(Re)
    #             super().solve_steady(**kwargs)
