import firedrake as fd
import numpy as np
import ufl
from firedrake import ds
from ufl import atan_2, cos, dot, sin

from .base import FlowConfigBase


class Cylinder(FlowConfigBase):
    from .mesh.cylinder import CYLINDER, FREESTREAM, INLET, OUTLET
    DEFAULT_MESH = "medium"
    DEFAULT_REYNOLDS = 100
    ACT_DIM = 1
    MAX_CONTROL = 0.5 * np.pi
    # TAU = 0.556  # Time constant for controller damping (0.1*vortex shedding period)
    TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)

    # def __init__(self, Re=100, mesh="medium", restart=None):
    #     """
    #     controller(t, y) -> omega
    #     y = (CL, CD)
    #     omega = scalar rotation rate
    #     """
    #     from .mesh.cylinder import load_mesh

    #     mesh = load_mesh(name=mesh)
    #     super().__init__(mesh, Re, restart=restart)

    def get_mesh_loader(self):
        from .mesh.cylinder import load_mesh
        return load_mesh

    def initialize_state(self):
        super().initialize_state()
        self.U_inf = fd.Constant((1.0, 0.0))

        # Set up tangential boundaries to cylinder
        theta = atan_2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        rad = fd.Constant(0.5)
        self.u_ctrl = [ufl.as_tensor(
            (rad * sin(theta), rad * cos(theta))
        )]  # Tangential velocity

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs
        self.bcu_actuation = [fd.DirichletBC(
            V, fd.interpolate(fd.Constant((0, 0)), V), self.CYLINDER
        )]
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

    # def control_vec(self, act_idx=0):
    #     # TODO: should this be in FlowConfigBase??
    #     (v, _) = fd.TestFunctions(self.mixed_space)
    #     self.linearize_bcs()

    #     # self.linearize_bcs() should have reset control, need to perturb it now
    #     self.set_control(1.0)
    #     B = fd.assemble(
    #         inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs()
    #     )  # As fd.Function

    #     self.reset_control()
    #     return [B]

    def num_controls(self):
        return 1

    def get_observations(self):
        return self.compute_forces()

    def evaluate_objective(self, q=None):
        CL, CD = self.compute_forces(q=q)
        return CD
