import numpy as np
import firedrake as fd
from firedrake import dx, ds
from firedrake.petsc import PETSc

import ufl
from ufl import sym, curl, dot, inner, nabla_grad, div, cos, sin, atan_2

from ..core import FlowConfig

class Cylinder(FlowConfig):
    from .mesh.cylinder import INLET, FREESTREAM, OUTLET, CYLINDER
    MAX_CONTROL = 0.5*np.pi

    def __init__(self, Re=100, mesh_name='noack', controller=None, h5_file=None):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        from .mesh.cylinder import load_mesh
        mesh = load_mesh(name=mesh_name)

        self.Re = fd.Constant(ufl.real(Re))
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh, h5_file=h5_file)

        self.omega = fd.Constant(0.0)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.FREESTREAM)  # Symmetry BCs
        self.bcu_cylinder = fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CYLINDER)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        self.update_rotation()

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def linearize_bcs(self, mixed=True):
        self.omega.assign(0.0)
        self.init_bcs(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def compute_forces(self, q=None):
        if q is None: q = self.q
        (u, p) = fd.split(q)
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = fd.assemble(2*force[1]*ds(self.CYLINDER))
        CD = fd.assemble(2*force[0]*ds(self.CYLINDER))
        return CL, CD

    def update_rotation(self):
        # return 
        # First set up tangential boundaries to cylinder
        theta = atan_2(ufl.real(self.y), ufl.real(self.x)) # Angle from origin
        rad = fd.Constant(0.5)
        self.u_tan = ufl.as_tensor((self.omega*rad*sin(theta), self.omega*rad*cos(theta)))  # Tangential velocity

        # If the boundary condition has already been defined, update it
        #   otherwise, the control will be applied with self.init_bcs()
        if hasattr(self, 'bcu_cylinder'):
            self.bcu_cylinder._function_arg.assign(
                fd.project(self.u_tan, self.velocity_space)
            )

    def clamp(self, u):
        return max(-self.MAX_CONTROL, min(self.MAX_CONTROL, u))

    def set_control(self, omega=None):
        """
        Sets the rotation rate of the cylinder

        Note that for time-varying controls it will be better to adjust the rotation rate
        in the timestepper, e.g. with `solver.step(iter, control=omega)`.  This could be used
        to change rotation rate for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        if omega is None: omega = 0.0
        self.omega.assign(omega)

        # TODO: Limit max control in a differentiable way
        # self.omega.assign(
        #     self.clamp( omega )
        # )

        self.update_rotation()


    def reset_control(self):
        self.set_control(0.0)
        self.init_bcs(mixed=False)

    def linearize_control(self, act_idx=0):
        (v, _) = fd.TestFunctions(self.mixed_space)
        self.linearize_bcs()
        # self.linearize_bcs() should have reset control, need to perturb it now
        eps = fd.Constant(1.0)
        self.set_control(eps)
        B = fd.assemble(inner(fd.Constant((0, 0)), v)*dx, bcs=self.collect_bcs())  # As fd.Function

        # # Convert to PETSc.Vec
        # with B.dat.vec_ro as vec:
        #     Bvec = vec/eps

        # Now unset the perturbed control
        # self.reset_control()
        # return Bvec

        self.reset_control()
        return B

    def num_controls(self):
        return 1

    def collect_observations(self):
        return self.compute_forces()