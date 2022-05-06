import numpy as np
import firedrake as fd
from firedrake import dx, ds
from firedrake.petsc import PETSc

import ufl
from ufl import sym, curl, dot, inner, nabla_grad, div, cos, sin, atan_2

def print(s):
    PETSc.Sys.Print(s)

class Flow:
    def __init__(self, mesh, h5_file=None):
        self.mesh = mesh
        self.n = fd.FacetNormal(self.mesh)
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(mesh, 'CG', 2)
        self.pressure_space = fd.FunctionSpace(mesh, 'CG', 1)
        self.mixed_space = fd.MixedFunctionSpace([self.velocity_space, self.pressure_space])
        self.q = fd.Function(self.mixed_space, name='q')
        self.split_solution()  # Break out and rename solution

        # TODO: Do this without having to reinitialize everything?
        if h5_file is not None:
            self.load_checkpoint(h5_file)

    def save_checkpoint(self, h5_file):
        with fd.CheckpointFile(h5_file, 'w') as chk:
            chk.save_mesh(self.mesh)  # optional
            chk.save_function(self.q)

    def load_checkpoint(self, h5_file):
        with fd.CheckpointFile(h5_file, 'r') as chk:
            mesh = chk.load_mesh('mesh')
            Flow.__init__(self, mesh)  # Reinitialize with new mesh
            self.q = chk.load_function(self.mesh, 'q')
        
        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def split_solution(self):
        self.u, self.p = self.q.split()
        self.u.rename('u')
        self.p.rename('p')

    def vorticity(self):
        vort = fd.project(curl(self.u), self.pressure_space)
        vort.rename('vort')
        return vort

    def init_bcs(self, mixed=False):
        """Define all boundary conditions"""
        pass

    def function_spaces(self, mixed=True):
        if mixed:
            V = self.mixed_space.sub(0)
            Q = self.mixed_space.sub(1)
        else:
            V = self.velocity_space
            Q = self.pressure_space
        return V, Q

    def collect_bcu(self):
        """List of velocity boundary conditions"""

    def collect_bcp(self):
        """List of pressure boundary conditions"""

    def collect_bcs(self):
        return self.collect_bcu() + self.collect_bcp()
    
    # Define symmetric gradient
    def epsilon(self, u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(self, u, p):
        return 2*(1/self.Re)*self.epsilon(u) - p*fd.Identity(len(u))

    def steady_form(self, w, w_test):
        """Define nonlinear variational problem for steady-state NS"""
        pass

    def solve_steady(self):
        self.init_bcs(mixed=True)

        F = self.steady_form()  # Nonlinear variational form
        J = fd.derivative(F, self.q)    # Jacobian

        bcs = self.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, self.q, bcs, J)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()

        return self.q.copy(deepcopy=True)
    
    def steady_form(self, q=None):
        if q is None: q = self.q
        (u, p) = fd.split(q)
        (v, s) = fd.TestFunctions(self.mixed_space)
        nu = fd.Constant(1/ufl.real(self.Re))

        F  = inner(dot(u, nabla_grad(u)), v)*dx \
            + inner(self.sigma(u, p), self.epsilon(v))*dx \
            + inner(p*self.n, v)*ds - inner(nu*nabla_grad(u)*self.n, v)*ds \
            + inner(div(u), s)*dx
        return F

    def linearized_forms(self, qB):
        (u, _) = fd.TrialFunctions(self.mixed_space)
        (v, _) = fd.TestFunctions(self.mixed_space)
        F = self.steady_form(q=qB)
        L = -fd.derivative(F, qB)
        M = inner(u, v)*dx
        return L, M

    def linearize(self, qB, control=False, backend='petsc'):
        A_form, M_form = self.linearized_forms(qB)
        self.linearize_bcs()
        A = fd.assemble(A_form, bcs=self.collect_bcs()).petscmat  # Dynamics matrix
        M = fd.assemble(M_form, bcs=self.collect_bcs()).petscmat  # Mass matrix
        if control and self.num_controls()!=0:
            B = [self.linearize_control(i) for i in range(self.num_controls())]
            sys = M, A, B
        else:
            sys = M, A

        if backend=='scipy':
            from ..utils import system_to_scipy
            sys = system_to_scipy(sys)
        return sys

    def linearize_control(self, act_idx=0):
        """Return a PETSc.Vec corresponding to the column of the control matrix"""
        pass

    def collect_observations(self):
        pass

    def set_control(self, u):
        pass

    def reset_control(self):
        pass

    def num_controls(self):
        return 0


class Cylinder(Flow):
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

    def linearize_bcs(self):
        self.omega.assign(0.0)
        self.init_bcs(mixed=True)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def compute_forces(self, u, p):
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

    def set_control(self, omega):
        self.omega.assign(omega)
        self.update_rotation()

        # TODO: Limit max control
        # self.rotation_rate.assign(
        #     self.clamp( omega )
        # )

    def reset_control(self):
        self.set_control(0.0)

    def linearize_control(self, act_idx=0):
        (v, _) = fd.TestFunctions(self.mixed_space)

        # self.linearize_bcs() should have already reset control, need to perturb it now
        eps = fd.Constant(1e-2)
        self.set_control(eps)
        B = fd.assemble(inner(fd.Constant((0, 0)), v)*dx, bcs=self.collect_bcs())  # As fd.Function
        # Convert to PETSc.Vec
        with B.dat.vec_ro as vec:
            Bvec = vec/eps

        # Now unset the perturbed control
        self.reset_control()
        return Bvec

    def num_controls(self):
        return 1

    def collect_observations(self):
        return self.compute_forces(self.u, self.p)

class Pinball(Flow):
    from .mesh.pinball import INLET, FREESTREAM, OUTLET, CYLINDER
    def __init__(self, mesh_name='coarse', h5_file=None, Re=20):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        from .mesh.pinball import load_mesh
        mesh = load_mesh(name=mesh_name)

        self.Re = fd.Constant(ufl.real(Re))
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh, h5_file=h5_file)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        # self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_freestream = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.FREESTREAM)  # Symmetry BCs
        self.bcu_cylinder = fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.CYLINDER)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def compute_forces(self, u, p):
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = [fd.assemble(2*force[1]*ds(cyl)) for cyl in self.CYLINDER]
        CD = [fd.assemble(2*force[0]*ds(cyl)) for cyl in self.CYLINDER]
        return CL, CD


class Cavity(Flow):
    from .mesh.cavity import INLET, FREESTREAM, OUTLET, SLIP, WALL
    def __init__(self, h5_file=None, Re=5000):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        from .mesh.cavity import load_mesh
        mesh = load_mesh()

        self.Re = fd.Constant(ufl.real(Re))
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh, h5_file=h5_file)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define actual boundary conditions
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.FREESTREAM)
        self.bcu_noslip = fd.DirichletBC(V, fd.interpolate(fd.Constant((0, 0)), V), self.WALL)
        self.bcu_slip = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.SLIP)  # Free-slip
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        self.bcu_slip = None
        raise NotImplementedError  # Still need to add slip boundary

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_noslip, self.bcu_slip]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def compute_forces(self, u, p):
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = [fd.assemble(2*force[1]*ds(cyl)) for cyl in self.CYLINDER]
        CD = [fd.assemble(2*force[0]*ds(cyl)) for cyl in self.CYLINDER]
        return CL, CD

class Step(Flow):
    pass