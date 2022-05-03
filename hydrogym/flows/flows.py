import numpy as np
import firedrake as fd
from firedrake import dx, ds
from firedrake.petsc import PETSc
from ufl import sym, grad, dot, inner, nabla_grad, div, cos, sin, atan_2

def print(s):
    PETSc.Sys.Print(s)

class Flow:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n = fd.FacetNormal(self.mesh)

        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(mesh, 'CG', 2)
        self.pressure_space = fd.FunctionSpace(mesh, 'CG', 1)
        self.mixed_space = fd.MixedFunctionSpace([self.velocity_space, self.pressure_space])
        self.sol = fd.Function(self.mixed_space, name='q')
        self.split_solution()  # Break out and rename solution

    def save_checkpoint(self, h5_file):
        with fd.CheckpointFile(h5_file, 'w') as chk:
            chk.save_mesh(self.mesh)  # optional
            chk.save_function(self.sol)

    def load_checkpoint(self, h5_file):
        with fd.CheckpointFile(h5_file, 'r') as chk:
            mesh = chk.load_mesh('mesh')
            Flow.__init__(self, mesh)  # Reinitialize with new mesh
            self.sol = chk.load_function(self.mesh, 'q')
        
        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def split_solution(self):
        self.u, self.p = self.sol.split()
        self.u.rename('u')
        self.p.rename('p')

    def init_bcs(self, mixed=False):
        """Define all boundary conditions"""
        pass

    def function_spaces(self, mixed=False):
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
        q = self.sol

        F = self.steady_form(q, fd.TestFunctions(self.mixed_space))  # Nonlinear variational form
        J = fd.derivative(F, q)    # Jacobian

        bcs = self.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, q, bcs, J)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()

        return q
        
    def collect_measurements(self):
        pass


class Cylinder(Flow):
    from .mesh.cylinder.mesh import INLET, FREESTREAM, OUTLET, CYLINDER
    MAX_CONTROL = 0.5*np.pi

    def __init__(self, mesh_name='noack', controller=None):
        """
        controller(t, y) -> omega
        y = (CL, CD)
        omega = scalar rotation rate
        """
        from .mesh.cylinder.mesh import load_mesh
        mesh = load_mesh(name=mesh_name)

        self.Re = fd.Constant(100)
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh)

        self.controller = None
        self.rotation_rate = fd.Constant(0.0)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # First set up tangential boundaries to cylinder
        x, y = fd.SpatialCoordinate(self.mesh)
        # Angle from origin
        theta = atan_2(y, x)
        rad = fd.Constant(0.5)
        u_tan = (self.rotation_rate*rad*sin(theta), self.rotation_rate*rad*cos(theta))  # Tangential velocity

        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_cylinder = fd.DirichletBC(V, u_tan, self.CYLINDER)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def steady_form(self, q, q_test):
        (u, p) = fd.split(q)
        (v, s) = q_test
        nu = fd.Constant(1/self.Re)

        F  = dot(dot(u, nabla_grad(u)), v)*dx \
            + inner(self.sigma(u, p), self.epsilon(v))*dx \
            + dot(p*self.n, v)*ds - inner(nu*nabla_grad(u)*self.n, v)*ds \
            + dot(div(u), s)*dx
        return F

    def compute_forces(self, u, p):
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = fd.assemble(2*force[1]*ds(self.CYLINDER))
        CD = fd.assemble(2*force[0]*ds(self.CYLINDER))
        return CL, CD

    def clamp(self, u):
        return max(-self.MAX_CONTROL, min(self.MAX_CONTROL, u))

    def update_control(self, u):
        self.rotation_rate.assign(
            self.clamp( u )
        )

    def collect_measurements(self):
        return self.compute_forces(self.u, self.p)