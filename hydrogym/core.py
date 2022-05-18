import firedrake as fd
from firedrake import dx, ds

import ufl
from ufl import dot, inner, grad, nabla_grad, div, sym, curl

from typing import Optional

class FlowConfig:
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

    def save_checkpoint(self, h5_file, write_mesh=True, idx=None):
        with fd.CheckpointFile(h5_file, 'w') as chk:
            if write_mesh:
                chk.save_mesh(self.mesh)  # optional
            chk.save_function(self.q, idx=idx)

    def load_checkpoint(self, h5_file, idx=None, read_mesh=True):
        with fd.CheckpointFile(h5_file, 'r') as chk:
            if read_mesh:
                mesh = chk.load_mesh('mesh')
                FlowConfig.__init__(self, mesh)  # Reinitialize with new mesh
            else:
                assert hasattr(self, 'mesh')
            self.q = chk.load_function(self.mesh, 'q', idx=idx)
        
        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def split_solution(self):
        self.u, self.p = self.q.split()
        self.u.rename('u')
        self.p.rename('p')

    def vorticity(self, u=None):
        if u is None: u = self.u
        vort = fd.project(curl(u), self.pressure_space)
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

    def solve_steady(self, solver_parameters={}, stabilization=None):
        self.init_bcs(mixed=True)

        F = self.steady_form(stabilization=stabilization)  # Nonlinear variational form
        J = fd.derivative(F, self.q)    # Jacobian

        bcs = self.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, self.q, bcs, J)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
        solver.solve()

        return self.q.copy(deepcopy=True)
    
    def steady_form(self, q=None, stabilization=None):
        if q is None: q = self.q
        (u, p) = fd.split(q)
        (v, s) = fd.TestFunctions(self.mixed_space)
        nu = fd.Constant(1/ufl.real(self.Re))

        F  = inner(dot(u, nabla_grad(u)), v)*dx \
            + inner(self.sigma(u, p), self.epsilon(v))*dx \
            + inner(p*self.n, v)*ds - inner(nu*nabla_grad(u)*self.n, v)*ds \
            + inner(div(u), s)*dx

        if stabilization=='gls':
            # Galerkin least-squares stabilization (see Tezduyar, 1991)
            res = lambda U, u, p: dot(U, nabla_grad(u)) - div(self.sigma(u, p))
            h = fd.CellSize(self.mesh)
            tau = ((4.0*dot(u, u)/h**2) + (4.0*nu/h**2)**2)**(-0.5)
            F += tau*inner(res(u, u, p), res(u, v, s))*dx

        return F

    def mass_matrix(self, backend='petsc'):
        (u, _) = fd.TrialFunctions(self.mixed_space)
        (v, _) = fd.TestFunctions(self.mixed_space)
        M = inner(u, v)*dx

        if backend=='scipy':
            from .utils import petsc_to_scipy
            M = petsc_to_scipy(
                fd.assemble(M).petscmat
            )
        return M

    def save_mass_matrix(self, filename):
        from scipy.sparse import save_npz
        assert (fd.COMM_WORLD.size == 1), "Not supported in parallel"

        M = self.mass_matrix(backend='scipy')

        if filename[-4:] != '.npz':
            filename += '.npz'
        save_npz(filename, M)

    def linearize_dynamics(self, qB, adjoint=False):
        F = self.steady_form(q=qB)
        L = -fd.derivative(F, qB)
        if adjoint:
            from .utils.linalg import adjoint
            return adjoint(L)
        else:
            return L

    def initialize_control(self, act_idx=0):
        """Return a PETSc.Vec corresponding to the column of the control matrix"""
        pass

    def linearize(self, qB, adjoint=False, backend='petsc'):
        assert (backend in ['petsc', 'scipy']), "Backend not recognized: use `petsc` or `scipy`"
        A_form = self.linearize_dynamics(qB, adjoint=adjoint)
        M_form = self.mass_matrix()
        self.linearize_bcs()
        A = fd.assemble(A_form, bcs=self.collect_bcs()).petscmat  # Dynamics matrix
        M = fd.assemble(M_form, bcs=self.collect_bcs()).petscmat  # Mass matrix

        sys = A, M
        if backend=='scipy':
            from .utils import system_to_scipy
            sys = system_to_scipy(sys)
        return sys

    def collect_observations(self):
        pass

    def set_control(self, u=None):
        if u is None: pass

    def reset_control(self):
        pass

    def num_controls(self):
        return 0

    def dot(self, q1, q2):
        u1, _ = q1.split()
        u2, _ = q2.split()
        return fd.assemble(inner(u1, u2)*dx)

class CallbackBase:
    def __init__(self, interval: Optional[int] = 1):
        """
        Base class for things that happen every so often in the simulation
        (e.g. save output for Paraview or write some info to a log file).
        See also `utils/io.py`

        Parameters:
            interval - how often to take action
        """
        self.interval = interval

    def __call__(self, iter: int, t: float, flow: FlowConfig):
        """
        Check if this is an 'iostep' by comparing to `self.interval`
            This assumes that a child class will do something with this information
        """
        iostep = (iter % self.interval == 0)
        return iostep

    def close(self):
        pass