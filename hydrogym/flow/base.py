import firedrake as fd

# import pyadjoint
import numpy as np
import ufl
from firedrake import ds, dx

# from firedrake import logging
from ufl import curl, div, dot, inner, nabla_grad, sym

from ..core import PDEModel


class FlowConfigBase(PDEModel):
    ACT_DIM = 1
    DEFAULT_MESH = ""
    DEFAULT_REYNOLDS = 1

    def __init__(self, Re=None, restart=None, mesh=None):
        load_mesh = self.get_mesh_loader()
        mesh = load_mesh(name=mesh or self.DEFAULT_MESH)
        super().__init__(mesh, restart=restart)
        self.Re = fd.Constant(ufl.real(Re or self.DEFAULT_REYNOLDS))

    def get_mesh_loader(self):
        """Return a function load_mesh(name) that will load a fd.Mesh"""
        return None

    def initialize_state(self):
        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(self.mesh, "CG", 2)
        self.pressure_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.mixed_space = fd.MixedFunctionSpace(
            [self.velocity_space, self.pressure_space]
        )
        self.q = fd.Function(self.mixed_space, name="q")
        self.split_solution()  # Break out and rename solution

    @property
    def nu(self):
        return fd.Constant(1 / ufl.real(self.Re))

    def split_solution(self):
        self.u, self.p = self.q.split()
        self.u.rename("u")
        self.p.rename("p")

    def vorticity(self, u=None):
        if u is None:
            u = self.u
        vort = fd.project(curl(u), self.pressure_space)
        vort.rename("vort")
        return vort

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

    # Symmetric gradient
    def epsilon(self, u):
        return sym(nabla_grad(u))

    # Stress tensor
    def sigma(self, u, p):
        return 2 * self.nu * self.epsilon(u) - p * fd.Identity(len(u))

    @property
    def body_force(self):
        return fd.interpolate(fd.Constant((0.0, 0.0)), self.velocity_space)

    def solve_steady(self, solver_parameters={}, stabilization=None):
        self.init_bcs(mixed=True)

        F = self.steady_form(stabilization=stabilization)  # Nonlinear variational form
        J = fd.derivative(F, self.q)  # Jacobian

        bcs = self.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, self.q, bcs, J)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=solver_parameters
        )
        solver.solve()

        return self.q.copy(deepcopy=True)

    def steady_form(self, q=None, stabilization=None):
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        (v, s) = fd.TestFunctions(self.mixed_space)

        F = (
            inner(dot(u, nabla_grad(u)), v) * dx
            + inner(self.sigma(u, p), self.epsilon(v)) * dx
            + inner(p * self.n, v) * ds
            - inner(self.nu * nabla_grad(u) * self.n, v) * ds
            + inner(div(u), s) * dx
        )

        # if stabilization == "gls":
        #     # Galerkin least-squares stabilization (see Tezduyar, 1991)
        #     def res(U, u, p):
        #         return dot(U, nabla_grad(u)) - div(self.sigma(u, p))

        #     h = fd.CellSize(self.mesh)
        #     F += tau * inner(res(u, u, p), res(u, v, s)) * dx

        return F

    def mass_matrix(self, backend="petsc"):
        (u, _) = fd.TrialFunctions(self.mixed_space)
        (v, _) = fd.TestFunctions(self.mixed_space)
        M = inner(u, v) * dx

        if backend == "scipy":
            from ..utils import petsc_to_scipy

            M = petsc_to_scipy(fd.assemble(M).petscmat)
        return M

    def save_mass_matrix(self, filename):
        from scipy.sparse import save_npz

        assert fd.COMM_WORLD.size == 1, "Not supported in parallel"

        M = self.mass_matrix(backend="scipy")

        if filename[-4:] != ".npz":
            filename += ".npz"
        save_npz(filename, M)

    def linearize_dynamics(self, qB, adjoint=False):
        F = self.steady_form(q=qB)
        L = -fd.derivative(F, qB)
        if adjoint:
            from ..utils.linalg import adjoint

            return adjoint(L)
        else:
            return L

    def set_control(self, control=None):
        """
        Sets the actuation

        Note that for time-varying controls it will be better to adjust the rotation rate
        in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
        to change control for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        if control is None:
            control = np.zeros(self.ACT_DIM)
        self.control = self.enlist_controls(control)

        if hasattr(self, "bcu_actuation"):
            for i in range(self.ACT_DIM):
                c = fd.Constant(self.control[i])
                self.bcu_actuation[i]._function_arg.assign(
                    fd.interpolate(
                        c * self.u_ctrl[i], self.velocity_space
                    )
                ) 

    def control_vec(self, mixed=False):
        """Return a list of PETSc.Vecs corresponding to the columns of the control matrix"""
        (v, _) = fd.TestFunctions(self.mixed_space)
        self.linearize_bcs()
        # self.linearize_bcs() should have reset control, need to perturb it now

        fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())

        B = []
        for i in range(self.ACT_DIM):
            c = np.zeros(self.ACT_DIM)
            c[i] = 1.0  # Perturb the ith control
            self.set_control(c)

            # Control as fd.Function
            B.append(
                fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())
            )

            # Have to have mixed function space for computing B functions
            self.reset_control(mixed=True)  

        # At the end the BC function spaces could be mixed or not
        self.reset_control(mixed=mixed)
        return B

    def num_controls(self):
        return self.ACT_DIM

    def linearize(self, qB, adjoint=False, backend="petsc"):
        assert backend in [
            "petsc",
            "scipy",
        ], "Backend not recognized: use `petsc` or `scipy`"
        A_form = self.linearize_dynamics(qB, adjoint=adjoint)
        M_form = self.mass_matrix()
        self.linearize_bcs()
        A = fd.assemble(A_form, bcs=self.collect_bcs()).petscmat  # Dynamics matrix
        M = fd.assemble(M_form, bcs=self.collect_bcs()).petscmat  # Mass matrix

        sys = A, M
        if backend == "scipy":
            from ..utils import system_to_scipy

            sys = system_to_scipy(sys)
        return sys

    def get_observations(self):
        pass

    def evaluate_objective(self, q=None):
        pass

    def dot(self, q1, q2):
        u1, _ = q1.split()
        u2, _ = q2.split()
        return fd.assemble(inner(u1, u2) * dx)
