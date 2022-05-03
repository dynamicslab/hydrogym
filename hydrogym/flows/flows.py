import numpy as np
import firedrake as fd
from firedrake import dx, ds
from firedrake.petsc import PETSc
from ufl import sym, grad, dot, inner, nabla_grad, div

import hydrogym

class Flow:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n = fd.FacetNormal(self.mesh)

        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(mesh, 'CG', 2)
        self.pressure_space = fd.FunctionSpace(mesh, 'CG', 1)
        self.mixed_space = fd.MixedFunctionSpace([self.velocity_space, self.pressure_space])
        
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
        w = fd.Function(self.mixed_space)

        F = self.steady_form(w, fd.TestFunctions(self.mixed_space))  # Nonlinear variational form
        J = fd.derivative(F, w)    # Jacobian

        bcs = self.collect_bcs()
        problem = fd.NonlinearVariationalProblem(F, w, bcs, J)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()

        return w


class Cylinder(Flow):
    from .cylinder.mesh.common import INLET, FREESTREAM, OUTLET, CYLINDER

    def __init__(self, mesh_name='noack'):
        mesh_root = f'{hydrogym.install_dir}/flows/cylinder/mesh'
        mesh = fd.Mesh(f'{mesh_root}/{mesh_name}/cyl.msh')

        self.Re = fd.Constant(100)
        self.U_inf = fd.Constant((1.0, 0.0))
        super().__init__(mesh)

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(V, self.U_inf, self.FREESTREAM)
        self.bcu_cylinder = fd.DirichletBC(V, fd.Constant((0, 0)), self.CYLINDER)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def steady_form(self, w, w_test):
        (u, p) = fd.split(w)
        (v, q) = w_test
        nu = fd.Constant(1/self.Re)

        F  = dot(dot(u, nabla_grad(u)), v)*dx \
            + inner(self.sigma(u, p), self.epsilon(v))*dx \
            + dot(p*self.n, v)*ds - inner(nu*nabla_grad(u)*self.n, v)*ds \
            + dot(div(u), q)*dx
        return F

    def compute_forces(self, u, p):
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = fd.assemble(2*force[1]*ds(self.CYLINDER))
        CD = fd.assemble(2*force[0]*ds(self.CYLINDER))
        return CL, CD

    def integrate(self, dt=1e-3, Tf=0.01, output_dir=None):
        num_steps = int(Tf//dt)
        k = fd.Constant(dt)
        nu = fd.Constant(1/self.Re)

        self.init_bcs()
        V, Q = self.function_spaces()

        # Boundary conditions
        bcu = self.collect_bcu()
        bcp = self.collect_bcp()

        u = fd.TrialFunction(V)
        p = fd.TrialFunction(Q)
        v = fd.TestFunction(V)
        q = fd.TestFunction(Q)
        u_n = fd.Function(V, name='u')
        p_n = fd.Function(Q, name='p')
        u_ = fd.Function(V)
        p_ = fd.Function(Q)

        U = 0.5*(u_n + u)  # Average for semi-implicit
        u_t = (u - u_n)/k  # Time derivative

        # Velocity predictor
        F1 = dot(u_t, v)*dx \
            + dot(dot(u_n, nabla_grad(u_n)), v)*dx \
            + inner(self.sigma(U, p_n), self.epsilon(v))*dx \
            + dot(p_n*self.n, v)*ds - dot(nu*nabla_grad(U)*self.n, v)*ds
            # - dot(f, v)*self.dx
        a1 = fd.lhs(F1)
        L1 = fd.rhs(F1)

        # Poisson equation
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

        # Projection step (pressure correction)
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

        # Assemble matrices
        A1 = fd.assemble(a1, bcs=bcu)
        A2 = fd.assemble(a2, bcs=bcp)
        A3 = fd.assemble(a3)

        # # Apply boundary conditions to matrices
        # [bc.apply(A1) for bc in bcu]
        # [bc.apply(A2) for bc in bcp]

        # Create XDMF files for visualization output
        if output_dir is not None:
            pvd_out = fd.File(f"{output_dir}/solution.pvd")

        t = 0
        for n in range(num_steps):
            t += dt  # Update current time

            # Step 1: Tentative velocity step
            b1 = fd.assemble(L1, bcs=bcu)
            fd.solve(A1, u_.vector(), b1, solver_parameters={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg"
            })

            # Step 2: Pressure correction step
            b2 = fd.assemble(L2, bcs=bcp)
            fd.solve(A2, p_.vector(), b2, solver_parameters={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg"
            })

            # Step 3: Velocity correction step
            b3 = fd.assemble(L3)
            fd.solve(A3, u_.vector(), b3, solver_parameters={
                "ksp_type": "cg",
                "pc_type": "sor"
            })

            # Update previous solution
            u_n.assign(u_)
            p_n.assign(p_)

            if output_dir is not None:
                # Save solution to file (XDMF/HDF5)
                pvd_out.write(u_n, p_n, time=t)
                # xdmffile_u.write(u_, t)
                # xdmffile_p.write(p_, t)

            CL, CD = self.compute_forces(u_n, p_n)
            print(f't:{t:08f}\t\t CL:{CL:08f} \t\tCD::{CD:08f}')