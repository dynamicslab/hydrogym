from functools import wraps
import numpy as np
from fenics import *
import hydrogym

class Flow:
    def __init__(self, mesh, mf):
        self.mesh = mesh
        self.mf = mf
        self.dx = dx
        self.ds = ds(subdomain_data=mf)  # Includes boundary labels
        self.n = FacetNormal(self.mesh)

        # Set up Taylor-Hood elements
        self.mixed_space = FunctionSpace(mesh,
                MixedElement([
                    VectorElement('CG', mesh.ufl_cell(), 2),
                    FiniteElement('CG', mesh.ufl_cell(), 1)
                ])
            )

        self.init_bcs()
        
    def init_bcs(self):
        """Define all boundary conditions"""
        pass

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
        return 2*(1/self.Re)*self.epsilon(u) - p*Identity(len(u))

    def steady_form(self, w, w_test):
        """Define nonlinear variational problem for steady-state NS"""
        pass

    def solve_steady(self):
        W = self.mixed_space()
        w_test = TestFunctions(W)
        w = Function(W)

        F = self.steady_form(w, TestFunctions(W))  # Nonlinear variational form
        w = w
        J = derivative(F, w)    # Jacobian

        bcs = self.collect_bcs()
        problem = NonlinearVariationalProblem(F, w, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.solve()

        return w


class Cylinder(Flow):
    from .cylinder.mesh.common import INLET, FREESTREAM, OUTLET, CYLINDER

    def __init__(self, mesh_name='noack'):
        mesh_root = f'{hydrogym.install_dir}/flows/cylinder/mesh'

        # TODO: Check for existence before converting
        if(MPI.rank(MPI.comm_world) == 0):
            hydrogym.utils.mesh.convert_to_xdmf(f'{mesh_root}/{mesh_name}/cyl.msh', out_dir=f'{mesh_root}/{mesh_name}', dim=2)
        mesh, mf = hydrogym.utils.mesh.load_mesh(f'{mesh_root}/{mesh_name}')

        self.Re = Constant(100)
        self.U_inf = Constant((1.0, 0.0))
        super().__init__(mesh, mf)

    def init_bcs(self):
        self.bcu_inflow = DirichletBC(self.function_space.sub(0), self.U_inf, self.mf, self.INLET)
        self.bcu_freestream = DirichletBC(self.function_space.sub(0), self.U_inf, self.mf, self.FREESTREAM)
        self.bcu_cylinder = DirichletBC(self.function_space.sub(0), Constant((0, 0)), self.mf, self.CYLINDER)
        self.bcp_outflow = DirichletBC(self.function_space.sub(1), Constant(0), self.mf, self.OUTLET)

    def collect_bcu(self):
        return [self.bcu_inflow, self.bcu_freestream, self.bcu_cylinder]
    
    def collect_bcp(self):
        return [self.bcp_outflow]

    def steady_form(self):
        (u, p) = split(self.sol)
        (v, q) = self.test
        nu = Constant(1/self.Re)

        F  = dot(dot(u, nabla_grad(u)), v)*self.dx \
            + inner(self.sigma(u, p), self.epsilon(v))*self.dx \
            + dot(p*self.n, v)*self.ds - inner(nu*nabla_grad(u)*self.n, v)*self.ds \
            + dot(div(u), q)*self.dx
        return F

    def compute_forces(self, u, p):
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = assemble(2*force[1]*self.ds(self.CYLINDER))
        CD = assemble(2*force[0]*self.ds(self.CYLINDER))
        return CL, CD

    def integrate(self, dt=1e-3, Tf=0.01, output_dir=None):
        num_steps = int(Tf//dt)
        k = Constant(dt)
        nu = Constant(1/self.Re)


        V = VectorFunctionSpace(self.mesh, 'P', 2)
        Q = FunctionSpace(self.mesh, 'P', 1)

        self.bcu_inflow = DirichletBC(V, self.U_inf, self.mf, self.INLET)
        self.bcu_freestream = DirichletBC(V, self.U_inf, self.mf, self.FREESTREAM)
        self.bcu_cylinder = DirichletBC(V, Constant((0, 0)), self.mf, self.CYLINDER)
        self.bcp_outflow = DirichletBC(Q, Constant(0), self.mf, self.OUTLET)

        # Boundary conditions
        bcu = self.collect_bcu()
        bcp = self.collect_bcp()

        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)
        u_n = Function(V)
        p_n = Function(Q)
        u_ = Function(V)
        p_ = Function(Q)

        U = 0.5*(u_n + u)  # Average for semi-implicit
        u_t = (u - u_n)/k  # Time derivative

        # Velocity predictor
        F1 = dot(u_t, v)*self.dx \
            + dot(dot(u_n, nabla_grad(u_n)), v)*self.dx \
            + inner(self.sigma(U, p_n), self.epsilon(v))*self.dx \
            + dot(p_n*self.n, v)*self.ds - dot(nu*nabla_grad(U)*self.n, v)*self.ds
            # - dot(f, v)*self.dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Poisson equation
        a2 = dot(nabla_grad(p), nabla_grad(q))*self.dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*self.dx - (1/k)*div(u_)*q*self.dx

        # Projection step (pressure correction)
        a3 = dot(u, v)*self.dx
        L3 = dot(u_, v)*self.dx - dt*dot(nabla_grad(p_ - p_n), v)*self.dx

        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        # Apply boundary conditions to matrices
        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]

        # Create XDMF files for visualization output
        if output_dir is not None:
            pvd_out = File(f"{output_dir}/u.pvd")
            # xdmffile_u = XDMFFile(f'{output_dir}/velocity.xdmf')
            # xdmffile_p = XDMFFile(f'{output_dir}/pressure.xdmf')

        t = 0
        for n in range(num_steps):
            t += dt  # Update current time

            # Step 1: Tentative velocity step
            b1 = assemble(L1)
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

            # Step 2: Pressure correction step
            b2 = assemble(L2)
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg', 'sor')

            # Update previous solution
            u_n.assign(u_)
            p_n.assign(p_)

            if output_dir is not None:
                # Save solution to file (XDMF/HDF5)
                pvd_out.write(u_)
                # xdmffile_u.write(u_, t)
                # xdmffile_p.write(p_, t)

            CL, CD = self.compute_forces(u_n, p_n)
            print(f't:{t:08f}\t\t CL:{CL:08f} \t\tCD::{CD:08f}')