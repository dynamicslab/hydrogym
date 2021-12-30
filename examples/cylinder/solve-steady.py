
from fenics import *
import numpy as np
import cfd_gym
from mesh.common import INLET, FREESTREAM, OUTLET, CYLINDER

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
mesh_name = 'sipp-lebedev'

if(MPI.rank(MPI.comm_world) == 0):
    cfd_gym.utils.mesh.convert_to_xdmf(f'mesh/{mesh_name}/cyl.msh', out_dir=f'mesh/{mesh_name}', dim=2)
mesh, mf = cfd_gym.utils.mesh.load_mesh(f'mesh/{mesh_name}')

Re = 40             # Reynolds number
rho = 1            # density
U_inf = Constant((1.0, 0))

# Define function space
W = FunctionSpace(mesh,
        MixedElement([
            VectorElement('CG', mesh.ufl_cell(), 2),
            FiniteElement('CG', mesh.ufl_cell(), 1)
        ])
    )

# dolfin.cpp.fem.DirichletBC(V: dolfin.cpp.function.FunctionSpace, g: dolfin.cpp.function.GenericFunction, sub_domains: dolfin.cpp.mesh.MeshFunctionSizet, sub_domain: int, method: str = ‘topological’)
bcu_inflow = DirichletBC(W.sub(0), U_inf, mf, INLET)
bcu_freestream = DirichletBC(W.sub(0), U_inf, mf, FREESTREAM)
bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), mf, CYLINDER)
bcp_outflow = DirichletBC(W.sub(1), Constant(0), mf, OUTLET)
bcs = [bcu_inflow, bcu_freestream, bcu_cylinder, bcp_outflow]

# Define trial and test functions
v, q = TestFunctions(W)
w = Function(W)
(u, p) = split(w)

# Define expressions used in variational forms
n  = FacetNormal(mesh)
nu = Constant(1/Re)
ds = ds(subdomain_data=mf)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*nu*epsilon(u) - p*Identity(len(u))

# Define variational problem for steady-state NS
F  = dot(dot(u, nabla_grad(u)), v)*dx \
   + inner(sigma(u, p), epsilon(v))*dx \
   + dot(p*n, v)*ds - dot(nu*nabla_grad(u)*n, v)*ds \
   + dot(div(u), q)*dx
J = derivative(F, w)

problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)
solver.solve()

(u_sol, p_sol) = w.split(deepcopy=True)

# Lift/drag on cylinder
force = -dot(sigma(u_sol, p_sol), n)
CL = assemble(2*force[1]*ds(CYLINDER))
CD = assemble(2*force[0]*ds(CYLINDER))

if(MPI.rank(MPI.comm_world) == 0):
    # print(f't: {t:0.02f}\t u max: {u_.vector()[:].max()}', flush=True)
    print(f'lift: {CL}\t drag: {CD}', flush=True)

file = XDMFFile("output/p_steady.xdmf")
p_sol.rename("p", "pressure")
file.write(p_sol)

file = XDMFFile("output/u_steady.xdmf")
u_sol.rename("u", "velocity")
file.write(u_sol)