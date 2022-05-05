import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np
assert(PETSc.ScalarType == np.complex128,
        "Complex PETSc configuration required for stability analysis")
        
import hydrogym as gym

output_dir = 'output'
cyl = gym.flow.Cylinder()
cyl.solve_steady()
Q0 = cyl.q.copy(deepcopy=True)

from firedrake import inner, dx, Constant
(u, p) = fd.TrialFunctions(cyl.mixed_space)
(v, q) = fd.TestFunctions(cyl.mixed_space)
F = cyl.steady_form(Q0, (v, q))
J = fd.derivative(-F, Q0)
M = inner(u, v)*dx

cyl.linearize_bcs()
B = fd.assemble(M, bcs=cyl.collect_bcs()).petscmat
A = fd.assemble(J, bcs=cyl.collect_bcs()).petscmat

### SLEPc
opts = PETSc.Options()
opts.setValue("eps_monitor_all", None)
opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("eps_target", "0.15+0.86i") 
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_largest_real", True)
opts.setValue("st_type", "sinvert")
# opts.setValue("st_ksp_type", "gmres")
# opts.setValue("st_ksp_monitor_true_residual", None)

# opts.setValue("st_pc_type", "bjacobi")
opts.setValue("st_pc_factor_shift_type", "NONZERO")
# opts.setValue("st_ksp_view", None)
# opts.setValue("eps_tol", 1e-10)

num_eigenvalues = 8
es = SLEPc.EPS().create(comm=fd.COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(A, B)
es.setFromOptions()
es.solve()

nconv = es.getConverged()
print(nconv)
vr, vi = A.getVecs()

lam = es.getEigenpair(0, vr, vi)
print(lam)
for i in range(nconv):
    print(es.getEigenpair(i, vr, vi))
