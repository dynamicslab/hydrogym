import firedrake as fd
import numpy as np
from common import Re, flow, temp_dir
from firedrake.petsc import PETSc
from ufl import real

import hydrogym as hgym

assert (
    PETSc.ScalarType == np.complex128
), "Complex PETSc configuration required for stability analysis"

Re_init = [500, 1000, 2000, 4000, Re]

for (i, R) in enumerate(Re_init):
    flow.Re.assign(real(R))
    hgym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

flow.save_checkpoint(f"{temp_dir}/steady.h5")

# Also save as arrays for conversion to real mode
qB = flow.q.copy(deepcopy=True)
with qB.dat.vec_ro as vec:
    np.save(f"{temp_dir}/steady_{fd.COMM_WORLD.rank}.npy", np.real(vec.array))

# Direct stability analysis
hgym.print("Direct analysis...")
A, M = flow.linearize(qB)
sigma = np.arange(0, 25, 0.5)
all_evals = np.array([])
evals, es = hgym.linalg.eig(A, M, num_eigenvalues=6, sigma=7.5j)
nconv = es.getConverged()
max_idx = np.argmax(np.real(evals))
vr, vi = A.getVecs()
hgym.print(f"Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}")
hgym.print(evals)
hgym.print(evals.shape)
sorted_idx = np.argsort(np.real(evals))
np.savetxt("global-modes/evals.dat", evals[sorted_idx], fmt="%.6f", delimiter="\t")

es.getEigenpair(max_idx, vr, vi)

hgym.utils.set_from_array(flow.q, vr.array)
pvd = fd.File(f"{temp_dir}/evec_dir.pvd")
vort = flow.vorticity()
pvd.write(flow.u, flow.p, vort)

for idx in range(nconv):
    es.getEigenpair(idx, vr, vi)
    np.save(f"{temp_dir}/direct{idx}_{fd.COMM_WORLD.rank}.npy", vr.array)

# Adjoint
hgym.print("Adjoint analysis...")
A, M = flow.linearize(qB, adjoint=True)
evals, es = hgym.linalg.eig(A, M, num_eigenvalues=6, sigma=7.5j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
hgym.print(f"Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}")

es.getEigenpair(max_idx, vr, vi)
hgym.utils.set_from_array(flow.q, vr.array)

pvd = fd.File(f"{temp_dir}/evec_adj.pvd")
vort = flow.vorticity()
pvd.write(flow.u, flow.p, vort)

for idx in range(nconv):
    es.getEigenpair(idx, vr, vi)
    rank = fd.COMM_WORLD.rank
    np.save(f"{temp_dir}/adjoint{idx}_{fd.COMM_WORLD.rank}.npy", vr.array)
