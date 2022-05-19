from common import *
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

qB = flow.solve_steady(solver_parameters={'snes_monitor': None})
flow.save_checkpoint(f'{temp_dir}/steady.h5')

# Also save as arrays for conversion to real mode
qB = flow.q.copy(deepcopy=True)
with qB.dat.vec_ro as vec:
    np.save(f'{temp_dir}/steady_{fd.COMM_WORLD.rank}.npy', np.real(vec.array))

### Direct stability analysis
gym.print("Direct analysis...")
A, M = flow.linearize(qB)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

es.getEigenpair(max_idx, vr, vi)

gym.utils.set_from_array(flow.q, vr.array)
pvd = fd.File(f'{temp_dir}/evec_dir.pvd')
vort = flow.vorticity()
pvd.write(flow.u, flow.p, vort)

for idx in range(nconv):
    es.getEigenpair(idx, vr, vi)
    np.save(f'{temp_dir}/direct{idx}_{fd.COMM_WORLD.rank}.npy', vr.array)

### Adjoint
gym.print("Adjoint analysis...")
A, M = flow.linearize(qB, adjoint=True)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

es.getEigenpair(max_idx, vr, vi)
gym.utils.set_from_array(flow.q, vr.array)

pvd = fd.File(f'{temp_dir}/evec_adj.pvd')
vort = flow.vorticity()
pvd.write(flow.u, flow.p, vort)

for idx in range(nconv):
    es.getEigenpair(idx, vr, vi)
    rank = fd.COMM_WORLD.rank
    np.save(f'{temp_dir}/adjoint{idx}_{fd.COMM_WORLD.rank}.npy', vr.array)