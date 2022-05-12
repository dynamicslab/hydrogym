import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

import hydrogym as gym
output_dir = 'slepc-out'
mesh = 'sipp-lebedev'
# mesh = 'noack'

Re = 50
flow = gym.flow.Cylinder(Re=Re, mesh_name=mesh)
qB = flow.solve_steady()
flow.save_checkpoint(f'{output_dir}/steady.h5')

### Direct stability analysis
gym.print("Direct analysis...")
A, M = flow.linearize(qB)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

evec_dir = fd.Function(flow.mixed_space, name='q')
es.getEigenpair(max_idx, vr, vi)
gym.utils.set_from_array(evec_dir, vr.array)

pvd = fd.File(f'{output_dir}/evec_dir.pvd')
u, p = evec_dir.split()
vort = fd.project(fd.curl(u), flow.pressure_space)
u.rename('u')
p.rename('p')
vort.rename('vort')
pvd.write(u, p, vort)

with fd.CheckpointFile(f'{output_dir}/evec_dir.h5', 'w') as file:
    for idx in range(nconv):
        es.getEigenpair(idx, vr, vi)
        gym.utils.set_from_array(evec_dir, vr.array)
        file.save_function(evec_dir, idx=idx)


### Adjoint
gym.print("Adjoint analysis...")
A, M = flow.linearize(qB, adjoint=True)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

es.getEigenpair(max_idx, vr, vi)
evec_adj = fd.Function(flow.mixed_space, name='q')
gym.utils.set_from_array(evec_adj, vr.array)

pvd = fd.File(f'{output_dir}/evec_adj.pvd')
u, p = evec_adj.split()
vort = fd.project(fd.curl(u), flow.pressure_space)
u.rename('u')
p.rename('p')
vort.rename('vort')
pvd.write(u, p, vort)

with fd.CheckpointFile(f'{output_dir}/evec_adj.h5', 'w') as file:
    for idx in range(nconv):
        es.getEigenpair(idx, vr, vi)
        gym.utils.set_from_array(evec_adj, vr.array)
        file.save_function(evec_adj, idx=idx)