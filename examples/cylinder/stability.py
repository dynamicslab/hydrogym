import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

import hydrogym as gym
mesh = 'sipp-lebedev'
# mesh = 'noack'

Re = 50
flow = gym.flow.Cylinder(Re=Re, mesh_name=mesh)
qB = flow.solve_steady()

### First, direct analysis
A, M = flow.linearize(qB)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

es.getEigenpair(max_idx, vr, vi)
evec_dir = fd.Function(flow.mixed_space)
gym.utils.set_from_array(evec_dir, vr.array)

pvd = fd.File('output/evec_dir.pvd')
u, p = evec_dir.split()
vort = fd.project(fd.curl(u), flow.pressure_space)
u.rename('u')
p.rename('p')
vort.rename('vort')
pvd.write(u, p, vort)

### Adjoint
A, M = flow.linearize(qB, adjoint=True)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
max_idx = np.argmax(np.real(evals))
nconv = es.getConverged()
vr, vi = A.getVecs()
gym.print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')

es.getEigenpair(max_idx, vr, vi)
evec_adj = fd.Function(flow.mixed_space)
gym.utils.set_from_array(evec_adj, vr.array)

pvd = fd.File('output/evec_adj.pvd')
u, p = evec_adj.split()
vort = fd.project(fd.curl(u), flow.pressure_space)
u.rename('u')
p.rename('p')
vort.rename('vort')
pvd.write(u, p, vort)