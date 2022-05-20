import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from ufl import real

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

import hydrogym as gym

output_dir='output'

## First we have to ramp up the Reynolds number to get the steady state
Re_init = [500, 1000, 2000, 4000]
flow = gym.flow.Cavity(Re=Re_init[0])
gym.print(f"Steady solve at Re={Re_init[0]}")
qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

for (i, Re) in enumerate(Re_init[1:]):
    flow.Re.assign(real(Re))
    gym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

gym.print("Starting direct stability analysis")
### First, direct analysis
A, M = flow.linearize(qB)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=7.5j)
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

np.save('output/evals.npy', evals)

### Adjoint
A, M = flow.linearize(qB, adjoint=True)
evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=7.5j)
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