import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl

import hydrogym as gym

output_dir = 'output'
flow = gym.flow.Cylinder()
qB = flow.solve_steady()
solver = gym.ts.IPCS(flow, dt=1e-2)

# M, A, B = flow.linearize(qB, control=True, backend='scipy')
# if fd.COMM_WORLD.rank == 0:
#     import scipy.io as sio
#     sio.savemat(f'{output_dir}/lti.mat', {'M': M, 'A': A, 'B': B})