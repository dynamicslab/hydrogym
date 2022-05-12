# Load Firedrake eigenmodes saved in complex mode and save PETSc data as real/imag parts
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"
assert fd.COMM_WORLD.size == 1, "Run with a single processor"

import hydrogym as gym

input_dir = 'slepc-out'
output_dir = 'output-split'
mesh = 'sipp-lebedev'

Re = 50
flow = gym.flow.Cylinder(Re=Re, mesh_name=mesh, h5_file=f'{input_dir}/steady.h5')

nconv = 20
with flow.q.dat.vec as vec:
    N = vec.size  # Length of PETSc arrays

evec = np.zeros((N, nconv), dtype=np.complex128)
with fd.CheckpointFile(f"{input_dir}/evec_dir.h5", 'r') as file:
    for idx in range(nconv):
        q = file.load_function(flow.mesh, "q", idx=idx)
        evec[:, idx] = gym.utils.get_array(q)

np.save(f"{output_dir}/evec.npy", evec)

evecH = np.zeros((N, nconv), dtype=np.complex128)
with fd.CheckpointFile(f"{input_dir}/evec_adj.h5", 'r') as file:
    for idx in range(nconv):
        q = file.load_function(flow.mesh, "q", idx=idx)
        evecH[:, idx] = gym.utils.get_array(q)

np.save(f"{output_dir}/evecH.npy", evecH)