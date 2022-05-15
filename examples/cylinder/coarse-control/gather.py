# The output of stability analysis is split by processor and saved in (complex-valued) numpy binaries
# This script just needs to load them (using the SAME number of processors) and reconstruct the
# Function in real mode
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
import ufl

import numpy as np
assert PETSc.ScalarType == np.float64, "Run in real mode"

import hydrogym as gym

input_dir = 'tmp'
output_dir = 'global-modes'
mesh = 'noack'

Re = 50
flow = gym.flow.Cylinder(Re=Re, mesh_name=mesh)

# First load base flow
with flow.q.dat.vec as vec:
    vec.array = np.load(f'{input_dir}/steady_{fd.COMM_WORLD.rank}.npy')
flow.save_checkpoint(f"{output_dir}/steady.h5")

nconv = 20

real_out = fd.CheckpointFile(f"{output_dir}/direct_real.h5", 'w')
imag_out = fd.CheckpointFile(f"{output_dir}/direct_imag.h5", 'w')
for idx in range(nconv):
    q_arr = np.load(f'{input_dir}/direct{idx}_{fd.COMM_WORLD.rank}.npy')
    gym.utils.set_from_array(flow.q, np.real(q_arr))
    real_out.save_function(flow.q, idx=idx)

    gym.utils.set_from_array(flow.q, np.imag(q_arr))
    imag_out.save_function(flow.q, idx=idx)
real_out.close()
imag_out.close()

real_out = fd.CheckpointFile(f"{output_dir}/adjoint_real.h5", 'w')
imag_out = fd.CheckpointFile(f"{output_dir}/adjoint_imag.h5", 'w')
for idx in range(nconv):
    q_arr = np.load(f'{input_dir}/adjoint{idx}_{fd.COMM_WORLD.rank}.npy')
    gym.utils.set_from_array(flow.q, np.real(q_arr))
    real_out.save_function(flow.q, idx=idx)

    gym.utils.set_from_array(flow.q, np.imag(q_arr))
    imag_out.save_function(flow.q, idx=idx)
real_out.close()
imag_out.close()