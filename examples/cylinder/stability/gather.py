# The output of stability analysis is split by processor and saved in (complex-valued) numpy binaries
# This script just needs to load them (using the SAME number of processors) and reconstruct the
# Function in real mode
from common import *
assert PETSc.ScalarType == np.float64, "Run in real mode"

# First load base flow
with flow.q.dat.vec as vec:
    vec.array = np.load(f'{temp_dir}/steady_{fd.COMM_WORLD.rank}.npy')
flow.save_checkpoint(f"{output_dir}/steady.h5")

nconv = 20

real_out = fd.CheckpointFile(f"{output_dir}/direct_real.h5", 'w')
imag_out = fd.CheckpointFile(f"{output_dir}/direct_imag.h5", 'w')
for idx in range(nconv):
    q_arr = np.load(f'{temp_dir}/direct{idx}_{fd.COMM_WORLD.rank}.npy')
    gym.utils.set_from_array(flow.q, np.real(q_arr))
    real_out.save_function(flow.q, idx=idx)

    gym.utils.set_from_array(flow.q, np.imag(q_arr))
    imag_out.save_function(flow.q, idx=idx)
real_out.close()
imag_out.close()

real_out = fd.CheckpointFile(f"{output_dir}/adjoint_real.h5", 'w')
imag_out = fd.CheckpointFile(f"{output_dir}/adjoint_imag.h5", 'w')
for idx in range(nconv):
    q_arr = np.load(f'{temp_dir}/adjoint{idx}_{fd.COMM_WORLD.rank}.npy')
    gym.utils.set_from_array(flow.q, np.real(q_arr))
    real_out.save_function(flow.q, idx=idx)

    gym.utils.set_from_array(flow.q, np.imag(q_arr))
    imag_out.save_function(flow.q, idx=idx)
real_out.close()
imag_out.close()