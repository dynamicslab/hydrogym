# Load snapshots as firedrake Functions and save as numpy arrays in individual files
# Idea is to avoid PETSc's parallelization in favor of modred's for modal analysis
from common import *
assert fd.COMM_WORLD.size == 1, "Run in serial"

r = 8

# flow.load_checkpoint(restart)
with fd.CheckpointFile(pod_file, 'w') as file:
    file.save_mesh(flow.mesh)
    gym.utils.set_from_array(flow.q, np.load(f'{output_dir}/mean.npy'))
    file.save_function(flow.q, idx=0)
    for idx in range(r):
        gym.utils.set_from_array(flow.q, np.load(f'{pod_prefix}{idx}.npy'))
        print(flow.dot(flow.q, flow.q))
        file.save_function(flow.q, idx=idx+1)


# # Check orthonormality
# from scipy import sparse
# M = sparse.load_npz(f'{output_dir}/mass_matrix.npz')
# q1 = fd.Function(flow.mixed_space)
# q2 = fd.Function(flow.mixed_space)
# for i in range(r):
#     for j in range(r):
#         flow.load_checkpoint(snap)
#         print(i, j, flow.dot(q1, q2))
