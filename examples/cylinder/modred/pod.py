from common import *

from firedrake import logging
logging.set_level(logging.DEBUG)

r = 8
coeffs, mode_handles = gym.linalg.pod(flow,
    snapshot_prefix = f'{output_dir}/snapshots/',
    decomp_indices = range(100),
    mass_matrix = f'{output_dir}/mass_matrix',
    output_dir = output_dir,
    remove_mean = True,
    r = r
)

# Check orthonormality
from scipy import sparse
M = sparse.load_npz(f'{output_dir}/mass_matrix.npz')
q1 = fd.Function(flow.mixed_space)
q2 = fd.Function(flow.mixed_space)
for i in range(r):
    for j in range(r):
        q1_vec = mode_handles[i].get()
        q2_vec = mode_handles[j].get()
        print(i, j, np.dot(q1_vec.conj(), M @ q2_vec))
        gym.utils.set_from_array(q1, q1_vec)
        gym.utils.set_from_array(q2, q2_vec)
        print(i, j, flow.dot(q1, q2))
