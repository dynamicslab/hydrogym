import firedrake as fd
import numpy as np
from common import flow, output_dir, snap_prefix, transient_prefix
from firedrake import logging
from scipy import sparse

import hydrogym as gym

logging.set_level(logging.DEBUG)

r = 8
m = 100  # Number of snapshots for POD
mass_matrix = f"{output_dir}/mass_matrix"
coeffs, basis_handles = gym.linalg.pod(
    flow,
    snapshot_handles=[
        gym.linalg.Snapshot(f"{snap_prefix}{idx}.npy") for idx in range(m)
    ],
    mass_matrix=mass_matrix,
    output_dir=output_dir,
    remove_mean=True,
    r=r,
)

# Project onto the transient time series
m = 3000
mean_handle = gym.linalg.Snapshot(f"{output_dir}/mean.npy")
data_handles = [
    gym.linalg.Snapshot(f"{transient_prefix}{idx}.npy", base_vec_handle=mean_handle)
    for idx in range(m)
]
coeffs = gym.linalg.project(basis_handles, data_handles, mass_matrix)
np.savetxt(f"{output_dir}/coeffs.dat", coeffs)


M = sparse.load_npz(f"{output_dir}/mass_matrix.npz")
q1 = fd.Function(flow.mixed_space)
q2 = fd.Function(flow.mixed_space)
for i in range(r):
    q1_vec = basis_handles[i].get()
    for j in range(r):
        q2_vec = basis_handles[j].get()
        print(i, j, np.dot(q1_vec.conj(), M @ q2_vec))
        gym.utils.set_from_array(q1, q1_vec)
        gym.utils.set_from_array(q2, q2_vec)
        print(i, j, flow.dot(q1, q2))
