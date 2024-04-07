"""Compute the BPOD modes of the cylinder wake flow."""

import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
from scipy import linalg

Re = 100
output_dir = f"./re{Re}_med_bpod_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

eig_dir = f"./re{Re}_med_eig_output"
flow = hgym.RotaryCylinder(
    Re=100,
    velocity_order=2,
    restart=f"{eig_dir}/base.h5"
)

qB = flow.q.copy(deepcopy=True)


#
# Load impulse response data
#
snapshot_dir = "re100_impulse_output"

# Direct impulse response solution
X = []
m_d = 100  # Number of direct snapshots
with fd.CheckpointFile(f"{snapshot_dir}/dir_snapshots.h5", "r") as chk:
    for i in range(m_d):
        q = chk.load_function(flow.mesh, f"q_{i}")
        X.append(q)

# Adjoint impulse response solution
Y = []
m_a = 100  # Number of adjoint snapshots
with fd.CheckpointFile(f"{snapshot_dir}/adj_snapshots.h5", "r") as chk:
    for i in range(m_a):
        q = chk.load_function(flow.mesh, f"q_{i}")
        Y.append(q)

#
# BPOD: Method of snapshots
#
R = np.zeros((m_a, m_a))  # Correlation matrix
for i in range(m_a):
    for j in range(m_d):
        R[i, j] = flow.inner_product(X[j], Y[i]).real


U, S, T = linalg.svd(R)

V_bpod = []   # direct modes: X @ T.T @ S ** (-1/2)
W_bpod = []   # adjoint modes: Y @ U @ S ** (-1/2)

r = 64  # Number of BPOD modes to compute for stable subspace
for i in range(r):
    psi = fd.Function(flow.mixed_space)
    for j in range(m_d):
        psi.assign(psi + X[j] * T[i, j] / np.sqrt(S[i]))
    V_bpod.append(psi)

    psi = fd.Function(flow.mixed_space)
    for j in range(m_a):
        psi.assign(psi + Y[j] * U[j, i] / np.sqrt(S[i]))
    W_bpod.append(psi)


#
# Save the BPOD modes
#
with fd.CheckpointFile(f"{output_dir}/bpod_modes.h5", "r") as chk:
    chk.save_mesh(flow.mesh)
    for i, v in enumerate(V_bpod):
        v.rename(f"v_{i}")
        chk.save_function(v)

    for i, w in enumerate(W_bpod):
        w.rename(f"w_{i}")
        chk.save_function(w)
