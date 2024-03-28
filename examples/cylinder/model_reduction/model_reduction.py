"""Model reduction using unstable modes and BPOD for the stable subspace"""
import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
from scipy import linalg

from lti_system import control_vec, measurement_matrix
from step_response import LinearBDFSolver

Re = 100
eig_dir = f"./re{Re}_med_eig_output"
snapshot_dir = f"./re{Re}_impulse_output"
output_dir = f"./re{Re}_rom"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

flow = hgym.RotaryCylinder(
    Re=Re,
    velocity_order=2,
    restart=f"{eig_dir}/base.h5"
)

qB = flow.q.copy(deepcopy=True)

# 2. Derive flow field associated with actuation BC
# See Barbagallo et al. (2009) for details on the "lifting" procedure
qC = control_vec(flow)

# 3. Derive the "measurement matrix"
# This is a field qM such that the inner product of qM with the flow field
# produces the same result as computing the observation (lift coefficient)
qM = measurement_matrix(flow)

#
# Unstable global modes
#
evals = np.load(f"{eig_dir}/evals.npy")

# Load the set of eigenvectors
Vu = []
with fd.CheckpointFile(f"{eig_dir}/evecs.h5", "r") as chk:
    for i in range(len(evals)):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        Vu.append(q)

Wu = []
with fd.CheckpointFile(f"{eig_dir}/adj_evecs.h5", "r") as chk:
    for i in range(len(evals)):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        Wu.append(q)

# Sort by descending real part
sort_idx = np.argsort(-evals.real)
evals = evals[sort_idx]

Vu = [Vu[i] for i in sort_idx]
Wu = [Wu[i] for i in sort_idx]

# Keep only the unstable modes
unstable_idx = np.where(evals.real > 0)[0]
Vu = [Vu[i] for i in unstable_idx]
Wu = [Wu[i] for i in unstable_idx]

ru = len(unstable_idx)  # Number of unstable modes

#
# Load impulse response data
#

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
# Balanced proper orthogonal decomposition
#

# Correlation matrix with the method of snapshots
R = np.zeros((m_a, m_a))
for i in range(m_a):
    for j in range(m_d):
        R[i, j] = flow.inner_product(X[j], Y[i]).real

# Singular value decomposition of the correlation matrix
U, S, T = linalg.svd(R)

# Print the Hankel singular values
print(f"Hankel singular values: {S[:20]}")

# Upper bound on the H-inf error
Hinf_err = np.zeros_like(S)
for i in range(len(S)):
    Hinf_err[i] = 2 * sum(S[i:])

print(f"Upper bound on H-inf error: {Hinf_err[:20]}")

# Truncation threshold based on H-inf error
threshold = 1e-2

Vs = []   # direct modes: X @ T.T @ S ** (-1/2)
Ws = []   # adjoint modes: Y @ U @ S ** (-1/2)

rs = 6  # Number of BPOD modes for stable subspace
for i in range(rs):
    q = fd.Function(flow.mixed_space)
    for j in range(m_d):
        q.assign(q + X[j] * T[i, j] / np.sqrt(S[i]))
    Vs.append(q)

    q = fd.Function(flow.mixed_space)
    for j in range(m_a):
        q.assign(q + Y[j] * U[j, i] / np.sqrt(S[i]))
    Ws.append(q)

#
# Petrov-Galerkin projection
#

r = ru + rs  # Total dimension of reduced-order model
Ar = np.zeros((r, r), dtype=np.complex128)
Br = np.zeros((r, 1), dtype=np.complex128)
Cr = np.zeros((1, r), dtype=np.complex128)

A = flow.linearize(qB)
A.copy_output = True

def meas(q):
    flow.q.assign(q)
    CL, _CD = flow.get_observations()
    return CL

V = Vu + Vs
W = Wu + Ws
for i in range(r):
    for j in range(r):
        # For the unstable subspace the matrix is diagonal
        if (i < ru) or (j < ru):
            Ar[j, i] = 0.0
            if i == j:
                Ar[j, i] = evals[unstable_idx[i]]
        else:
            Ar[j, i] = flow.inner_product(A @ V[i], W[j])

    Br[i, 0] = flow.inner_product(qC, W[i])
    Cr[0, i] = meas(V[i])

# Finally the feedthrough term
Dr = meas(qC)

# Save the reduced-order basis
with fd.CheckpointFile(f"{output_dir}/rom_basis.h5", "w") as chk:
    chk.save_mesh(flow.mesh)
    for i in range(r):
        V[i].rename(f"V_{i}")
        chk.save_function(V[i])
        W[i].rename(f"W_{i}")
        chk.save_function(W[i])

# Save the ROM matrices
np.savez(f"{output_dir}/rom.npz", A=Ar, B=Br, C=Cr, D=Dr)