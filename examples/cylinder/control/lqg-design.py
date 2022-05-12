import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

from scipy import linalg
import scipy.io as sio

import hydrogym as gym

import control

mesh = 'sipp-lebedev'
# flow = gym.flow.Cylinder(Re=50, h5_file=f'../stability/steady.h5')
flow = gym.flow.Cylinder(Re=50, h5_file=f'../output/{mesh}-steady.h5')
qB = flow.q.copy(deepcopy=True)
dt = 1e-2
solver = gym.ts.IPCS(flow, dt=dt)
A, B = solver.linearize(qB)  # Full-size dynamics and control matrices

# Load results of stability analysis
evec_path = '../stability/output-split'
evec = np.load(f'{evec_path}/evec.npy')
evecH = np.load(f'{evec_path}/evecH.npy')

### Petrov-Galerkin projection

# Construct modal basis from real/imag parts
V = np.zeros((evec.shape[0], 2))
W = np.zeros_like(V)

V[:, 0] = np.real(evec[:, 0])
V[:, 1] = np.imag(evec[:, 0])
W[:, 0] = np.real(evecH[:, 0])
W[:, 1] = np.imag(evecH[:, 0])

M = flow.mass_matrix(backend='scipy')

# Construct action of "A" and "C" on basis vectors
N = evec.shape[0]
r = 2
y_dim = 2
AV = np.zeros((N, r))
Cr = np.zeros((y_dim, r))
for i in range(r):
    AV[:, i] = A * V[:, i]

    gym.utils.set_from_array(flow.q, V[:, i])
    Cr[:, i] = flow.collect_observations()

# Projection
P = linalg.inv(W.T @ M @ V) @ (W.T @ M)
Ar = P @ AV
Br = P @ B[:, None]


dt_evals, _ = linalg.eig(Ar)
print(np.log(dt_evals)/dt) # Continuous-time eigenvalues

### Controller design
# LQR
Q = np.eye(r)
R = 1e8
K, _, ctrl_evals = control.dlqr(Ar, Br, Q, R)
print(np.log(ctrl_evals)/dt) # Stabilized eigenvalues

# LQE
QN = 1e-6*np.eye(r)
RN = 1e-6*np.eye(y_dim)
GN = np.eye(r)
L, _, _ = control.dlqe(Ar, GN, Cr, QN, RN)

### Save results
sio.savemat('controller.mat', {'A': Ar, 'B': Br, 'C': Cr, 'K': K, 'L': L, 'V': V, 'P': P})

### Also save eigenvector as Function
gym.utils.set_from_array(flow.q, V[:, 0])
flow.save_checkpoint('evec.h5')