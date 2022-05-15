import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

assert (PETSc.ScalarType == np.float64), "Run in real mode"
assert (fd.COMM_WORLD.size == 1), "Run on single core"

from scipy import linalg
import scipy.io as sio

import hydrogym as gym

import control

mesh = 'noack'
evec_dir = 'global-modes'

output_dir = 'controller'
# flow = gym.flow.Cylinder(Re=50, h5_file=f'../stability/steady.h5')
flow = gym.flow.Cylinder(Re=50, h5_file=f'{evec_dir}/steady.h5')
qB = flow.q.copy(deepcopy=True)

# Construct modal basis from real/imag parts
with flow.q.dat.vec_ro as vec:
    N = vec.size
V = np.zeros((N, 2))
W = np.zeros_like(V)

flow.load_checkpoint(f'{evec_dir}/direct_real.h5', idx=0, read_mesh=False)
V[:, 0] = gym.utils.get_array(flow.q)
    
flow.load_checkpoint(f'{evec_dir}/direct_imag.h5', idx=0, read_mesh=False)
V[:, 1] = gym.utils.get_array(flow.q)
    
flow.load_checkpoint(f'{evec_dir}/adjoint_real.h5', idx=0, read_mesh=False)
W[:, 0] = gym.utils.get_array(flow.q)

flow.load_checkpoint(f'{evec_dir}/adjoint_imag.h5', idx=0, read_mesh=False)
W[:, 1] = gym.utils.get_array(flow.q)

M = flow.mass_matrix(backend='scipy')

# Action of "A" and "C" on basis vectors
dt = 1e-2
solver = gym.ts.IPCS(flow, dt=dt)
A, B = solver.linearize(qB)

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
Br = P @ B

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
sio.savemat(f'{output_dir}/controller.mat', {'A': Ar, 'B': Br, 'C': Cr, 'K': K, 'L': L, 'V': V, 'P': P})

### Also save eigenvector as Function
gym.utils.set_from_array(flow.q, V[:, 0])
# flow.save_checkpoint('evec.h5')

### Test Kalman (no control)
print("Testing Kalman filter...")
Tf = 100
dt = 1e-2
n_steps = int(Tf//dt)

x = np.zeros((r, n_steps))
y = np.zeros((y_dim, n_steps))
x_hat = np.zeros((r, n_steps))
q = 1e-2*V[:, 0]

# Initialize
x[:, 0] = P @ q

for i in range(1, n_steps):
    # Advance state and collect measurements
    q = A @ q  # Advance natural state
    gym.utils.set_from_array(flow.q, q)
    y[:, i] = flow.collect_observations()
    x[:, i] = P @ q  # True projected state
    
    err = y[:, i] - Cr @ x_hat[:, i-1]
    x_hat[:, i] = Ar @ x_hat[:, i-1] + L @ err
    
    print(i, x[:, i], y[:, i], x_hat[:, i], np.linalg.norm(err))

sio.savemat(f'{output_dir}/lti-kalman.mat', {'x': x, 'y': y, 'x_hat': x_hat})

### Test LQR controller
print("Testing LQR...")

x = np.zeros((r, n_steps))
y = np.zeros((y_dim, n_steps))
x_hat = np.zeros((r, n_steps))
q = 1e-2*V[:, 0]

# Initialize
x[:, 0] = P @ q

for i in range(1, n_steps):
    # Advance state and collect measurements
    q = A @ q  - B @ (K @ x_hat[:, i-1]) # Advance controlled state
    gym.utils.set_from_array(flow.q, q)
    y[:, i] = flow.collect_observations()
    x[:, i] = P @ q  # True projected state
    
    err = y[:, i] - Cr @ x_hat[:, i-1]
    x_hat[:, i] = Ar @ x_hat[:, i-1] + L @ err
    
    print(i, x[:, i], y[:, i], x_hat[:, i], np.linalg.norm(err))

sio.savemat(f'{output_dir}/lti-ctrl.mat', {'x': x, 'y': y, 'x_hat': x_hat})