import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np

assert (PETSc.ScalarType == np.float64), "Run in real mode"

import scipy.io as sio

import hydrogym as gym

mesh = 'noack'
evec_dir = 'global-modes'
output_dir = 'continuous'

flow = gym.flow.Cylinder(Re=50, h5_file=f'{evec_dir}/steady.h5')
qB = flow.q.copy(deepcopy=True)

ctrl = sio.loadmat(f'{output_dir}/controller.mat')
A = ctrl['A']
B = ctrl['B']
C = ctrl['C']
K = ctrl['K']
L = ctrl['L']
V = ctrl['V']
P = ctrl['P']



with fd.CheckpointFile(f'{output_dir}/v0.h5', 'r') as file:
    vr = file.load_function(flow.mesh, 'q')

with fd.CheckpointFile(f'{output_dir}/v1.h5', 'r') as file:
    vi = file.load_function(flow.mesh, 'q')

with fd.CheckpointFile(f'{output_dir}/w0.h5', 'r') as file:
    wr = file.load_function(flow.mesh, 'q')

with fd.CheckpointFile(f'{output_dir}/w1.h5', 'r') as file:
    wi = file.load_function(flow.mesh, 'q')

### Test
dt = 1e-2
# solver = gym.ts.IPCS(flow, dt=dt)
# solver.linearize(qB, return_operators=False)
Tf = 100
n_steps = int(Tf//dt)

r = A.shape[0]
x = np.zeros((r, n_steps))
x_hat = np.zeros((r, n_steps))
u = np.zeros((2, n_steps))  # Rotation rate and derivative
y = np.zeros((C.shape[0], n_steps))
y_hat = np.zeros((C.shape[0], n_steps))

flow.q.assign(fd.Constant(1e-2)*vr)

# Initialize
q = gym.utils.get_array(flow.q)
x[0, 0] = -flow.dot(flow.q, wr)
x[1, 0] =  flow.dot(flow.q, wi)
x_hat[:, 0] = x[:, 0]

solver = gym.ts.LinearizedIPCS(flow, dt, qB)

B = -B*dt #???
for i in range(1, n_steps):
    # Advance state and collect measurements
    # if i > n_steps//2:
    #     u[1, i] = - K @ x_hat[:, i-1]     # Update derivative with feedback control

    if i == 1000:
        u[1, i] = 1/dt  # Delta function input

    u[0, i] = u[0, i-1] + dt*u[1, i]  # Integrate for actual rotation rate
    solver.step(i, control=u[0, i])

    # Low-pass filter measurements
    y_hat[:, i] = flow.collect_observations()
    # y[:, i] = y[:, i-1] + (dt/flow.TAU)*(y_hat[:, i] - y[:, i-1])
    y[:, i] = y_hat[:, i]

    # # True projected state for debugging
    x[0, i] = -flow.dot(flow.q, wr)
    x[1, i] =  flow.dot(flow.q, wi)
    
    # Update Kalman filter (forward Euler)
    err = y[:, i] - C @ x_hat[:, i-1]
    x_hat[:, i] = x_hat[:, i-1] + dt*(A @ x_hat[:, i-1] + B[:, 0]*u[1, i] + L @ err)
    
    gym.print((i, x[:, i], x_hat[:, i], np.linalg.norm(err)))
    gym.print((y[:, i], y_hat[:, i], u[:, i]))
    if gym.is_rank_zero():
        sio.savemat(f'{output_dir}/lti-ctrl.mat', {'u': u, 'x': x, 'y': y, 'x_hat': x_hat, 'y_hat': y_hat})