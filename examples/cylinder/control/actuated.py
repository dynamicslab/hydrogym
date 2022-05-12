import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

import scipy.io as sio
from scipy import linalg

output_dir = 'controlled'
pvd_out = f"{output_dir}/solution.pvd"

Re = 50

def compute_vort(flow):
    u, p = flow.u, flow.p
    return (u, p, flow.vorticity())

print_fmt = "t: {0:0.2f},\t\t CL:{1:0.5f},\t\t CD:{2:0.05f}"  # This will format the output
log = gym.io.LogCallback(
    postprocess=lambda flow: flow.collect_observations(),
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename='force_ctrl.dat'
)

callbacks = [
    gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    log
]

Tf = 100
dt = 1e-2
num_steps = int(Tf//dt)

mesh = 'sipp-lebedev'
Re = 50
env = gym.env.CylEnv(Re=Re, checkpoint=f'./steady.h5', dt=dt, callbacks=callbacks)
qB = env.flow.q.copy(deepcopy=True)
y0 = np.array(env.flow.collect_observations())

# Load system
sys = sio.loadmat('controller.mat')
A = sys['A']  # Dynamics matrix
B = sys['B']  # Control matrix
C = sys['C']  # Measurement matrix
K = sys['K'][0]  # Feedback gain
L = sys['L']  # Kalman gains
V = sys['V']  # Stability modes
P = sys['P']  # Petrov-Galerkin projection matrix

# Perturb from origin
from hydrogym.utils import get_array
with fd.CheckpointFile('evec.h5', 'r') as chk:
    q1 = chk.load_function(env.flow.mesh, 'q')

u1, _ = q1.split()
q1_norm = np.sqrt(fd.assemble(fd.inner(u1, u1)*fd.dx))
q1 = q1/q1_norm
env.flow.q.assign(qB + 1e2*q1)

x = np.zeros((num_steps, 2))
x_hat = np.zeros_like(x)
# x[0, :] = P @ get_array(env.flow.q)
x_hat[0, :] = x[0, :]

y = np.zeros((num_steps, 2))
y[0, :] = np.array(env.flow.collect_observations()) - y0

# NOTE: This just always blows up... seems like we must be missing some offset or something.
for i in range(1, num_steps):
    # obs, reward, done, info = env.step()
    action = -np.dot(K, x_hat[i-1, :])   # Negative sign???

    obs, reward, done, info = env.step(action)
    y[i, :] = np.array(obs) - y0  # Linearize observation

    err = y[i, :] - C @ x_hat[i-1, :]
    x_hat[i, :] = A @ x_hat[i-1, :] + L @ err
    gym.print((i, action, y[i, :], x_hat[i, :], np.linalg.norm(err)))