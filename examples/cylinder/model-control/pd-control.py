import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

from scipy import linalg
import scipy.io as sio

import hydrogym as gym

output_dir = 'output'
def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())

callbacks = [
    gym.io.ParaviewCallback(interval=10, filename=f'{output_dir}/pd-control.pvd', postprocess=compute_vort)
]

env = gym.env.CylEnv(Re=100, checkpoint=f'output/checkpoint.h5', callbacks=callbacks)
Tf = 100
dt = env.solver.dt
n_steps = int(Tf//dt)

u = np.zeros(n_steps)      # Actuation history
y = np.zeros(n_steps)      # Lift coefficient
dy = np.zeros(n_steps)     # Derivative of lift coefficient

Kp = -4.0   # Proportional gain
Kd = 0.0    # Derivative gain

for i in range(1, n_steps):
    # Turn on feedback control halfway through
    if i > n_steps//2:
        u[i] = -Kp*y[i-1] - Kd*dy[i-1]

    # Advance state and collect measurements
    (CL, CD), _, _, _ = env.step(u[i])

    # Low-pass filter and estimate derivative
    y[i] = y[i-1] + (dt/env.flow.TAU)*(CL - y[i-1])
    dy[i] = (y[i] - y[i-1])/dt

    gym.print(f'Step: {i:04d},\t\t CL: {y[i]:0.4f}, \t\tCL_dot: {dy[i]:0.4f},\t\tu: {u[i]:0.4f}')
    sio.savemat(f'{output_dir}/pd-control.mat', {'y': y, 'u': u})