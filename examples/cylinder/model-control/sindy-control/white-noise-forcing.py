import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'noise-response'
pvd_out = f"{output_dir}/solution.pvd"
restart = f"checkpoint.h5"

# Time step
Tf = 300
dt = 1e-2

n_steps = int(Tf//dt)
u = 1e-2*gym.utils.white_noise(n_steps, fs=1/dt, cutoff=0.2)  # Actuation

def log_postprocess(flow):
    CL, CD = flow.collect_observations()
    omega = flow.omega.values()[0]
    return CL, CD, omega
    
# Set up the callback
print_fmt = "t: {0},\t\t CL:{1:0.3f},\t\t CD:{2:0.03f}"  # This will format the output
log = gym.io.LogCallback(
    postprocess=log_postprocess,
    nvals=3,
    interval=1,
    print_fmt=print_fmt,
    filename=f'{output_dir}/force.dat'
)

callbacks = [
    log,
    gym.io.SnapshotCallback(interval=5, filename=f'{output_dir}/snapshots.h5')
]

env = gym.env.CylEnv(Re=100, checkpoint=restart, mesh='noack', callbacks=callbacks)

for i in range(n_steps):
    env.step(u[i])