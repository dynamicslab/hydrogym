import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = '.'
pvd_out = None
restart = None
checkpoint = f'checkpoint-coarse.h5'

flow = gym.flow.Cylinder(Re=100, h5_file=restart, mesh='coarse')

# Time step
Tf = 300
dt = 1e-2

def log_postprocess(flow):
    return flow.collect_observations()
    
# Set up the callback
print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"  # This will format the output
log = gym.io.LogCallback(
    postprocess=log_postprocess,
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename=None
)

callbacks = [
    log,
    gym.io.CheckpointCallback(interval=100, filename=checkpoint)
]

gym.print("Beginning integration")
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')