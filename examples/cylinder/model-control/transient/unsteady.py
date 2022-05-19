import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'output'
pvd_out = f"{output_dir}/solution.pvd"
stability_path = '../stability/global-modes/'
restart = f"{stability_path}/steady.h5"
checkpoint = f'checkpoint.h5'
evec_file = f"{stability_path}/direct_real.h5"

flow = gym.flow.Cylinder(Re=100, h5_file=restart)

# Initialize with unstable eigenmode
with fd.CheckpointFile(evec_file, 'r') as file:
    v = file.load_function(flow.mesh, 'q', idx=0)
flow.q += fd.Constant(1e-2)*v

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
    filename=f'{output_dir}/force.dat'
)

callbacks = [
    log,
    gym.io.SnapshotCallback(interval=10, filename=f'{output_dir}/snapshots.h5'),
    gym.io.CheckpointCallback(interval=100, filename=checkpoint)
]

gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')