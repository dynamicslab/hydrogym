import firedrake as fd
from firedrake.petsc import PETSc
from ufl import sqrt
import numpy as np

import hydrogym as gym

Re = 5000
output_dir = f'./{Re}_output'
pvd_out = f"{output_dir}/solution.pvd"
stability_path = './stability/global-modes/'
restart = f"{stability_path}/steady.h5"
checkpoint = f'checkpoint.h5'
evec_file = f"{stability_path}/direct_real.h5"

flow = gym.flow.Cavity(Re=Re)
flow.load_checkpoint(restart)  # Reload previous solution

# Initialize with unstable eigenmode
with fd.CheckpointFile(evec_file, 'r') as file:
    v = file.load_function(flow.mesh, 'q', idx=0)
    
flow.q += fd.Constant(1e-2)*v

# Time step
Tf = 100
dt = 1e-4

h = fd.CellSize(flow.mesh)
def log_postprocess(flow):
    KE = 0.5*fd.assemble(fd.inner(flow.u, flow.u)*fd.dx)
    CFL = fd.project(dt*sqrt(flow.u.sub(0)**2 + flow.u.sub(1)**2)/h, flow.pressure_space).vector().max()
    return [CFL, KE]

print_fmt = 't: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}'
callbacks = [
    gym.io.CheckpointCallback(interval=100, filename=checkpoint),
    gym.io.LogCallback(postprocess=log_postprocess, nvals=2, interval=10,
            filename=f'{output_dir}/stats.dat',
            print_fmt=print_fmt
    )
]

gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')