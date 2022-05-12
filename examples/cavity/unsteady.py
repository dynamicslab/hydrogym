import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl, sqrt
import numpy as np
from firedrake import logging
# logging.set_level(logging.DEBUG)

import hydrogym as gym

Re = 4000
output_dir = f'{Re}_output'
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Cavity(Re=Re)

# flow.solve_steady()  # Initialize with steady state
flow.load_checkpoint(chk_out)  # Reload previous solution

# Time step
dt = 1e-4
Tf = 100.0

def compute_vort(flow):
    u, p = flow.u, flow.p
    return (u, p, flow.vorticity())

h = fd.CellSize(flow.mesh)
def log_postprocess(flow):
    KE = 0.5*fd.assemble(fd.inner(flow.u, flow.u)*fd.dx)
    CFL = fd.project(dt*sqrt(flow.u.sub(0)**2 + flow.u.sub(1)**2)/h, flow.pressure_space).vector().max()
    return [CFL, KE]

print_fmt = 't: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}'
callbacks = [
    gym.io.CheckpointCallback(interval=100, filename=chk_out),
    gym.io.LogCallback(postprocess=log_postprocess, nvals=2, interval=10,
            filename=f'{output_dir}/stats.dat',
            print_fmt=print_fmt
    )
]

# callbacks = []
# solver = gym.ts.IPCSSolver(flow, dt=dt, callbacks=callbacks)
# solver.solve(Tf)
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks)