import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'output'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

cyl = gym.flows.Cylinder()

# cyl.solve_steady()  # Initialize with steady state
cyl.load_checkpoint(chk_out)  # Reload previous solution

# Time step
dt = 1e-2
Tf = 100.0

vort = fd.Function(cyl.pressure_space, name='vort')
def compute_vort(state):
    u, p = state
    vort.assign(fd.project(curl(u), cyl.pressure_space))
    return (u, p, vort)

data = np.array([0, 0, 0], ndmin=2)
def forces(iter, t, state):
    global data
    u, p = state
    CL, CD = cyl.compute_forces(u, p)
    if fd.COMM_WORLD.rank == 0:
        data = np.append(data, np.array([t, CL, CD], ndmin=2), axis=0)
        np.savetxt(f'{output_dir}/coeffs.dat', data)
    PETSc.Sys.Print(f't:{t:08f}\t\t CL:{CL:08f} \t\tCD::{CD:08f}')

callbacks = [
    gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    # gym.io.CheckpointCallback(flow=cyl, interval=100, filename=chk_out),
    gym.io.GenericCallback(callback=forces, interval=1)
]

# callbacks = []
solver = gym.IPCSSolver(cyl, dt=dt, callbacks=callbacks)
solver.solve(Tf)