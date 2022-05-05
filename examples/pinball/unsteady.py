import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

Re = 30
output_dir = 'output'
pvd_out = f"{output_dir}/{Re}_solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Pinball(Re=Re)

# flow.solve_steady()  # Initialize with steady state
flow.load_checkpoint(chk_out)  # Reload previous solution

# Time step
dt = 1e-2
Tf = 100.0

def compute_vort(state):
    u, p = state
    return (u, p, flow.vorticity())

data = np.zeros((1, 7))
def forces(iter, t, state):
    global data
    u, p = state
    CL, CD = flow.compute_forces(u, p)
    if fd.COMM_WORLD.rank == 0:
        data = np.append(data, np.array([t, *CL, *CD], ndmin=2), axis=0)
        np.savetxt(f'{output_dir}/{Re}_coeffs.dat', data)
    PETSc.Sys.Print(f't:{t:0.2f}\t\t CL:{np.array(CL)} \t\tCD::{np.array(CD)}')

callbacks = [
    gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    gym.io.CheckpointCallback(flow=flow, interval=100, filename=chk_out),
    gym.io.GenericCallback(callback=forces, interval=1)
]

# callbacks = []
solver = gym.ts.IPCSSolver(flow, dt=dt, callbacks=callbacks)
solver.solve(Tf)