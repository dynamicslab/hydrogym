import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

Re = 130
output_dir = f'{Re}_output'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Pinball(Re=Re)

# flow.solve_steady()  # Initialize with steady state
flow.load_checkpoint(chk_out)  # Reload previous solution

# Time step
dt = 1e-2
Tf = 300.0

def compute_vort(flow):
    u, p = flow.u, flow.p
    return (u, p, flow.vorticity())

data = np.zeros((1, 7))
def forces(iter, t, flow):
    global data
    CL, CD = flow.compute_forces(flow.q)
    if fd.COMM_WORLD.rank == 0:
        data = np.append(data, np.array([t, *CL, *CD], ndmin=2), axis=0)
        np.savetxt(f'{output_dir}/coeffs.dat', data)
    PETSc.Sys.Print(f't:{t:0.2f}\t\t CL:{np.array(CL)} \t\tCD::{np.array(CD)}')

callbacks = [
    gym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    gym.io.CheckpointCallback(interval=100, filename=chk_out),
    gym.io.GenericCallback(callback=forces, interval=10)
]

# callbacks = []
# solver = gym.ts.IPCSSolver(flow, dt=dt, callbacks=callbacks)
# solver.solve(Tf)
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks)