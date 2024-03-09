import os

import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
mesh_resolution = "fine"
output_dir = f"./{Re}_{mesh_resolution}_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

method = "BDF"
stabilization = "gls"

flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=1,
    noise_amplitude=1.0,
)

# *** 1: Solve steady state (for computing fluctuation KE) ***
solver_parameters = {"snes_monitor": None}

# Since this flow is at high Reynolds number we have to
#    ramp to get the steady state
Re_init = np.arange(100, Re + 100, 100, dtype=float)

for i, Re in enumerate(Re_init):
    flow.Re.assign(Re)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    solver = hgym.NewtonSolver(
        flow, solver_parameters=solver_parameters, stabilization=stabilization
    )
    flow.qB.assign(solver.solve())

flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")

# *** 2: Transient solve with natural flow ***
tf = 500
dt = 1e-2


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = flow.evaluate_objective()
    CFL = flow.max_cfl(dt)
    mem_usage = psutil.virtual_memory().percent
    return [CFL, KE, TKE, mem_usage]


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
)
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
)
