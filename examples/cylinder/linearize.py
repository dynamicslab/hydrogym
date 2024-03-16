# https://github.com/dynamicslab/hydrogym/blob/4c1c0e838147fff6cd3fd300294db14451ae120c/examples/cylinder/model-control/stability/stability.py
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from firedrake.pyplot import tripcolor, triplot
from scipy import sparse
from ufl import div, dot, dx, inner, nabla_grad

import hydrogym.firedrake as hgym

show_plots = False

checkpoint = "output/linearized.h5"

velocity_order = 2
stabilization = "none"

flow = hgym.Cylinder(
    Re=100,
    mesh="medium",
    velocity_order=velocity_order,
)

# Compute base flow
steady_solver = hgym.NewtonSolver(
    flow,
    stabilization=stabilization,
)
qB = steady_solver.solve()

# Check lift/drag
hgym.print(flow.compute_forces(qB))

if show_plots:
    vort = flow.vorticity(qB.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)
    plt.show()


# Time step
tf = 100.0
dt = 1e-2


# Set up the callbacks for logging and checkpointing
def log_postprocess(flow):
    CL, CD = flow.get_observations()
    return CL, CD


print_fmt = (
    "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"  # This will format the output
)
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename=None,
)

interval = int(1.0 // dt)
callbacks = [
    log,
    hgym.utils.io.CheckpointCallback(interval=interval, filename=checkpoint),
]


# Initialize the flow to a small random field
rng = fd.RandomGenerator(fd.PCG64())
flow.q.assign(1e-2 * rng.standard_normal(flow.mixed_space))

# Set up the linearized BDF solver
solver = hgym.LinearizedBDF(
    flow,
    dt,
    qB=qB,
    stabilization=stabilization,
)

solver.solve((0.0, tf), callbacks=callbacks, controller=None)
