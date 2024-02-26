import psutil

import hydrogym.firedrake as hgym

output_dir = "."
pvd_out = None
restart = None
checkpoint = "checkpoint.h5"

flow = hgym.Cylinder(
    Re=100,
    restart=restart,
    mesh="medium",
    velocity_order=1,
)

# Time step
Tf = 1.0
dt = 0.1  # Tested as high as 0.25


def log_postprocess(flow):
    mem_usage = psutil.virtual_memory().percent
    CL, CD = flow.get_observations()
    return CL, CD, mem_usage


# Set up the callback
print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"  # This will format the output
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=3,
    interval=1,
    print_fmt=print_fmt,
    filename=None,
)

callbacks = [
    log,
    # hgym.utils.io.CheckpointCallback(interval=10, filename=checkpoint),
]


def controller(t, y):
    return [1.0]


hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method="BDF",
    stabilization="gls",
    # controller=controller,
)
