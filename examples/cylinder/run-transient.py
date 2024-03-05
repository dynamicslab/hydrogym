import psutil

import hydrogym.firedrake as hgym

output_dir = "output"
pvd_out = None
restart = None
mesh_resolution = "medium"

element_type = "p1p1"
velocity_order = 1
stabilization = "gls"

checkpoint = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"

flow = hgym.RotaryCylinder(
    Re=100,
    restart=restart,
    mesh=mesh_resolution,
    velocity_order=velocity_order,
)

# Time step
tf = 300.0
dt = 0.01


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

interval = int(100 / dt)
callbacks = [
    log,
    hgym.utils.io.CheckpointCallback(interval=interval, filename=checkpoint),
]


hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    stabilization=stabilization,
)
