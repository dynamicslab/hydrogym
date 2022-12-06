import hydrogym.firedrake as hgym

output_dir = "."
pvd_out = None
restart = None
checkpoint = "checkpoint-coarse.h5"

flow = hgym.Cylinder(Re=100, h5_file=restart, mesh="coarse")

# Time step
Tf = 300
dt = 1e-2


def log_postprocess(flow):
    return flow.get_observations()


# Set up the callback
print_fmt = (
    "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"  # This will format the output
)
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess, nvals=2, interval=1, print_fmt=print_fmt, filename=None
)

callbacks = [log, hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint)]

hgym.print("Beginning integration")
hgym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method="IPCS")
