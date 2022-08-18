from common import checkpoint, flow, output_dir, snap_file

import hydrogym as gym

# Time step
Tf = 5.56
dt = Tf / 500


def log_postprocess(flow):
    return flow.collect_observations()


# Set up the callback
print_fmt = (
    "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"  # This will format the output
)
log = gym.io.LogCallback(
    postprocess=log_postprocess,
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename=f"{output_dir}/force.dat",
)

callbacks = [
    log,
    gym.io.SnapshotCallback(interval=5, filename=snap_file),
    gym.io.CheckpointCallback(
        interval=100, filename=checkpoint
    ),  # Shouldn't need this, just for debugging
]
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method="IPCS")
