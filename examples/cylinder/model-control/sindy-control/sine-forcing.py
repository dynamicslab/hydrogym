import numpy as np

import hydrogym as gym

output_dir = "sine-response"
pvd_out = f"{output_dir}/solution.pvd"
restart = "../transient/checkpoint.h5"

# Time step
Tf = 300
dt = 1e-2

n_steps = int(Tf // dt)
t = np.arange(0, Tf, dt)
omega = 1.13  # Natural frequency
u = 1e-2 * np.sin(0.95 * omega * t)  # Actuation

env = gym.env.CylEnv(Re=100, checkpoint=restart)


def log_postprocess(flow):
    CL, CD = flow.collect_observations()
    omega = flow.omega.values()[0]
    return CL, CD, omega


# Set up the callbacks
log_cb = gym.io.LogCallback(
    postprocess=log_postprocess,
    nvals=3,
    interval=1,
    print_fmt="t: {0:0.2f},\t\t CL:{1:0.3f},\t\t CD:{2:0.03f}",
    filename=f"{output_dir}/force.dat",
)

env.set_callbacks(
    [
        log_cb,
        # pod_cb,
        gym.io.SnapshotCallback(interval=10, filename=f"{output_dir}/snapshots.h5"),
    ]
)

for i in range(n_steps):
    env.step(u[i])
