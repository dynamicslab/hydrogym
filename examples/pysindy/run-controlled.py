import numpy as np
import psutil
import scipy.io as sio

import hydrogym.firedrake as hgym

output_dir = "output"
checkpoint_dir = "checkpoints"
restart = f"{checkpoint_dir}/checkpoint.h5"
# restart = None

flow = hgym.Cylinder(Re=100, restart=restart)


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


paraview = hgym.io.ParaviewCallback(
    interval=100, filename=f"{output_dir}/controlled.pvd", postprocess=compute_vort
)


def log_postprocess(flow):
    mem_usage = psutil.virtual_memory().percent
    CL, CD = flow.get_observations()
    u = flow.control_state[0].values()[0]
    return CL, CD, u, mem_usage


print_fmt = (
    "t: {0:0.3f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t u: {3:0.6f}\t\t Mem: {4:0.1f}"
)
log = hgym.io.LogCallback(
    postprocess=log_postprocess,
    nvals=4,
    interval=1,
    print_fmt=print_fmt,
    filename=f"{output_dir}/results_controlled.dat",
)


callbacks = [
    # paraview,
    log,
]

Tf = 300
dt = 1e-4
n_steps = int(Tf // dt)

u = np.zeros(n_steps)  # Actuation history
y = np.zeros(n_steps)  # Lift coefficient
dy = np.zeros(n_steps)  # Derivative of lift coefficient

# See notebooks/sindy.ipynb for derivation
Kp = 0.0  # Proportional gain
# Kd = -65.4  # Derivative gain
Kd = -0.2  # Derivative gain

# Kp = -4.0    # Proportional gain
# Kd = 0.0  # Derivative gain

tau = 0.1 * flow.TAU


def controller(t, obs):
    # return np.sin(t)

    i = int(np.round(t / dt))

    # Turn on feedback control halfway through
    # if i > n_steps // 2:
    if i > 10:
        u[i] = -Kp * y[i - 1] - Kd * dy[i - 1]

    CL, CD = obs

    if i == 0:
        y[i] = CL

    else:
        # Low-pass filter
        y[i] = y[i - 1] + (dt / tau) * (CL - y[i - 1])

    # Estimate derivative
    dy[i] = (y[i] - y[i - 1]) / dt

    # dy_hat = (y[i] - y[i - 1]) / dt
    # dy[i] = dy[i - 1] + (dt / tau) * (dy_hat - dy[i - 1])

    sio.savemat(f"{output_dir}/pd-control.mat", {"y": y, "dy": dy, "u": u})

    return u[i]


hgym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, controller=controller)
