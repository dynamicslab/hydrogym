import numpy as np
import psutil
import scipy.io as sio

import hydrogym.firedrake as hgym

output_dir = "output"


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


def log_postprocess(flow):
    CL, CD = flow.get_observations()
    mem_usage = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_usage = psutil.virtual_memory().percent
    return CL, CD, mem_usage


callbacks = [
    # hgym.io.ParaviewCallback(
    #     interval=10, filename=f"{output_dir}/pd-control.pvd", postprocess=compute_vort
    # ),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=10,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=None,
    ),
]


flow = hgym.Cylinder(
    Re=100,
    mesh="coarse",
    restart="../demo/checkpoint-coarse.h5",
    callbacks=callbacks,
)
Tf = 1000
dt = 1e-2
n_steps = int(Tf // dt)

u = np.zeros(n_steps)  # Actuation history
y = np.zeros(n_steps)  # Lift coefficient
dy = np.zeros(n_steps)  # Derivative of lift coefficient

Kp = -4.0  # Proportional gain
Kd = 0.0  # Derivative gain


def controller(t, obs):
    # return np.sin(t)

    i = int(t // dt)

    # Turn on feedback control halfway through
    # if i > n_steps // 2:
    #     u[i] = -Kp * y[i - 1] - Kd * dy[i - 1]

    u[i] = -Kp * y[i - 1] - Kd * dy[i - 1]

    CL, CD = obs

    # Low-pass filter and estimate derivative
    y[i] = y[i - 1] + (dt / flow.TAU) * (CL - y[i - 1])
    dy[i] = (y[i] - y[i - 1]) / dt

    sio.savemat(f"{output_dir}/pd-control.mat", {"y": y, "u": u})

    return u[i]


hgym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, controller=controller)

# for i in range(1, n_steps):
#     # Turn on feedback control halfway through
#     if i > n_steps // 2:
#         u[i] = -Kp * y[i - 1] - Kd * dy[i - 1]

#     # Advance state and collect measurements
#     (CL, CD), _, _, _ = env.step(u[i])

#     # Low-pass filter and estimate derivative
#     y[i] = y[i - 1] + (dt / env.flow.TAU) * (CL - y[i - 1])
#     dy[i] = (y[i] - y[i - 1]) / dt

#     hg.print(
#         f"Step: {i:04d},\t\t CL: {y[i]:0.4f}, \t\tCL_dot: {dy[i]:0.4f},\t\tu: {u[i]:0.4f}"
#     )
#     sio.savemat(f"{output_dir}/pd-control.mat", {"y": y, "u": u})
