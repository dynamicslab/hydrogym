import numpy as np

# import scipy.io as sio
import psutil

import hydrogym as gym

# from firedrake import logging
# logging.set_log_level(logging.DEBUG)

output_dir = "output"


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


def log_postprocess(flow):
    CL, CD = flow.get_observations()
    mem_usage = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_usage = psutil.virtual_memory().percent
    return CL, CD, mem_usage


callbacks = [
    # gym.io.ParaviewCallback(
    #     interval=10, filename=f"{output_dir}/pd-control.pvd", postprocess=compute_vort
    # ),
    gym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=10,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=None,
    ),
]

env_config = {
    "Re": 100,
    "dt": 1e-2,
    "mesh": "coarse",
    "callbacks": callbacks,
    # "differentiable": True,
    "checkpoint": "../demo/checkpoint-coarse.h5",
}

env = gym.env.CylEnv(env_config)
Tf = 1000
dt = env.solver.dt
n_steps = int(Tf // dt)

env.reset()
Kp = -4.0  # Proportional gain
y = 0.0
for i in range(1, n_steps):
    # Advance state and collect measurements
    # u = -Kp*y
    # env.flow.reset_control()
    u = np.sin(i * dt)
    (CL, CD), _, _, _ = env.step(u)

    # Low-pass filter and estimate derivative
    y += (dt / env.flow.TAU) * (CL - y)


# # With control: crashes in 26/100 with Tf=10, 3/100 with Tf=100, 1/100 with Tf=1000
# n_epoch = 100  # "epochs" to test apparent memory leak issue
# for i in range(n_epoch):
#     gym.print(f'Beginning epoch {i+1}/{n_epoch}')

#     env.reset()

#     u = np.zeros(n_steps)  # Actuation history
#     y = np.zeros(n_steps)  # Lift coefficient
#     dy = np.zeros(n_steps)  # Derivative of lift coefficient

#     Kp = -4.0  # Proportional gain
#     Kd = 0.0  # Derivative gain

#     for i in range(1, n_steps):
#         # Turn on feedback control halfway through
#         # if i > n_steps // 2:
#         if False:
#             u[i] = -Kp * y[i - 1] - Kd * dy[i - 1]

#         # Advance state and collect measurements
#         (CL, CD), _, _, _ = env.step(u[i])

#         # Low-pass filter and estimate derivative
#         y[i] = y[i - 1] + (dt / env.flow.TAU) * (CL - y[i - 1])
#         dy[i] = (y[i] - y[i - 1]) / dt

#         # gym.print(
#         #     f"Step: {i:04d},\t\t CL: {y[i]:0.4f}, \t\tCL_dot: {dy[i]:0.4f},\t\tu: {u[i]:0.4f}"
#         # )
#         # sio.savemat(f"{output_dir}/pd-control.mat", {"y": y, "u": u})
