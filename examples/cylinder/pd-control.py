import numpy as np
import psutil  # For memory tracking
import os
import scipy.io as sio

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_resolution = "medium"

element_type = "p1p1"
velocity_order = 1
stabilization = "gls"

# Make sure to run the transient simulation first to generate the restart file
restart = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"
checkpoint = f"{output_dir}/pd_{mesh_resolution}_{element_type}.h5"


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
    # hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=1,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=f"{output_dir}/pd-control.dat",
    ),
]


flow = hgym.RotaryCylinder(
    Re=100,
    mesh=mesh_resolution,
    restart=restart,
    callbacks=callbacks,
    velocity_order=velocity_order,
)


omega = 1 / 5.56  # Vortex shedding frequency
k = 2.0  # Base gain
theta = 4.0  # Phase angle
kp = k * np.cos(theta)
kd = k * np.sin(theta)

tf = 100.0
dt = 0.01
n_steps = int(tf // dt) + 2

u = np.zeros(n_steps)  # Actuation history
x = np.zeros(n_steps)  # Actuator state
y_raw = np.zeros(n_steps)  # Lift coefficient (unfiltered)
y = np.zeros(n_steps)  # Lift coefficient (filtered)
dy = np.zeros(n_steps)  # Derivative of lift coefficient

CL, CD = flow.get_observations()
y[0] = CL
y_raw[0] = CL

i = 0
tau = 10 * flow.TAU

# Bilinear filter coefficients for derivative
N = 100
filter_type = "bilinear"
if filter_type == "none":
    a = [1, -1]
    b = [1, 0]
elif filter_type == "forward":
    a = [1, N*dt - 1]
    b = [N, -N]
elif filter_type == "backward":
    a = [N*dt + 1, -1]
    b = [N, -N]
elif filter_type == "bilinear":
    a = [N*dt + 2, N*dt - 2]
    b = [2*N, -2*N]

def controller(t, obs):
    global i  # FIXME: Don't use global variable here
    i += 1

    # Turn on feedback control halfway through
    if t > tf / 2:
        # if t > 10:
        u[i] = -kp * y[i - 1] - kd * dy[i - 1]

    CL, CD = obs

    # Low-pass filter and estimate derivative
    y[i] = y[i - 1] + (dt / tau) * (CL - y[i - 1])
    dy[i] = (y[i] - y[i - 1]) / dt
    y_raw[i] = CL
    # dy[i] = (b[0]*y_raw[i] + b[1]*y_raw[i-1] - a[1]*dy[i-1]) / a[0]

    x[i] = flow.actuators[0].state

    if i % 100 == 0:
        data = {
            "y": y[:i],
            "dy": dy[:i],
            "u": u[:i],
            "x": x[:i],
            "CL": y_raw[:i],
            "t": np.arange(0, i * dt, dt),
        }
        sio.savemat(f"{output_dir}/pd-control.mat", data)

    return u[i]


hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization=stabilization,
)

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
