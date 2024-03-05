import numpy as np
import psutil
import scipy.io as sio

import hydrogym.firedrake as hgym

output_dir = "output"

mesh_resolution = "medium"

element_type = "p1p1"
velocity_order = 1
stabilization = "gls"

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
        filename=f"{output_dir}/phase-sweep.dat",
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
k = 0.1  # Base gain
ctrl_time = 10 / omega  # Time for each phasor - approx 10 vortex shedding periods
n_phase = 20
phasors = np.linspace(0.0, 2 * np.pi, n_phase)  # Phasor angles

tf = (n_phase + 1) * ctrl_time
dt = 1e-2
n_steps = int(tf // dt) + 2

u = np.zeros(n_steps)  # Actuation history
x = np.zeros(n_steps)  # Actuator state
y_raw = np.zeros(n_steps)  # Lift coefficient (unfiltered)
y = np.zeros(n_steps)  # Lift coefficient (filtered)
dy = np.zeros(n_steps)  # Derivative of lift coefficient
theta = np.zeros(n_steps)

CL, CD = flow.get_observations()
y[0] = CL
y_raw[0] = CL

i = 0
tau = 0.1 / omega


def controller(t, obs):
    global i  # FIXME: Don't use global variable here

    i += 1

    # Loop over phase angles for phasor control
    # First interval is zero actuation
    kp = 0.0
    kd = 0.0
    for j in range(n_phase):
        if t > (j + 1) * ctrl_time:
            theta[i] = phasors[j]
            kp = k * np.cos(theta[i])
            kd = k * np.sin(theta[i])
            u[i] = -kp * y[i - 1] - kd * dy[i - 1]

    CL, CD = obs

    # Low-pass filter and estimate derivative
    y[i] = y[i - 1] + (dt / tau) * (CL - y[i - 1])
    dy[i] = (y[i] - y[i - 1]) / dt
    y_raw[i] = CL

    x[i] = flow.actuators[0].state

    if i % 100 == 0:
        data = {
            "y": y[:i],
            "dy": dy[:i],
            "u": u[:i],
            "x": x[:i],
            "CL": y_raw[:i],
            "theta": theta[:i],
            "t": np.arange(0, i * dt, dt),
        }
        sio.savemat(f"{output_dir}/phase-sweep.mat", data)

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
