import os
import numpy as np
import psutil

from hydrogym.firedrake.utils.pd import PDController
import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

pd_controller = PDController(
    0.0,
    0.0,
    dt,
    n_steps,
    filter_type="bilinear",
    N=20,
)


def controller(t, obs):
    # Loop over phase angles for phasor control
    # First interval is zero actuation
    pd_controller.kp = 0.0
    pd_controller.kd = 0.0
    for j in range(n_phase):
        if t > (j + 1) * ctrl_time:
            pd_controller.kp = k * np.cos(phasors[j])
            pd_controller.kd = k * np.sin(phasors[j])

    return pd_controller(t, obs)


hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization=stabilization,
)
