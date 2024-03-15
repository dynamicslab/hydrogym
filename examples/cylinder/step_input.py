"""Simulate step function input at Re=40"""
import os

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

stabilization = "gls"
flow = hgym.Cylinder(
    Re=40,
    mesh="medium",
    velocity_order=1,
)

# 1. Compute base flow
steady_solver = hgym.NewtonSolver(
    flow,
    stabilization=stabilization,
)
qB = steady_solver.solve()


# 2. Set up step input
def controller(t, u):
    return flow.MAX_CONTROL if t > 5.0 else 0.0


# 3. Set up the logging callback
def log_postprocess(flow):
    CL, CD = flow.get_observations()
    return CL, CD


print_fmt = (
    "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"  # This will format the output
)
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename=f"{output_dir}/step_response.dat",
)

callbacks = [
    log,
    # hgym.utils.io.CheckpointCallback(interval=10, filename=checkpoint),
]

# 4. Simulate the flow
tf = 10.0
dt = 0.1
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization=stabilization,
)
