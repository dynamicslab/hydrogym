#!/usr/bin/env python3
"""
Backward-Facing Step - Step Input Control

Demonstrates open-loop step input control on backward-facing step flow.
Applies a step change in actuation at t=50 and observes flow response.

Usage:
    python step-control.py

Physical setup:
    - Backward-facing step with sudden expansion
    - Re = 600 (default)
    - Actuation: Step input at t=50 (off → on)
    - Observations: Kinetic energy, turbulent kinetic energy
"""

import firedrake as fd
import psutil
import hydrogym.firedrake as hgym

Re = 600
mesh_resolution = "medium"
output_dir = f"./step_control_Re{Re}_{mesh_resolution}_output"
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

flow = hgym.Step(Re=Re, mesh=mesh_resolution, velocity_order=2)

# Simulation parameters
Tf = 100  # Total time
t_switch = 50.0  # Time to switch control on
dt = 0.01


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = flow.evaluate_objective()
    CFL = flow.max_cfl(dt)
    mem_usage = psutil.virtual_memory().percent
    return [CFL, KE, TKE, mem_usage]


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


# Simple step controller: off until t_switch, then constant actuation
def controller(t, obs):
    """Step input: 0 before t_switch, constant after"""
    if t < t_switch:
        return [0.0]  # No control
    else:
        return [0.5]  # Constant actuation


print_fmt = "t: {0:0.2f}\t\tCFL: {1:0.2f}\t\tKE: {2:0.6e}\t\tTKE: {3:0.6e}\t\tMem: {4:0.1f}"
callbacks = [
    hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=500, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

hgym.print("=" * 70)
hgym.print("Backward-Facing Step: Step Input Control")
hgym.print("=" * 70)
hgym.print(f"Control switches on at t={t_switch}")
hgym.print("")

hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization="gls",
)

hgym.print("\n" + "=" * 70)
hgym.print("Simulation complete!")
hgym.print(f"Output saved to {output_dir}/")
hgym.print("=" * 70)
