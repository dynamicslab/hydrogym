#!/usr/bin/env python3
"""
Cylinder Flow - Step Input Response for System Identification

Applies a step function control input to measure the flow's dynamic response.
This is a fundamental system identification technique used to characterize
linear and nonlinear dynamics for control design.

Physical Setup:
    - Reynolds number Re=40 (subcritical, stable steady state)
    - Control: Jet blowing/suction at ±90° from stagnation point
    - Input: Zero actuation → maximum actuation at t=5s
    - Output: Lift (CL) and drag (CD) coefficients

Purpose:
    - Extract impulse response for linear systems theory
    - Measure time constants (rise time, settling time)
    - Identify control authority (steady-state gain)
    - Validate control hardware/actuators in experiments

Applications:
    - Model-based control design (build transfer function)
    - Sensor/actuator characterization
    - Comparing CFD predictions to experiments
    - Input for reduced-order modeling

Usage:
    python step_input.py

Outputs:
    - output/step_response.dat: Time series of CL, CD
    - Console: Real-time force coefficients

Expected Behavior (Re=40, stable flow):
    - t < 5s: Steady state with CL ≈ 0, CD ≈ constant
    - t = 5s: Step input applied
    - t > 5s: CL, CD transition to new steady state
"""

import os

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Numerical configuration
# Note: P1-P1 elements require stabilization (GLS used here)
stabilization = "gls"  # Galerkin Least Squares stabilization

# Create cylinder flow with jet actuation
# Re=40 is below critical Reynolds number (~47) → stable steady state
flow = hgym.Cylinder(
    Re=40,
    mesh="medium",
    velocity_order=1,  # P1 elements (linear velocity)
    use_HF_data_manager=False,
)

# ========================================================================
# Step 1: Compute steady-state base flow (initial condition)
# ========================================================================
# This finds the equilibrium flow before actuation is applied
steady_solver = hgym.NewtonSolver(
    flow,
    stabilization=stabilization,
)
qB = steady_solver.solve()  # Steady solution (velocity, pressure)


# ========================================================================
# Step 2: Define step input controller
# ========================================================================
# Classic step function: zero for t < 5s, maximum for t ≥ 5s
# This is the standard input for system identification
def controller(t, u):
    """
    Step function control input.

    Args:
      t: Current time
      u: Observation (unused in open-loop step test)

    Returns:
      Control input: 0 before t=5s, MAX_CONTROL after
    """
    return flow.MAX_CONTROL if t > 5.0 else 0.0


# ========================================================================
# Step 3: Configure logging to capture response
# ========================================================================
def log_postprocess(flow):
    """Extract force coefficients at each time step."""
    CL, CD = flow.get_observations()  # Lift and drag
    return CL, CD


# Print and save time series data
print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=2,  # CL and CD
    interval=1,  # Log every time step
    print_fmt=print_fmt,
    filename=f"{output_dir}/step_response.dat",
)

callbacks = [
    log,
    # Optional: Enable checkpointing for restart
    # hgym.utils.io.CheckpointCallback(interval=10, filename=checkpoint),
]

# ========================================================================
# Step 4: Run time-domain simulation
# ========================================================================
# Integrate long enough to capture transient response + new steady state
tf = 10.0  # Total time (5s before step, 5s after step)
dt = 0.1  # Time step

hgym.print("Running step input experiment...")
hgym.print("Control activates at t=5.0s")
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,  # Apply step input
    stabilization=stabilization,
)
hgym.print(f"Step response saved to {output_dir}/step_response.dat")
