#!/usr/bin/env python3
"""
Cylinder Flow - Linear Time-Invariant (LTI) System Construction

Demonstrates extracting linearized flow operators for control-theoretic analysis.
This script computes the base flow and control influence vector, which are
building blocks for state-space models of the form:

    x' = A*x + B*u  (state evolution)
    y  = C*x + D*u  (output measurement)

where:
    - x: State vector (velocity/pressure perturbations)
    - u: Control input (scalar actuation)
    - y: Output (e.g., lift/drag coefficients)
    - A: Linearized Navier-Stokes operator (Jacobian around base flow)
    - B: Control influence operator (how actuation affects state)
    - C: Observation operator (how state maps to measurements)
    - D: Feedthrough operator (direct control-to-output)

Current Implementation:
    - Computes steady base flow qB using Newton solver
    - Computes control influence qC by solving linearized BCs problem
    - Future: Add full A, B, C, D matrix assembly for reduced-order modeling

Applications:
    - Model-based control (LQR, LQG, H-infinity)
    - Reduced-order modeling (POD, DMD, balanced truncation)
    - Frequency-domain analysis (Bode plots, Nyquist stability)
    - Optimal sensor/actuator placement

Mathematical Background:
    Linearize Navier-Stokes around base flow qB:
        N(q) = 0  (steady Navier-Stokes)
    Perturb: q = qB + ε*q'
    Linearize: J(qB)*q' = forcing
    where J = dN/dq is the Jacobian (Frechet derivative)

Usage:
    python lti_system.py

Outputs:
    - output/lin_fields.h5: Base flow (qB) and control influence (qC)
    - Console: Steady-state lift/drag coefficients

Note: This is a work-in-progress example demonstrating linearization concepts.
      Full LTI operator construction requires additional LinearOperator wrappers.
"""

import os

import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor
from ufl import dx, inner

import hydrogym.firedrake as hgym

show_plots = False  # Set to True to display vorticity plots
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create flow configuration with rotary cylinder actuation
# RotaryCylinder uses tangential velocity BC (cylinder rotation)
flow = hgym.RotaryCylinder(Re=100, mesh="medium", use_HF_data_manager=False)

# ========================================================================
# Step 1: Compute steady base flow qB
# ========================================================================
# This is the equilibrium around which we linearize
# At Re=100, this is an unstable steady state (saddle point)
hgym.print("Computing steady base flow...")
steady_solver = hgym.NewtonSolver(flow)
qB = steady_solver.solve()  # qB = (velocity, pressure) base flow
qB.rename("qB")

# Verify base flow by computing forces
CL_base, CD_base = flow.compute_forces(qB)
hgym.print(f"Base flow forces: CL = {CL_base:.6f}, CD = {CD_base:.6f}")

# Optional: Visualize base flow vorticity
if show_plots:
    vort = flow.vorticity(qB.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)
    ax.set_title("Base Flow Vorticity")

# ========================================================================
# Step 2: Compute control influence field qC (the "B" operator effect)
# ========================================================================
# This represents the flow response to unit actuation:
#   J * qC = 0  with inhomogeneous boundary conditions (u_control = 1.0)
#
# Physical interpretation: qC shows how cylinder rotation affects the flow
# For control design: B*u ≈ qC * u (when linearized around qB)

hgym.print("Computing control influence field (B operator)...")

# Get Jacobian of steady Navier-Stokes around base flow
F = steady_solver.steady_form(fd.split(qB))  # Nonlinear residual F(qB) = 0
J = fd.derivative(F, qB)  # Linearization: J = dF/dq evaluated at qB

# Set up linearized boundary conditions with unit control input
flow.linearize_bcs(function_spaces=flow.function_spaces(mixed=True))
flow.set_control([1.0])  # Unit actuation input
bcs = flow.collect_bcs()  # Collect all boundary conditions

# Solve linear system: J * qC = 0 with inhomogeneous BCs
# This gives the steady flow field caused by unit actuation
qC = fd.Function(flow.mixed_space, name="qC")
v, s = fd.TestFunctions(flow.mixed_space)  # Test functions for velocity, pressure
zero = inner(fd.Constant((0.0, 0.0)), v) * dx  # Zero forcing on RHS
fd.solve(J == zero, qC, bcs=bcs)

hgym.print("Control influence field computed successfully")

# Optional: Visualize control influence field
if show_plots:
    vort = flow.vorticity(qC.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)
    ax.set_title("Control Influence Field (qC) Vorticity")
    plt.show()

# ========================================================================
# Step 3: Save linearized fields for future use
# ========================================================================
# These fields can be used for:
#   - Reduced-order modeling (POD basis construction)
#   - Optimal control design (LQR, MPC)
#   - Sensitivity analysis
#   - Validation of linear approximations

hgym.print(f"Saving linearized fields to {output_dir}/lin_fields.h5...")
with fd.CheckpointFile(f"{output_dir}/lin_fields.h5", "w") as chk:
    chk.save_mesh(flow.mesh)
    chk.save_function(qB)  # Base flow (steady state)
    chk.save_function(qC)  # Control influence (B operator effect)

hgym.print("LTI system components saved successfully!")
hgym.print("\nNext steps:")
hgym.print("  - Construct full A operator (linearized dynamics)")
hgym.print("  - Add C operator (observation mapping)")
hgym.print("  - Build reduced-order model (ROM)")
hgym.print("  - Design model-based controllers")
