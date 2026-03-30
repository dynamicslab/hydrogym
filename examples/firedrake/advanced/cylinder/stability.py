#!/usr/bin/env python3
"""
Cylinder Flow - Linear Stability Analysis

Compute eigenvalues and eigenmodes of the linearized Navier-Stokes operator around
a steady base flow. This reveals the frequencies and growth rates of flow instabilities.

Physical Interpretation:
    - Eigenvalue real part > 0: Flow is unstable (vortex shedding grows)
    - Eigenvalue imaginary part: Oscillation frequency (related to Strouhal number)

Method:
    - Arnoldi iteration to find leading eigenvalues/eigenvectors
    - Optional Krylov-Schur restart for large problems
    - Computes both direct and adjoint modes (for control applications)

Usage:
    python stability.py                    # Default: Re=100, medium mesh
    python stability.py --reynolds 150     # Higher Reynolds number
    python stability.py --krylov-dim 200   # More Arnoldi vectors
    python stability.py --base-flow base.h5  # Use existing steady solution

Outputs:
    - evals.npy: Converged eigenvalues (real, imag, residual)
    - evecs.h5: Direct eigenvectors (velocity/pressure modes)
    - adj_evecs.h5: Adjoint eigenvectors (for control design)
"""

import argparse
import os

import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

# Parse command-line arguments for flexible configuration
parser = argparse.ArgumentParser(description="Stability analysis of the Re=100 cylinder wake.")
parser.add_argument(
    "--mesh",
    default="medium",
    type=str,
    help='Identifier for the mesh resolution. Options: ["medium", "fine"]',
)
parser.add_argument(
    "--reynolds",
    default=100.0,
    type=float,
    help="Reynolds number of the flow",
)
parser.add_argument(
    "--krylov-dim",
    default=100,
    type=int,
    help="Dimension of the Krylov subspace (number of Arnoldi vectors)",
)
parser.add_argument(
    "--tol",
    default=1e-10,
    type=float,
    help="Tolerance to use for determining converged eigenvalues.",
)
parser.add_argument(
    "--schur",
    action="store_true",
    dest="schur",
    default=False,
    help="Use Krylov-Schur iteration to restart the Arnoldi process.",
)
parser.add_argument(
    "--no-adjoint",
    action="store_true",
    default=False,
    help="Skip computing the adjoint modes.",
)
parser.add_argument(
    "--sigma",
    default=0.0,
    type=complex,
    help="Shift for the shift-invert Arnoldi method.",
)
parser.add_argument(
    "--base-flow",
    type=str,
    help="Path to the HDF5 checkpoint containing the base flow.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="eig_output",
    help="Directory in which output files will be stored.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Extract configuration from command-line arguments
    mesh = args.mesh
    m = args.krylov_dim  # Number of Arnoldi vectors (controls eigenvalue accuracy)
    sigma = args.sigma  # Spectral shift for shift-invert (targets eigenvalues near sigma)
    tol = args.tol  # Convergence tolerance for eigenvalues
    Re = args.reynolds

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use P2-P1 Taylor-Hood elements (inf-sup stable, no stabilization needed)
    velocity_order = 2
    stabilization = "none"

    # Create flow configuration (mesh, boundary conditions, etc.)
    flow = hgym.Cylinder(Re=Re, mesh=mesh, velocity_order=velocity_order)

    hgym.print("|------------------------------------------------|")
    hgym.print("| Linear stability analysis of the cylinder wake |")
    hgym.print("|------------------------------------------------|")
    hgym.print(f"Reynolds number:       {Re}")
    hgym.print(f"Krylov dimension:      {m}")
    hgym.print(f"Spectral shift:        {sigma}")
    hgym.print(f"Include adjoint modes: {not args.no_adjoint}")
    hgym.print(f"Krylov-Schur restart:  {args.schur}")
    hgym.print("")

    # Option 1: Load pre-computed base flow from checkpoint
    if args.base_flow:
        hgym.print(f"Loading base flow from checkpoint {args.base_flow}...")
        flow.load_checkpoint(args.base_flow)

    # Option 2: Compute base flow using Newton solver
    else:
        hgym.print("Solving the steady-state problem for the cylinder base flow...")

        steady_solver = hgym.NewtonSolver(flow, stabilization=stabilization, solver_parameters={"snes_monitor": None})

        # Reynolds ramping: Start from lower Re for better convergence
        if Re > 50:
            hgym.print("Solving steady-state problem at Re=50...")
            flow.Re.assign(50)
            steady_solver.solve()

        # Solve at target Reynolds number
        hgym.print(f"Solving steady-state problem at target Reynolds number Re={Re}...")
        flow.Re.assign(Re)
        steady_solver.solve()
        CL, CD = flow.compute_forces()
        hgym.print(f"Lift: {CL:0.3e}, Drag: {CD:0.3e}")
        flow.save_checkpoint(f"{args.output_dir}/base.h5")

    # ========================================================================
    # Compute direct modes (standard eigenvalue problem)
    # ========================================================================
    # Direct modes represent physical flow disturbances:
    #   q' = q_mode * exp(lambda*t)  where lambda = eigenvalue
    hgym.print("Computing direct modes...")
    dir_results = hgym.utils.stability_analysis(flow, sigma, m, tol, schur_restart=args.schur, adjoint=False)
    np.save(f"{output_dir}/raw_evals", dir_results.raw_evals)

    # Save converged eigenvalues with residuals
    evals = dir_results.evals
    eval_data = np.zeros((len(evals), 3), dtype=np.float64)
    eval_data[:, 0] = evals.real  # Growth rate (positive = unstable)
    eval_data[:, 1] = evals.imag  # Frequency (2*pi*f = omega)
    eval_data[:, 2] = dir_results.residuals  # Convergence check
    np.save(f"{output_dir}/evals", eval_data)

    # Save eigenvectors (velocity/pressure modes) for visualization
    with fd.CheckpointFile(f"{output_dir}/evecs.h5", "w") as chk:
        for i in range(len(evals)):
            chk.save_mesh(flow.mesh)
            # Complex eigenvector = real part + i * imaginary part
            dir_results.evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(dir_results.evecs_real[i])
            dir_results.evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(dir_results.evecs_imag[i])

    # ========================================================================
    # Compute adjoint modes (transpose eigenvalue problem)
    # ========================================================================
    # Adjoint modes are used for:
    #   - Optimal control design (determines sensor/actuator placement)
    #   - Receptivity analysis (how sensitive is flow to forcing)
    #   - Proper Orthogonal Decomposition (POD) modes
    if not args.no_adjoint:
        hgym.print("Computing adjoint modes...")
        adj_results = hgym.utils.stability_analysis(flow, sigma, m, tol, adjoint=True)

    # Save adjoint eigenvalues (same as direct, but computed independently)
    evals = adj_results.evals
    eval_data = np.zeros((len(evals), 3), dtype=np.float64)
    eval_data[:, 0] = evals.real
    eval_data[:, 1] = evals.imag
    eval_data[:, 2] = adj_results.residuals
    np.save(f"{output_dir}/adj_evals", eval_data)

    with fd.CheckpointFile(f"{output_dir}/adj_evecs.h5", "w") as chk:
        for i in range(len(evals)):
            chk.save_mesh(flow.mesh)
            adj_results.evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(adj_results.evecs_real[i])
            adj_results.evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(adj_results.evecs_imag[i])

    hgym.print(
        "NOTE: If there is a warning following this, ignore it.  It is raised by PETSc "
        "CLI argument handling and not argparse. It does not indicate that any CLI "
        "arguments are ignored by this script."
    )
