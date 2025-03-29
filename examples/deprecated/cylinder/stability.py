import argparse
import os

import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

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

    mesh = args.mesh
    m = args.krylov_dim
    sigma = args.sigma
    tol = args.tol
    Re = args.reynolds

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    velocity_order = 2
    stabilization = "none"

    flow = hgym.Cylinder(
        Re=Re,
        mesh=mesh,
        velocity_order=velocity_order,
    )

    hgym.print("|------------------------------------------------|")
    hgym.print("| Linear stability analysis of the cylinder wake |")
    hgym.print("|------------------------------------------------|")
    hgym.print(f"Reynolds number:       {Re}")
    hgym.print(f"Krylov dimension:      {m}")
    hgym.print(f"Spectral shift:        {sigma}")
    hgym.print(f"Include adjoint modes: {not args.no_adjoint}")
    hgym.print(f"Krylov-Schur restart:  {args.schur}")
    hgym.print("")

    if args.base_flow:
        hgym.print(f"Loading base flow from checkpoint {args.base_flow}...")
        flow.load_checkpoint(args.base_flow)

    else:
        hgym.print("Solving the steady-state problem for the cylinder base flow...")

        steady_solver = hgym.NewtonSolver(flow, stabilization=stabilization, solver_parameters={"snes_monitor": None})
        if Re > 50:
            hgym.print("Solving steady-state problem at Re=50...")
            flow.Re.assign(50)
            steady_solver.solve()

        hgym.print(f"Solving steady-state problem at target Reynolds number Re={Re}...")
        flow.Re.assign(Re)
        steady_solver.solve()
        CL, CD = flow.compute_forces()
        hgym.print(f"Lift: {CL:0.3e}, Drag: {CD:0.3e}")
        flow.save_checkpoint(f"{args.output_dir}/base.h5")

    hgym.print("Computing direct modes...")
    dir_results = hgym.utils.stability_analysis(flow, sigma, m, tol, schur_restart=args.schur, adjoint=False)
    np.save(f"{output_dir}/raw_evals", dir_results.raw_evals)

    # Save checkpoints
    evals = dir_results.evals
    eval_data = np.zeros((len(evals), 3), dtype=np.float64)
    eval_data[:, 0] = evals.real
    eval_data[:, 1] = evals.imag
    eval_data[:, 2] = dir_results.residuals
    np.save(f"{output_dir}/evals", eval_data)

    with fd.CheckpointFile(f"{output_dir}/evecs.h5", "w") as chk:
        for i in range(len(evals)):
            chk.save_mesh(flow.mesh)
            dir_results.evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(dir_results.evecs_real[i])
            dir_results.evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(dir_results.evecs_imag[i])

    if not args.no_adjoint:
        hgym.print("Computing adjoint modes...")
        adj_results = hgym.utils.stability_analysis(flow, sigma, m, tol, adjoint=True)

        # Save checkpoints
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
