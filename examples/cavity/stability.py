"""
To get the leading mode, run with:

```
mpiexec -np 20 python stability.py --output-dir eig_output --sigma 1.0+11j
```
"""
import argparse
import os

# from cyl_common import base_checkpoint, evec_checkpoint, flow
from typing import NamedTuple

import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym


class StabilityResults(NamedTuple):
    evals: np.ndarray
    evecs_real: np.ndarray
    evecs_imag: np.ndarray
    residuals: np.ndarray
    raw_evals: np.ndarray


def _filter_evals(evals, sigma, residuals, tol):
    # The shifted eigenvalues have imaginary parts +/- (wi - si), but there
    # are duplicates at +/- (wi + si). We want to discard these duplicates.
    # We can check by doing a positive and negative shift by si and checking
    # for duplicates.
    keep = np.ones_like(evals, dtype=bool)
    for i in range(len(evals)):
        if not keep[i]:
            continue
        if residuals[i] > tol:
            hgym.print(f"Not converged: {evals[i]}: {residuals[i]}")
            keep[i] = False
            continue
        near_zero = (
            abs(evals[i].imag - sigma.imag) < tol
            or abs(evals[i].imag + sigma.imag) < tol
        )
        if near_zero:
            hgym.print(f"Found near zero: {evals[i]}")
            keep[i] = False
            continue
        for j in range(i + 1, len(evals)):
            real_close = abs(evals[i].real - evals[j].real) < tol
            shift_imag_close = (
                abs(evals[i].imag - evals[j].imag - 2 * sigma.imag) < tol
                or abs(evals[i].imag - evals[j].imag + 2 * sigma.imag) < tol
            )
            if real_close and shift_imag_close:
                hgym.print(f"Found duplicate: {evals[i]} and {evals[j]}")
                # Keep whichever has the smaller residual
                if residuals[i] < residuals[j]:
                    keep[j] = False
                else:
                    keep[i] = False
    return keep


def stability_analysis(
    flow,
    sigma=0.0,
    m=100,
    tol=1e-6,
    adjoint=False,
    schur_restart=False,
    schur_delta=0.1,
):

    n_evals = 12
    evals, evecs_real, evecs_imag, residuals = hgym.utils.eig(
        flow,
        sigma=sigma,
        adjoint=adjoint,
        schur_restart=schur_restart,
        m=m,
        tol=tol,
        n_evals=n_evals,
        sort=lambda x: x.real > -schur_delta,
    )

    raw_evals = evals.copy()

    # Remove duplicate eigenvalues (only needed when sigma is complex)
    if sigma.imag != 0:
        keep = _filter_evals(evals, sigma, residuals, tol)

    else:
        # Just filter by non-converged
        keep = residuals < tol

    hgym.print(f"{np.sum(keep)} / {len(keep)} converged:")

    evals = evals[keep]
    evals = np.where(evals.imag > 0, evals + sigma, evals + sigma.conjugate())

    residuals = residuals[keep]
    evecs_real = [evecs_real[i] for i in range(len(keep)) if keep[i]]
    evecs_imag = [evecs_imag[i] for i in range(len(keep)) if keep[i]]

    n_save = min(len(evals), 32)
    for _eval in evals[:n_save]:
        hgym.print(_eval)

    return StabilityResults(
        evals=evals,
        evecs_real=evecs_real,
        evecs_imag=evecs_imag,
        residuals=residuals,
        raw_evals=raw_evals,
    )


parser = argparse.ArgumentParser(
    description="Stability analysis of the open cavity flow."
)
parser.add_argument(
    "--mesh",
    default="fine",
    type=str,
    help="Identifier for the mesh resolution",
)
parser.add_argument(
    "--reynolds",
    default=7500.0,
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

    flow = hgym.Cavity(
        Re=Re,
        mesh=mesh,
        velocity_order=velocity_order,
    )

    dof = flow.mixed_space.dim()

    hgym.print("|--------------------------------------------------|")
    hgym.print("| Linear stability analysis of the open cavity flow |")
    hgym.print("|--------------------------------------------------|")
    hgym.print(f"Reynolds number:       {Re}")
    hgym.print(f"Krylov dimension:      {m}")
    hgym.print(f"Spectral shift:        {sigma}")
    hgym.print(f"Include adjoint modes: {not args.no_adjoint}")
    hgym.print(f"Krylov-Schur restart:  {args.schur}")
    hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")
    hgym.print("")

    if args.base_flow:
        hgym.print(f"Loading base flow from checkpoint {args.base_flow}...")
        flow.load_checkpoint(args.base_flow)

    else:

        # Since this flow is at high Reynolds number we have to
        #    ramp to get the steady state
        hgym.print("Solving the steady-state problem...")

        steady_solver = hgym.NewtonSolver(
            flow, stabilization=stabilization, solver_parameters={"snes_monitor": None}
        )
        Re_init = [500, 1000, 2000, 4000, Re]

        for i, Re in enumerate(Re_init):
            flow.Re.assign(Re)
            hgym.print(f"Steady solve at Re={Re_init[i]}")
            qB = steady_solver.solve()

        flow.save_checkpoint(f"{args.output_dir}/base.h5")

    hgym.print("Computing direct modes...")
    dir_results = stability_analysis(
        flow, sigma, m, tol, schur_restart=args.schur, adjoint=False
    )
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
        adj_results = stability_analysis(flow, sigma, m, tol, adjoint=True)

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
