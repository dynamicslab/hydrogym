import argparse

import firedrake as fd
import numpy as np
from cyl_common import base_checkpoint, evec_checkpoint, flow

import hydrogym.firedrake as hgym

chk_dir, chk_file = evec_checkpoint.split("/")


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


parser = argparse.ArgumentParser(
    description="Stability analysis of the Re=100 cylinder wake."
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
    "--adjoint",
    action="store_true",
    dest="adjoint",
    default=True,
    help="Compute the adjoint modes along with the direct modes.",
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
if __name__ == "__main__":
    args = parser.parse_args()
    m = args.krylov_dim
    sigma = args.sigma

    hgym.print("|-------------------------------------------------------|")
    hgym.print("| Linear stability analysis of the Re=100 cylinder wake |")
    hgym.print("|-------------------------------------------------------|")
    hgym.print(f"Krylov dimension:     {m}")
    hgym.print(f"Spectral shift:       {sigma}")
    hgym.print(f"Adjoint:              {args.adjoint}")
    hgym.print(f"Krylov-Schur restart: {args.schur}")
    hgym.print("")

    # # TODO: add config to either load the base flow or re-run the steady solver
    # if args.base_flow:
    #     hgym.print(f"Loading base flow from checkpoint {args.base_flow}...")
    #     flow.load_checkpoint(args.base_flow)

    # else:
    #     raise NotImplementedError("TODO: re-compute base flow")

    # exit(0)

    flow.load_checkpoint(base_checkpoint)
    fn_space = flow.mixed_space

    # TODO: Make this a LinearOperator-type object with __matmul__ and transpose
    arnoldi = hgym.utils.make_st_iterator(flow, sigma=sigma, adjoint=False)

    tol = 1e-10
    if args.schur:
        n_evals = 24
        evals, evecs_real, evecs_imag, residuals = hgym.utils.eig_ks(
            arnoldi,
            m=m,
            tol=tol,
            n_evals=n_evals,
            sort=lambda x: x.real > -1.0,
        )
    else:
        evals, evecs_real, evecs_imag, residuals = hgym.utils.eig_arnoldi(arnoldi, m=m)

    np.save("/".join([chk_dir, "st_shift_evals_unfilt"]), evals)

    # Remove duplicate eigenvalues (only needed when sigma.imag != 0)
    if sigma.imag != 0:
        keep = _filter_evals(evals, sigma, residuals, tol)

    else:
        # Just filter by non-converged
        keep = residuals < tol

    hgym.print(f"Converged: {np.sum(keep)} / {len(keep)}")

    evals = evals[keep]
    evals = np.where(evals.imag > 0, evals + sigma, evals + sigma.conjugate())

    residuals = residuals[keep]
    evecs_real = [evecs_real[i] for i in range(len(keep)) if keep[i]]
    evecs_imag = [evecs_imag[i] for i in range(len(keep)) if keep[i]]

    n_save = min(len(evals), 32)
    hgym.print(f"Arnoldi eigenvalues: {evals[:n_save]}")

    # Save checkpoints
    chk_path = "/".join([chk_dir, f"st_shift_{chk_file}"])
    eval_data = np.zeros((n_save, 3), dtype=np.float64)
    eval_data[:, 0] = evals[:n_save].real
    eval_data[:, 1] = evals[:n_save].imag
    eval_data[:, 2] = residuals[:n_save]
    np.save("/".join([chk_dir, "st_shift_evals"]), eval_data)

    with fd.CheckpointFile(chk_path, "w") as chk:
        for i in range(n_save):
            chk.save_mesh(flow.mesh)
            evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(evecs_real[i])
            evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(evecs_imag[i])

    # TODO:
    # [x] Add a general function for defining the Navier-Stokes residual
    # [x] General function for the real-shifted and complex-shifted operators
    # [x] Class structure for Arnoldi methods
    # [x] Krylov-Schur for shift-inverse iteration
    # [] Clean up and merge

    hgym.print(
        "NOTE: If there is a warning following this, ignore it.  It is raised by PETSc "
        "CLI argument handling and not argparse. It does not indicate that any CLI "
        "arguments are ignored by this script."
    )
