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


if __name__ == "__main__":
    flow.load_checkpoint(base_checkpoint)
    fn_space = flow.mixed_space

    sigma = 0.1 + 0.8j

    arnoldi = hgym.utils.make_st_iterator(flow, sigma=sigma, adjoint=False)

    evals, evecs_real, evecs_imag, residuals = hgym.utils.eig_arnoldi(arnoldi, m=100)

    np.save("/".join([chk_dir, "st_shift_evals_unfilt"]), evals)

    # Remove duplicate eigenvalues
    tol = 1e-2
    keep = _filter_evals(evals, sigma, residuals, tol)

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
            evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(evecs_real[i])
            evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(evecs_imag[i])

    # TODO:
    # [x] Add a general function for defining the Navier-Stokes residual
    # [x] General function for the real-shifted and complex-shifted operators
    # [x] Class structure for Arnoldi methods
    # [] Krylov-Schur for shift-inverse iteration
    # [] Sweep complex plane and test for duplicate eigenvalues
