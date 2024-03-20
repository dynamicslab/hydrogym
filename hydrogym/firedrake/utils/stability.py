from typing import NamedTuple

import firedrake as fd
import numpy as np

from .eig import eig
from .utils import print as parprint

__all__ = ["stability_analysis"]


class StabilityResults(NamedTuple):
    evals: np.ndarray
    evecs_real: list[fd.Function]
    evecs_imag: list[fd.Function]
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
            parprint(f"Not converged: {evals[i]}: {residuals[i]}")
            keep[i] = False
            continue
        near_zero = (
            abs(evals[i].imag - sigma.imag) < tol
            or abs(evals[i].imag + sigma.imag) < tol
        )
        if near_zero:
            parprint(f"Found near zero: {evals[i]}")
            keep[i] = False
            continue
        for j in range(i + 1, len(evals)):
            real_close = abs(evals[i].real - evals[j].real) < tol
            shift_imag_close = (
                abs(evals[i].imag - evals[j].imag - 2 * sigma.imag) < tol
                or abs(evals[i].imag - evals[j].imag + 2 * sigma.imag) < tol
            )
            if real_close and shift_imag_close:
                parprint(f"Found duplicate: {evals[i]} and {evals[j]}")
                # Keep whichever has the smaller residual
                if residuals[i] < residuals[j]:
                    keep[j] = False
                else:
                    keep[i] = False
    return keep


def stability_analysis(
    flow,
    sigma=0.0,
    krylov_dim=100,
    tol=1e-6,
    adjoint=False,
    schur_restart=False,
    schur_delta=0.1,
    n_evals=12,
):
    """Linear stability analysis of the flow.

    Args:
        flow: The FlowConfiguration to analyze.
        sigma:
            The shift for the shift-invert Arnoldi method. The algorithm will converge
            most quickly if the shift is close to the eigenvalues of interest.
        m: The dimension of the Krylov subspace (number of Arnoldi vectors).
        tol: Tolerance to use for determining converged eigenvalues.
        adjoint: If True, compute the adjoint modes along with the direct modes.
        schur_restart:
            If True, use Krylov-Schur iteration to restart the Arnoldi process.
        schur_delta:
            The stability margin to use when determining which Schur eigenvalues
            to keep in Krylov-Schur iteration. Ignored if `schur_restart` is False.
        n_evals:
            The number of eigenvalues to converge in order to terminate Krylov-Schur
            iteration. Ignored if `schur_restart` is False.
    """

    evals, evecs_real, evecs_imag, residuals = eig(
        flow,
        sigma=sigma,
        adjoint=adjoint,
        schur_restart=schur_restart,
        krylov_dim=krylov_dim,
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

    parprint(f"{np.sum(keep)} / {len(keep)} converged:")

    evals = evals[keep]
    evals = np.where(evals.imag > 0, evals + sigma, evals + sigma.conjugate())

    residuals = residuals[keep]
    evecs_real = [evecs_real[i] for i in range(len(keep)) if keep[i]]
    evecs_imag = [evecs_imag[i] for i in range(len(keep)) if keep[i]]

    n_save = min(len(evals), 32)
    for _eval in evals[:n_save]:
        parprint(_eval)

    return StabilityResults(
        evals=evals,
        evecs_real=evecs_real,
        evecs_imag=evecs_imag,
        residuals=residuals,
        raw_evals=raw_evals,
    )
