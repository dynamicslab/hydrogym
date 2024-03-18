import dataclasses
from typing import Callable

import firedrake as fd
import numpy as np
from scipy import linalg

from .utils import print as parprint

__all__ = ["eig"]


def sorted_eig(H, which, inverse=True):
    evals, evecs = linalg.eig(H)
    if inverse:
        evals = 1 / evals  # Undo spectral shift

    x = {
        "lm": -abs(evals),
        "sm": abs(evals),
        "lr": -evals.real,
        "sr": evals.real,
        "li": -evals.imag,
        "si": evals.imag,
    }[which]
    sort_idx = np.argsort(x)

    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    return evals, evecs


def _arnoldi_factorization(
    iterate,
    inner_product,
    v0,
    m=100,
    print_evals=True,
    which="lr",
    inverse=True,
    restart=None,
):
    """Run Arnoldi iteration.

    Note that `which`, and `inverse` here are only used for printing the eigenvalues.
    Otherwise these are not used as part of the Arnoldi factorization. If
    `print_evals=False` these will be ignored.  Also, the printed eigenvalues do not
    include any spectral transformation.

    Given functions `iterate(f, v)` to calculate a matrix-vector product `f = A @ v`
    (or inverse iteration `f = A^{-1} @ v`, shifted matrix pencil iteration, etc.),
    an inner product `inner_product(u, v)`, and a starting vector `v0`, this
    function computes an m-stage Arnoldi decomposition of the matrix `A` (or matrix
    pencil (A, M) for mass matrix M) and returns the resulting Krylov vectors and
    upper Hessenberg matrix.
    """
    if restart is None:
        V = [v0.copy(deepcopy=True).assign(0.0) for _ in range(m + 1)]
        H = np.zeros((m, m))
        start_idx = 1
        V[0].assign(v0)
    else:
        V, H, start_idx = restart

    f = v0.copy(deepcopy=True)
    for j in range(start_idx, m + 1):
        # Apply iteration A @ f = M @ v
        # This is the iteration f = (A^{-1} @ M) @ v
        # which is inverse to M @ f = A @ v
        # Stores the result in `f`
        iterate(f, V[j - 1])

        # Gram-Schmidt
        h = np.zeros(j)
        for i in range(j):
            h[i] = inner_product(V[i], f)
            f.assign(f - V[i] * h[i])

        # Fill in the upper Hessenberg matrix
        H[:j, j - 1] = h

        # Norm of the orthogonal residual
        beta = np.sqrt(inner_product(f, f))

        if j < m:
            H[j, j - 1] = beta

        # Normalize the orthogonal residual for use as a new Krylov vector
        V[j].assign(f / beta)

        # DEBUG: Print the eigenvalues
        if print_evals:
            ritz_vals, ritz_vecs = sorted_eig(H[:j, :j], which, inverse=inverse)
            res = abs(beta * ritz_vecs[-1])
            parprint("\n*************************************")
            parprint(f"****** Arnoldi iteration {j} ********")
            parprint("*************************************")
            parprint("|        evals        |  residuals  |")
            parprint("|---------------------|-------------|")
            for i in range(min(10, j)):
                parprint(f"| {ritz_vals[i]:.2e} |  {res[i]:.2e}  |")

    return V, H, beta


@dataclasses.dataclass
class ArnoldiIterator:
    iterate: Callable
    inner_product: Callable
    random_vec: Callable = None
    print_evals: bool = True
    which: str = "lr"
    inverse: bool = True

    def __call__(self, v0, m, restart=None):
        return _arnoldi_factorization(
            self.iterate,
            self.inner_product,
            v0,
            m=m,
            restart=restart,
            print_evals=self.print_evals,
            which=self.which,
            inverse=self.inverse,
        )


def eig(
    arnoldi,
    v0=None,
    schur_restart=False,
    n_evals=10,
    m=100,
    sort=None,
    tol=1e-10,
    delta=0.1,
    maxiter=None,
    rng_seed=None,
):
    if v0 is None:
        v0 = arnoldi.random_vec(rng_seed)

    # TODO: Handle sort functions better - use a lookup of pre-defined
    # functions for "lr", "sm", etc.
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return abs(x) > 1.0 - delta

    fn_space = v0.function_space()

    restart = None
    is_converged = False
    k = 0  # Iteration count (for logging)
    while not is_converged:
        V, H, beta = arnoldi(v0, m, restart=restart)

        # Check for convergence based on Arnoldi residuals
        evals, ritz_vecs = sorted_eig(H, which="lr", inverse=True)
        residuals = abs(beta * ritz_vecs[-1])

        converged = []
        for i in range(m):
            if residuals[i] < tol:
                converged.append((evals[i], ritz_vecs[:, i]))

        parprint(f"Iteration {k}: {len(converged)} converged")

        # If enough eigenvalues have converged, return them and exit
        if (
            (not schur_restart)
            or (len(converged) >= n_evals)
            or (maxiter is not None and k >= maxiter)
        ):
            is_converged = True
            parprint(f"Converged: {len(converged)} eigenvalues")

            p = len(converged)
            evals = np.zeros(p, dtype=np.complex128)
            evecs_real = [fd.Function(fn_space) for _ in range(p)]
            evecs_imag = [fd.Function(fn_space) for _ in range(p)]
            for i, (mu, y) in enumerate(converged):
                evals[i] = mu
                evecs_real[i].assign(sum(V[j] * y[j].real for j in range(m)))
                evecs_imag[i].assign(sum(V[j] * y[j].imag for j in range(m)))
            return evals, evecs_real, evecs_imag, residuals

        # Schur decomposition
        S, Q, p = linalg.schur(H, sort=sort)
        parprint(f"   Schur form: {p} retained eigenvalues")

        # Keep the "wanted" part of the Schur form. These will be the eigenvalues
        # that are closest to the unit circle (least stable).
        S[p:, :p] = 0
        S[:p, p:] = 0
        S[p:, p:] = 0

        # Re-order the Krylov basis.  Since Q is real, these fields are also real.
        Vp = [fd.Function(fn_space) for _ in range(m + 1)]
        v0.assign(sum(V[j] * Q[j, p] for j in range(m)))
        for i in range(p):
            Vp[i].assign(sum(V[j] * Q[j, i] for j in range(m)))

        # Update the matrix with the "b" vector for residuals
        b = np.zeros(m)
        b[-1] = beta
        b = Q.T @ b
        S[p, :p] = b[:p]
        Vp[p].assign(V[-1])  # Restart from the last Krylov basis vector

        # p += 1  #??

        # Restart with the wanted part of the Krylov basis
        restart = (Vp, S, p)
        k += 1
