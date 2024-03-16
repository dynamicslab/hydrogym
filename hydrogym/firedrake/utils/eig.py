import dataclasses
from typing import Callable

import firedrake as fd
import numpy as np
from scipy import linalg

from .utils import print as parprint

__all__ = ["eig_arnoldi"]


def sorted_eig(H, sort, inverse=True):
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
    }[sort]
    sort_idx = np.argsort(x)

    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    return evals, evecs


def _arnoldi_factorization(
    iterate, inner_product, v0, m=100, print_evals=True, sort="lr", inverse=True
):
    """Run Arnoldi iteration.

    Note that `sort`, and `inverse` here are only used for printing the eigenvalues.
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
    V = [v0.copy(deepcopy=True).assign(0.0) for _ in range(m + 1)]
    H = np.zeros((m, m))
    start_idx = 1
    V[0].assign(v0)

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
            ritz_vals, ritz_vecs = sorted_eig(H[:j, :j], sort, inverse=inverse)
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
    sort: str = "lr"
    inverse: bool = True

    def __call__(self, v0, m):
        return _arnoldi_factorization(
            self.iterate,
            self.inner_product,
            v0,
            m=m,
            print_evals=self.print_evals,
            sort=self.sort,
            inverse=self.inverse,
        )


def eig_arnoldi(arnoldi, v0=None, m=100, sort="lr", inverse=True, rng_seed=None):
    if v0 is None:
        v0 = arnoldi.random_vec(rng_seed)

    V, H, beta = arnoldi(v0, m)
    evals, ritz_vecs = sorted_eig(H, sort, inverse=inverse)

    residuals = abs(beta * ritz_vecs[-1])

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(m)]
    evecs_imag = [fd.Function(fn_space) for _ in range(m)]
    for i in range(m):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(m)))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(m)))
    return evals, evecs_real, evecs_imag, residuals
