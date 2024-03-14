import firedrake as fd
import numpy as np
from scipy import linalg

from .utils import print as parprint

__all__ = ["eig_arnoldi"]


def _arnoldi_factorization(step, v0, inner_product, m=100, print_evals=True):
    """Run Arnoldi iteration."""
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
        step(f, V[j - 1])

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
            ritz_vals, ritz_vecs = sorted_eig(H[:j, :j])
            res = abs(beta * ritz_vecs[-1])
            parprint("\n*************************************")
            parprint(f"****** Arnoldi iteration {j} ********")
            parprint("*************************************")
            parprint("|        evals        |  residuals  |")
            parprint("|---------------------|-------------|")
            for i in range(min(10, j)):
                parprint(f"| {ritz_vals[i]:.2e} |  {res[i]:.2e}  |")

    return V, H, beta


def sorted_eig(H, sort=None, inverse=True):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-x.real)

    evals, evecs = linalg.eig(H)
    if inverse:
        evals = 1 / evals  # Undo spectral shift
    sort_idx = sort(evals)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    return evals, evecs


def eig_arnoldi(step, v0, inner_product, m=100, sort=None):
    V, H, _ = _arnoldi_factorization(step, v0, inner_product, m)
    evals, ritz_vecs = sorted_eig(H, sort=sort, inverse=True)

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(m)]
    evecs_imag = [fd.Function(fn_space) for _ in range(m)]
    for i in range(m):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(m)))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(m)))
    return evals, evecs_real, evecs_imag
