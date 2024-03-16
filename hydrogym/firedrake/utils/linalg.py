from typing import Iterable

import firedrake as fd
import numpy as np
import ufl
from firedrake import logging
from scipy import sparse

# Type suggestions
from hydrogym.firedrake import FlowConfig

from .modred_interface import Snapshot, vec_handle_mean


def adjoint(L):
    args = L.arguments()
    L_adj = ufl.adjoint(L, reordered_arguments=(args[0], args[1]))
    return L_adj


EPS_PARAMETERS = {
    "eps_gen_non_hermitian": None,
    "eps_target": "0",
    "eps_type": "krylovschur",
    "eps_largest_real": True,
    "st_type": "sinvert",
    "st_pc_factor_shift_type": "NONZERO",
    "eps_tol": 1e-10,
}


def eig(A, M, num_eigenvalues=1, sigma=None, options={}):
    """
    Compute eigendecomposition of the matrix pencil `(A, M)` using SLEPc,
        where `A` is the dynamics matrix and `M` is the mass matrix.

    The default behavior is to use a shift-invert transformation to avoid inverting `M`,
    which is singular in the case of the incompressible Navier-Stokes equations.
    Ultimately this computes `[A - sigma * M]^-1` instead of `M^-1*A`, making it somewhat sensitive
    to the choice of the shift `sigma`.  This should be chosen near eigenvalues of interest
    """
    import numpy as np
    from firedrake import COMM_WORLD
    from firedrake.petsc import PETSc
    from slepc4py import SLEPc

    assert (
        PETSc.ScalarType == np.complex128
    ), "Complex PETSc configuration required for stability analysis"

    assert not (
        (sigma is not None) and ("eps_target" in options)
    ), "Shift value specified twice: use either `sigma` or `options['eps_target']` (behavior is the same)"

    slepc_options = EPS_PARAMETERS.copy()
    slepc_options.update(options)
    if sigma is not None:
        slepc_options.update({"eps_target": f"{sigma.real}+{sigma.imag}i"})

    # SLEPc Setup
    opts = PETSc.Options()
    for key, val in slepc_options.items():
        opts.setValue(key, val)

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setDimensions(num_eigenvalues)
    es.setOperators(A, M)
    es.setFromOptions()
    es.solve()

    nconv = es.getConverged()
    vr, vi = A.getVecs()

    evals = np.array([es.getEigenpair(i, vr, vi) for i in range(nconv)])
    return evals, es


def define_inner_product(mass_matrix):
    if isinstance(mass_matrix, sparse.csr_matrix):
        M = mass_matrix
    else:
        # Assume filename
        if mass_matrix[-4:] != ".npz":
            mass_matrix += ".npz"
        M = sparse.load_npz(mass_matrix)

    def inner_product(u, v):
        return np.dot(u.conj(), M @ v)

    return inner_product


def project(basis_handles, data_handles, mass_matrix):
    inner_product = define_inner_product(mass_matrix)
    vec_space = VectorSpaceHandles(inner_product)
    coeffs = vec_space.compute_inner_product_array(basis_handles, data_handles).T
    return coeffs
