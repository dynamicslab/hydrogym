from typing import Iterable

import firedrake as fd
import numpy as np
import ufl
from firedrake import logging
from modred import PODHandles, VectorSpaceHandles
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


def pod(
    flow: FlowConfig,
    snapshot_handles: Iterable[Snapshot],
    r: int,
    mass_matrix,
    decomp_indices=None,
    remove_mean=True,
    mean_dest="mean",
    atol=1e-13,
    rtol=None,
    max_vecs_per_node=None,
    verbosity=1,
    output_dir=".",
    modes_dest="pod",
    eigvals_dest="eigvals.dat",
    pvd_dest=None,
    coeffs_dest="coeffs.dat",
    field_name="q",
):
    """
    Args:
        ``flow``: ``FlowConfig`` for the POD (used for weighted inner product)

        ``snapshots``: List of Snapshot handles

        ``r``: Number of modes to compute

    Kwargs:
        ``decomp_indices``: Indices to use in method of snapshots (defaults to ``range(r)``)

    NOTE: could actually take the "flow" dependence out if we replaced the PVD with a postprocessing callback...
    """
    if fd.COMM_WORLD.size > 1:
        raise NotImplementedError("Not yet supported in parallel")

    # Compute actual POD
    if decomp_indices is None:
        decomp_indices = range(r)

    inner_product = define_inner_product(mass_matrix)

    if remove_mean:
        base_vec_handle = Snapshot(f"{output_dir}/{mean_dest}")
        base_vec_handle.put(vec_handle_mean(snapshot_handles))

        # Redefine snapshots with mean subtraction
        snapshot_handles = [
            Snapshot(snap.filename, base_vec_handle=base_vec_handle)
            for snap in snapshot_handles
        ]
        logging.log(logging.DEBUG, "Mean subtracted")

    logging.log(logging.DEBUG, "Computing POD")
    POD = PODHandles(
        inner_product=inner_product,
        max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity,
    )
    POD.compute_decomp(snapshot_handles, atol=atol, rtol=rtol)

    # Vector handles for storing snapshots
    mode_handles = [
        Snapshot(filename=f"{output_dir}/{modes_dest}{i}") for i in range(r)
    ]
    POD.compute_modes(range(r), mode_handles)

    POD.put_eigvals(f"{output_dir}/{eigvals_dest}")

    # Save for visualization
    if pvd_dest is not None:
        pvd = fd.File(f"{output_dir}/{pvd_dest}", "w")
        for (i, mode) in enumerate(mode_handles):
            u, p = mode.get().as_function().split()
            pvd.write(u, p, flow.vorticity(u))

    # Compute temporal coefficients
    coeffs = POD.compute_proj_coeffs().T  # If all snapshots used for POD
    np.savetxt(f"{output_dir}/{coeffs_dest}", coeffs, fmt="%0.6f", delimiter="\t")

    return coeffs, mode_handles


def project(basis_handles, data_handles, mass_matrix):
    inner_product = define_inner_product(mass_matrix)
    vec_space = VectorSpaceHandles(inner_product)
    coeffs = vec_space.compute_inner_product_array(basis_handles, data_handles).T
    return coeffs
