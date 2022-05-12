import numpy as np
import ufl
import firedrake as fd
from firedrake import logging
from .modred_interface import vec_handle_mean, Snapshot, SnapshotVector, PODHandles
from .utils import print

# Type suggestions
from ..core import FlowConfig

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
    "eps_tol": 1e-10
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
    assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

    assert not((sigma is not None) and ("eps_target" in options)), \
        "Shift value specified twice: use either `sigma` or `options['eps_target']` (behavior is the same)"

    slepc_options = EPS_PARAMETERS.copy()
    slepc_options.update(options)
    if sigma is not None:
        slepc_options.update({"eps_target": f'{sigma.real}+{sigma.imag}i'})

    ### SLEPc Setup
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

def pod(flow: FlowConfig, snapshot_file: str, r: int,
        decomp_indices=None,
        coeff_indices=None,
        remove_mean=True,
        mean_dest='mean',
        atol=1e-13,
        rtol=None,
        max_vecs_per_node=None,
        verbosity=1,
        output_dir='.',
        modes_dest='pod',
        eigvals_dest='eigvals.dat',
        pvd_dest=None,
        coeffs_dest='coeffs.dat',
        field_name='q',
        ):
    """
    Args:
        ``flow``: ``FlowConfig`` for the POD (used for weighted inner product)

        ``snapshot_file``: Name of ``fd.CheckpointFile`` where snapshots are stored

        ``r``: Number of modes to compute

    Kwargs:
        ``decomp_indices``: Indices to use in method of snapshots (defaults to ``range(r)``)

        ``coeff_indices``: Indices to compute temporal coefficients from (defaults to ``decomp_indices``)


    NOTE: This doesn't work in parallel... "cannot pickle 'petsc4py.PETSc.DMPlex' object".
        I think everything would need to be converted to numpy arrays somewhere... maybe just for the communication step?
    """
    # Compute actual POD
    if decomp_indices is None:
        decomp_indices = range(r)

    decomp_snapshots = [Snapshot(snapshot_file, flow, idx=i, name=field_name) for i in decomp_indices]
    inner_product = lambda u, v: u.dot(v)  # Use weighted inner product from flow

    if remove_mean:
        base_vec_handle = Snapshot(f'{output_dir}/{mean_dest}.h5', flow)
        base_vec_handle.put(vec_handle_mean(decomp_snapshots))

        # Redefine snapshots with mean subtraction
        decomp_snapshots = [Snapshot(snapshot_file, flow, idx=i, name=field_name,
                                base_vec_handle=base_vec_handle) for i in decomp_indices]


    POD = PODHandles(inner_product=inner_product, max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
    POD.compute_decomp(decomp_snapshots, atol=atol, rtol=rtol)

    # Vector handles for storing snapshots
    # mode_handles = [Snapshot(filename=f'{output_dir}/{modes_dest}', flow=flow, idx=i, name=field_name) for i in range(r)]
    mode_handles = [Snapshot(filename=f'{output_dir}/{modes_dest}{i}', flow=flow, idx=None, name=field_name) for i in range(r)]
    POD.compute_modes(range(r), mode_handles)

    POD.put_eigvals(f'{output_dir}/{eigvals_dest}')

    # Save for visualization
    if pvd_dest is not None:
        pvd = fd.File(f'{output_dir}/{pvd_dest}', 'w')
        for (i, mode) in enumerate(mode_handles):
            u, p = mode.get().as_function().split()
            pvd.write(u, p, flow.vorticity(u))

    # Compute temporal coefficients
    if coeff_indices is None:
        coeffs = POD.compute_proj_coeffs()  # If all snapshots used for POD
    else:
        timeseries_snapshots = [Snapshot(snapshot_file, flow, idx=i, name=field_name) for i in coeff_indices]
        coeffs = POD.vec_space.compute_inner_product_array(mode_handles, timeseries_snapshots)  # If different snapshots are used for computing the decomposition and coefficients

    np.savetxt(f'{output_dir}/{coeffs_dest}', coeffs, fmt='%0.6f', delimiter='\t')

    return coeffs, mode_handles

# def pod(flow: FlowConfig, snapshot_file: str, r: int,
#         decomp_indices=None,
#         coeff_indices=None,
#         remove_mean=True,
#         mean_dest='mean',
#         output_dir='.',
#         modes_dest='pod',
#         eigvals_dest='eigvals.dat',
#         pvd_dest=None,
#         coeffs_dest='coeffs.dat',
#         field_name='q',
#         ):
#     """
#     Args:
#         ``flow``: ``FlowConfig`` for the POD (used for weighted inner product)

#         ``snapshot_file``: Name of ``fd.CheckpointFile`` where snapshots are stored

#         ``r``: Number of modes to compute

#     Kwargs:
#         ``decomp_indices``: Indices to use in method of snapshots (defaults to ``range(r)``)

#         ``coeff_indices``: Indices to compute temporal coefficients from (defaults to ``decomp_indices``)
#     """
#     from scipy.linalg import eigh
    
#     if decomp_indices is None:
#         decomp_indices = range(r)

#     U = [Snapshot(snapshot_file, flow, idx=i, name=field_name) for i in decomp_indices]

#     if remove_mean:
#         base_vec_handle = Snapshot(f'{output_dir}/{mean_dest}.h5', flow)
#         base_vec_handle.put(vec_handle_mean(decomp_snapshots))

#         # Redefine snapshots with mean subtraction
#         decomp_snapshots = [Snapshot(snapshot_file, flow, idx=i, name=field_name,
#                                 base_vec_handle=base_vec_handle) for i in decomp_indices]
    
#     # Snapshot POD
#     m = len(U)
#     R = np.zeros((m, m))
#     logging.log(logging.DEBUG, "Computing correlation matrix...")
#     for i in range(m):
#         print("Row {0}/{1}".format(i+1, m))
#         for j in range(i, m):
#             u, v = U[i].get(), U[j].get()
#             R[i, j] = u.dot(v)
#             R[j, i] = R[i, j]
#     R /= m

#     evals, evecs = eigh(R)   # Eigendecomposition of (Hermitian) correlation matrix

#     # Sort eigenvalues and corresponding eigenvectors in descending order
#     sort_idx = np.argsort( evals )[range(m-1, -1, -1)]  # Flip array to descending order, as in SVD
#     evals = abs(evals[sort_idx])  # Get rid of small negative part
#     evecs = evecs[:, sort_idx]

#     np.savetxt(f'{output_dir}/{eigvals_dest}', evals, fmt='%0.6f') # Save singular values

#     # Compute modes
#     mode_handles = [Snapshot(filename=f'{output_dir}/{modes_dest}{i}', flow=flow, idx=None, name=field_name) for i in range(r)]
#     if pvd_dest is not None:
#         pvd = fd.File(f'{output_dir}/{pvd_dest}', 'w')
#     for i in range(r):
#         mode = mode_handles[i]
#         vec = SnapshotVector(flow, fd.Function(flow.mixed_space, name=field_name))
#         for j in range(m):
#             scale = evecs[j, i] / np.sqrt(m*evals[i])
#             u_vec = U[i].get()
#             vec += scale*u_vec

#         # func = vec.as_function()

#         mode.put(vec)

#         if pvd_dest is not None:
#             u, p = mode.get().as_function().split()
#             pvd.write(u, p, flow.vorticity(u))

#     # Compute temporal coefficients a(t)
#     if coeff_indices is None:
#         coeffs = np.diag(evals**0.5).dot(evecs.conj().T)  # If all snapshots used for POD
#     else:
#         U = [Snapshot(snapshot_file, flow, idx=i, name=field_name) for i in coeff_indices]
#         coeffs = np.zeros((len(U), r))
#         for i in range(len(U)):
#             for j in range(r):
#                 u, v = U[i].get(), mode[j].get()
#                 coeffs[i, j] = u.dot(v)

#     np.savetxt(f'{output_dir}/{coeffs_dest}', coeffs, fmt='%0.6f', delimiter='\t')

#     return coeffs, mode_handles