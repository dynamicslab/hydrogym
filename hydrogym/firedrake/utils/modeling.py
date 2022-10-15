import warnings
from functools import wraps

import firedrake as fd
import numpy as np
from scipy import sparse
from ufl import dx, inner

from hydrogym.firedrake import FlowConfig, NewtonSolver

from . import linalg
from .utils import get_array


# Ignore deprecation warnings in projection
def ignore_deprecation_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as _:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            response = f(*args, **kwargs)
        return response

    return inner


def petsc_to_scipy(petsc_mat):
    """Convert the PETSc matrix to a scipy CSR matrix"""
    indptr, indices, data = petsc_mat.getValuesCSR()
    scipy_mat = sparse.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
    return scipy_mat


def system_to_scipy(sys):
    """Convert the LTI system tuple (A, M, B) to scipy/numpy arrays"""
    A = petsc_to_scipy(sys[0])
    M = petsc_to_scipy(sys[1])
    if len(sys) == 2:
        return A, M
    B = np.vstack(sys[2]).T
    return A, M, B


def mass_matrix(flow, backend="petsc"):
    (u, _) = fd.TrialFunctions(flow.mixed_space)
    (v, _) = fd.TestFunctions(flow.mixed_space)
    M = inner(u, v) * dx

    if backend == "scipy":
        M = petsc_to_scipy(fd.assemble(M).petscmat)
    return M


def save_mass_matrix(flow, filename):
    from scipy.sparse import save_npz

    assert fd.COMM_WORLD.size == 1, "Not supported in parallel"

    M = mass_matrix(flow, backend="scipy")

    if filename[-4:] != ".npz":
        filename += ".npz"
    save_npz(filename, M)


@ignore_deprecation_warnings
def snapshots_to_numpy(flow, filename, save_prefix, m):
    """
    Load from CheckpointFile in `filename` and project to the mesh in `flow`
    """
    from firedrake import logging

    with fd.CheckpointFile(filename, "r") as file:
        mesh = file.load_mesh("mesh")
        for idx in range(m):
            logging.log(logging.DEBUG, f"Converting snapshot {idx+1}/{m}")
            q = file.load_function(mesh, "q", idx=idx)  # Load on different mesh
            u, p = q.split()

            # Project to new mesh
            flow.u.assign(fd.project(u, flow.velocity_space))
            flow.p.assign(fd.project(p, flow.pressure_space))

            # Save with new mesh
            np.save(f"{save_prefix}{idx}.npy", get_array(flow.q))


def linearize_dynamics(flow: FlowConfig, qB: fd.Function, adjoint: bool = False):
    solver = NewtonSolver(flow)
    F = solver.steady_form(q=qB)
    L = -fd.derivative(F, qB)
    if adjoint:
        return linalg.adjoint(L)
    else:
        return L


def linearize(
    flow: FlowConfig, qB: fd.Function, adjoint: bool = False, backend: str = "petsc"
):
    assert backend in [
        "petsc",
        "scipy",
    ], "Backend not recognized: use `petsc` or `scipy`"

    A_form = linearize_dynamics(flow, qB, adjoint=adjoint)
    M_form = mass_matrix(flow)
    flow.linearize_bcs()
    A = fd.assemble(A_form, bcs=flow.collect_bcs()).petscmat  # Dynamics matrix
    M = fd.assemble(M_form, bcs=flow.collect_bcs()).petscmat  # Mass matrix

    sys = A, M
    if backend == "scipy":
        sys = system_to_scipy(sys)
    return sys
