import warnings
from functools import wraps

import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
from scipy import sparse
from ufl import dx, inner


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


# Ignore deprecation warnings in projection
def ignore_deprecation_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as _:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            response = f(*args, **kwargs)
        return response

    return inner


# Parallel utility functions
def print(s):
    """Print only from first process"""
    PETSc.Sys.Print(s)


def is_rank_zero():
    """Is this the first process?"""
    return fd.COMM_WORLD.rank == 0


def set_from_array(func, array):
    with func.dat.vec as vec:
        vec.setArray(array)


def get_array(func):
    with func.dat.vec_ro as vec:
        array = vec.getArray()
    return array


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


def white_noise(n_samples, fs, cutoff):
    """Generate band-limited white noise"""
    from scipy import signal

    rng = fd.Generator(fd.PCG64())
    noise = rng.standard_normal(n_samples)

    # Set up butterworth filter
    sos = signal.butter(N=4, Wn=cutoff, btype="lp", fs=fs, output="sos")
    filt = signal.sosfilt(sos, noise)
    return filt
