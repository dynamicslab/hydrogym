import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
from scipy import sparse

__all__ = ["print", "is_rank_zero", "petsc_to_scipy", "system_to_scipy", "set_from_array", "get_array"]

## Parallel utility functions
def print(s):
    """Print only from first process"""
    PETSc.Sys.Print(s)

def is_rank_zero():
    """Is this the first process?"""
    return fd.COMM_WORLD.rank == 0

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

def set_from_array(func, array):
    with func.dat.vec as vec:
        vec.setArray(array)

def get_array(func):
    with func.dat.vec_ro as vec:
        array = vec.getArray()
    return array