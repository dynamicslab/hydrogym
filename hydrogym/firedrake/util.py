import firedrake as fd
from ufl import dx, inner


def mass_matrix(flow, backend="petsc"):
    (u, _) = fd.TrialFunctions(flow.mixed_space)
    (v, _) = fd.TestFunctions(flow.mixed_space)
    M = inner(u, v) * dx

    if backend == "scipy":
        from ..utils import petsc_to_scipy

        M = petsc_to_scipy(fd.assemble(M).petscmat)
    return M


def save_mass_matrix(flow, filename):
    from scipy.sparse import save_npz

    assert fd.COMM_WORLD.size == 1, "Not supported in parallel"

    M = mass_matrix(flow, backend="scipy")

    if filename[-4:] != ".npz":
        filename += ".npz"
    save_npz(filename, M)
