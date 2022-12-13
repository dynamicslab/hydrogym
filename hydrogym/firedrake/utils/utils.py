import firedrake as fd
from firedrake.petsc import PETSc


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


def white_noise(n_samples, fs, cutoff):
    """Generate band-limited white noise"""
    from scipy import signal

    rng = fd.Generator(fd.PCG64())
    noise = rng.standard_normal(n_samples)

    # Set up butterworth filter
    sos = signal.butter(N=4, Wn=cutoff, btype="lp", fs=fs, output="sos")
    filt = signal.sosfilt(sos, noise)
    return filt
