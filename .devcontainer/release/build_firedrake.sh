#!/bin/bash
# PETSc (CPU), configured as firedrake-configure does, and a Firedrake venv
# with petsc4py, h5py and mpi4py built against the image's MPI. Used by the
# release images; the same recipe as rocm/Dockerfile.hydrogym-rocm.
#
# Environment:
#   MPI_CC, MPI_CXX, MPI_FC  MPI compiler wrappers (GCC/gfortran underneath)
#   PETSC_DIR, PETSC_ARCH    install location (/opt/petsc, arch-firedrake-default)
#   PETSC_VERSION            git tag (v3.25.0)
#   FIREDRAKE_COMMIT         Firedrake commit
#   FIREDRAKE_VENV           venv (/opt/venvs/firedrake)
#   CPU_OPTFLAGS             portable optimisation flags (no -march=native:
#                            the image runs on other machines than it is built on)

set -euo pipefail

: "${MPI_CC:?}" "${MPI_CXX:?}" "${MPI_FC:?}" "${PETSC_DIR:?}" "${PETSC_ARCH:?}"
PETSC_VERSION="${PETSC_VERSION:-v3.25.0}"
FIREDRAKE_COMMIT="${FIREDRAKE_COMMIT:-22c6bc44c79a06d576f49790d1a6d00dc10112ca}"
FIREDRAKE_VENV="${FIREDRAKE_VENV:-/opt/venvs/firedrake}"
CPU_OPTFLAGS="${CPU_OPTFLAGS:--O3 -march=x86-64-v3 -mtune=generic}"
PYTHON="${PYTHON:-/usr/bin/python3}"

# Options from `firedrake-configure --show-petsc-configure-options` at
# FIREDRAKE_COMMIT.
git clone --depth 1 --branch "${PETSC_VERSION}" https://gitlab.com/petsc/petsc.git "${PETSC_DIR}"
cd "${PETSC_DIR}"
"${PYTHON}" ./configure PETSC_ARCH="${PETSC_ARCH}" \
    --with-cc="${MPI_CC}" --with-cxx="${MPI_CXX}" --with-fc="${MPI_FC}" \
    --COPTFLAGS="${CPU_OPTFLAGS}" --CXXOPTFLAGS="${CPU_OPTFLAGS}" --FOPTFLAGS="${CPU_OPTFLAGS}" \
    --with-c2html=0 --with-debugging=0 --with-fortran-bindings=0 --with-shared-libraries=1 \
    --with-strict-petscerrorcode \
    --download-bison --download-fftw --download-hdf5 --download-hwloc --download-hypre \
    --download-metis --download-mumps --download-netcdf --download-pnetcdf --download-ptscotch \
    --download-scalapack --download-suitesparse --download-superlu_dist --download-zlib \
    --with-make-np="${JOBS:-$(nproc)}"
make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}" all
rm -rf "${PETSC_DIR}/.git" "${PETSC_DIR}/${PETSC_ARCH}/externalpackages" "${PETSC_DIR}/${PETSC_ARCH}/obj"

# Order as in features/firedrake/install.sh: petsc4py 3.25 needs Cython < 3.1
# and setuptools < 72 to build, h5py and Firedrake a newer setuptools. Pinned
# to the releases current at FIREDRAKE_COMMIT: firedrake-rtree 2026.3 dropped
# get_lib_filename(), which its setup.py calls.
src="$(mktemp -d)"
git clone https://github.com/firedrakeproject/firedrake.git "${src}/firedrake"
cd "${src}/firedrake"
git checkout "${FIREDRAKE_COMMIT}"
uv venv "${FIREDRAKE_VENV}" --python "${PYTHON}" --seed
# shellcheck disable=SC1091
. "${FIREDRAKE_VENV}/bin/activate"
export CC="${MPI_CC}" CXX="${MPI_CXX}" FC="${MPI_FC}" MPICC="${MPI_CC}" HDF5_MPI=ON HDF5_DIR="${PETSC_DIR}/${PETSC_ARCH}"
# CMake-built dependencies (libsupermesh) may still pick the MPI's own
# wrappers next to mpiexec: make those call the GNU compilers too (no-op for
# MPIs other than Open MPI).
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
printf 'firedrake-rtree==2026.2.0\nlibsupermesh==2026.0\n' > "${src}/constraints.txt"
export PIP_CONSTRAINT="${src}/constraints.txt"
pip install --no-cache-dir -r requirements-build.txt
pip install --no-cache-dir "cython<3.1"
pip install --no-cache-dir --no-binary mpi4py mpi4py
pip install --no-cache-dir --no-build-isolation --no-binary h5py h5py
pip install --no-cache-dir "setuptools<72"
pip install --no-cache-dir --no-build-isolation "${PETSC_DIR}/src/binding/petsc4py"
pip install --no-cache-dir --upgrade setuptools
pip install --no-cache-dir --no-build-isolation .
deactivate
cd / && rm -rf "${src}"

# Firedrake keeps its compiled kernels (PyOP2, TSFC) in $VIRTUAL_ENV/.cache,
# here inside the image: root-owned, or read-only under Apptainer. For a user
# who cannot write there, they go to the user's cache directory
# (XDG_CACHE_HOME, else ~/.cache) under firedrake/; if that cannot be written
# either (e.g. `docker run --user` gives HOME=/), to /tmp/firedrake-cache-uid<uid>,
# which then also serves as XDG_CACHE_HOME (loopy's cache) and holds
# matplotlib's configuration directory. PYOP2_CACHE_DIR and
# FIREDRAKE_TSFC_KERNEL_CACHE_DIR, if set, are kept. A .pth file runs this at
# the start of every Python of the venv, activated or not.
site="$("${FIREDRAKE_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "${site}/_firedrake_user_cache.py" <<'EOF'
"""Firedrake caches for users who cannot write to the venv (release images)."""
import os
import tempfile


def _writable(path):
    """Whether `path` exists as a writable directory or could be created."""
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return os.access(path, os.W_OK | os.X_OK)


def _setup():
    home = os.path.expanduser("~")
    # Firedrake's default (firedrake.configuration.setup_cache_dirs)
    if _writable(os.path.join(os.environ.get("VIRTUAL_ENV") or home, ".cache")):
        return
    user_cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    if _writable(user_cache):
        root = os.path.join(user_cache, "firedrake")
    else:
        root = os.path.join(tempfile.gettempdir(), f"firedrake-cache-uid{os.getuid()}")
        os.environ["XDG_CACHE_HOME"] = root
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(root, "matplotlib"))
    os.environ.setdefault("PYOP2_CACHE_DIR", os.path.join(root, "pyop2"))
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", os.path.join(root, "tsfc"))


_setup()
EOF
echo "import _firedrake_user_cache" > "${site}/_firedrake_user_cache.pth"

OMP_NUM_THREADS=1 "${FIREDRAKE_VENV}/bin/python" -c \
    "import firedrake, importlib.metadata as m; print('firedrake', m.version('firedrake'))"
