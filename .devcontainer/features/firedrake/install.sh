#!/bin/bash
# Firedrake + HydroGym CPU Feature Install Script
# Creates venv with --system-site-packages, installs Firedrake + HydroGym
# Requires PETSc feature to be installed first

set -euo pipefail

# Feature options
# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. "hydrogymExtras" -> HYDROGYMEXTRAS, not HYDROGYM_EXTRAS) - read
# from those, or a configured option silently falls back to the default.
FIREDRAKE_COMMIT="${FIREDRAKECOMMIT:-ef4c1bf9e6caa9c70dc75413f633c8f97551a4cc}"
HYDROGYM_EXTRAS="${HYDROGYMEXTRAS:-maia,firedrake,nek}"

echo "=== Firedrake + HydroGym CPU Feature Installation ==="
echo "Firedrake commit: ${FIREDRAKE_COMMIT}"
echo "HydroGym extras: ${HYDROGYM_EXTRAS}"

# Persisted for postCreateCommand.sh, which installs HydroGym itself once
# /workspace is live (see note near the end of this script).
mkdir -p /opt/maia-feature-config
cat > /opt/maia-feature-config/firedrake.env <<EOF
HYDROGYM_EXTRAS=${HYDROGYM_EXTRAS}
EOF

# PETSc from petsc feature
export PETSC_DIR="/opt/petsc"
export PETSC_ARCH="arch-firedrake-default"

if [[ ! -f "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so" ]]; then
    echo "ERROR: PETSc not found at ${PETSC_DIR}/${PETSC_ARCH}"
    echo "Please install the 'petsc' feature first"
    exit 1
fi

# Create venv with --system-site-packages to reuse base mpi4py, h5py
VENV_DIR="/opt/venvs/firedrake"
echo "Creating venv at ${VENV_DIR} with --system-site-packages..."
uv venv "${VENV_DIR}" --system-site-packages --seed

# Activate venv
source "${VENV_DIR}/bin/activate"

# mpi4py (installed below) compiles a C extension against whatever `mpicc`
# it finds - bare `mpicc` resolves to NVHPC's compiler via HPCX (first on
# PATH, see base/Dockerfile.base), which would silently build mpi4py
# against the wrong (ABI-incompatible) MPI compared to petsc4py/h5py/
# Firedrake later in this script, which all explicitly use apt-OpenMPI.
# This must be set before the first pip install that touches MPI, not just
# before the h5py one further down. FC/F77 matter too: requirements-build.txt
# pulls in libsupermesh, which has a Fortran component built via CMake -
# without FC set, CMake's Fortran detection finds bare `mpif90` on PATH
# (HPCX's nvfortran wrapper), which rejects gfortran-only flags like
# -ffree-line-length-none with "nvfortran-Error-Unknown switch".
export CC=/usr/bin/mpicc.openmpi
export CXX=/usr/bin/mpicxx.openmpi
export FC=/usr/bin/mpifort.openmpi
export F77=/usr/bin/mpif77.openmpi

# Same runtime resolution trap as the petsc feature (see its install.sh):
# the container's default LD_LIBRARY_PATH puts HPCX's libmpi.so.40 and the
# NVHPC-toolchain HDF5/FFTW builds ahead of their GNU-toolchain
# counterparts, so compiled extensions here (mpi4py, h5py, petsc4py) would
# dynamically resolve to the wrong ones at import time even when correctly
# compiled against apt-OpenMPI/GNU HDF5.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}"

# Upgrade pip and install build dependencies
# NOTE: h5py is deliberately NOT installed here - it's built later with
# --no-build-isolation --no-binary=h5py against our GNU-toolchain HDF5 (with
# HDF5_MPI=ON for parallel I/O). Installing it here first would satisfy pip
# with a prebuilt manylinux wheel (bundling its own vendored, non-MPI HDF5),
# and pip's default "already satisfied" behavior would then silently skip
# the later, correctly-configured rebuild entirely - h5py would end up
# working but without MPI support and against the wrong HDF5 version.
pip install --upgrade pip setuptools wheel
# petsc4py's Cython sources under this pinned PETSC_VERSION predate the fix
# for Cython 3.1's stricter pointer-array index-type checking - building
# with an unpinned (latest) Cython fails with "Invalid index type 'int'" in
# PC.pyx (RT_Pi_mat[i] = ...). Pin below 3.1; requirements-build.txt below
# only needs Cython>=3.0, so this satisfies both.
pip install "cython<3.1" numpy mpi4py

# Set PETSc environment for firedrake-configure
export PETSC_DIR="${PETSC_DIR}"
export PETSC_ARCH="${PETSC_ARCH}"

# Get PETSc configuration from firedrake-configure
echo "Setting up Firedrake build environment..."
cd /workspace
if [[ ! -f "firedrake-configure" ]]; then
    # We'll download it as part of Firedrake clone
    :
fi

# Clone Firedrake at specific commit
FIREDRAKE_DIR="/opt/firedrake"
echo "Cloning Firedrake..."
if [[ -d "${FIREDRAKE_DIR}" ]]; then
    echo "Firedrake directory exists, updating..."
    cd "${FIREDRAKE_DIR}"
    git fetch origin
else
    git clone https://github.com/firedrakeproject/firedrake.git "${FIREDRAKE_DIR}"
    cd "${FIREDRAKE_DIR}"
fi

git checkout "${FIREDRAKE_COMMIT}"
git submodule update --init --recursive

# Copy firedrake-configure script
cp "${FIREDRAKE_DIR}/scripts/firedrake-configure" /workspace/firedrake-configure
chmod +x /workspace/firedrake-configure

# Get PETSc environment from firedrake-configure
echo "Getting PETSc environment from firedrake-configure..."
eval "$(python3 /workspace/firedrake-configure --no-package-manager --show-env)"

# firedrake-configure computes PETSC_DIR as "$(pwd)/petsc" (see its
# prepare_environment_vars()) - relative to whatever directory happens to
# be current when it's invoked (here, .../opt/firedrake, from the git clone
# above), NOT the actual PETSc install location. The eval above silently
# clobbers our correct PETSC_DIR/PETSC_ARCH (set from the petsc feature)
# with that bogus, nonexistent path - restore the real values here so the
# petsc4py install path below (and everything after it) is actually right.
export PETSC_DIR="/opt/petsc"
export PETSC_ARCH="arch-firedrake-default"

# requirements-build.txt lists Firedrake's actual build-time dependencies
# (meson-python, scikit_build_core, petsctools, pybind11, libsupermesh,
# etc, per its pyproject.toml [build-system]). --no-build-isolation below
# means pip will NOT auto-install these into an isolated env, so they must
# already be present or the Firedrake build fails on a missing backend.
echo "Installing Firedrake build dependencies..."
pip install -r "${FIREDRAKE_DIR}/requirements-build.txt"

# Patch petsc4py for setuptools<72 compatibility
PETSC4PY_PXD="${PETSC_DIR}/src/binding/petsc4py/src/petsc4py/PETSc.pxd"
if [[ -f "${PETSC4PY_PXD}" ]] && ! grep -q "CHKERRMPI(int)" "${PETSC4PY_PXD}"; then
    echo "Patching petsc4py..."
    echo 'cdef PetscErrorCode CHKERRMPI(int) except PETSC_ERR_PYTHON nogil' >> "${PETSC4PY_PXD}"
fi

# Install h5py with MPI support (no build isolation)
echo "Installing h5py with MPI..."
# Bare mpicc resolves to NVHPC's nvc (HPCX is first on PATH), which isn't
# ABI-compatible with the GNU-toolchain HDF5 below (see base/Dockerfile.base).
export CC=/usr/bin/mpicc.openmpi
export HDF5_MPI=ON
export HDF5_DIR="${HDF5_HOME:-/opt/maia_deps_gnu/hdf5-1.14.5}"
pip install --no-build-isolation --no-binary=h5py h5py

# Install petsc4py
echo "Installing petsc4py..."
# Downgrade setuptools right before petsc4py, not earlier: requirements-
# build.txt (just installed above) pins setuptools>=77.0.3, which would
# silently re-upgrade an earlier downgrade and reintroduce the failure this
# is working around - petsc4py's confpetsc.py calls distutils' execute()
# with a `dry_run` kwarg that newer setuptools/distutils removed, failing
# with "TypeError: execute() got an unexpected keyword argument 'dry_run'".
pip install "setuptools<72"
pip install --no-build-isolation "${PETSC_DIR}/src/binding/petsc4py"

# Upgrade setuptools back
pip install --upgrade setuptools

# Install Firedrake
echo "Installing Firedrake..."
# FIREDRAKE_DIR is already absolute (/opt/firedrake) - a "./" prefix here
# would make this a *relative* path (`.//opt/firedrake` normalizes to
# `opt/firedrake`, not `/opt/firedrake`), which combined with cwd already
# being ${FIREDRAKE_DIR} at this point resolves to the nonexistent
# /opt/firedrake/opt/firedrake and fails the install.
pip install --no-build-isolation "${FIREDRAKE_DIR}[check,docs]"

# HydroGym itself is installed in postCreateCommand instead of here: this
# devcontainer bind-mounts the actual hydrogym repo to /workspace
# at container runtime (see devcontainer.json "workspaceMount"), but that
# mount isn't attached yet during this build-time install step - cloning a
# second copy here would just get shadowed by the real mount later and
# waste build time.

# Verify installations
echo "Verifying installations..."
# cwd is still ${FIREDRAKE_DIR} (/opt/firedrake) here, which contains
# "firedrake/" and "pyop2/" subdirectories that are themselves the raw,
# uncompiled source packages (no built Cython extensions) - `python -c`
# adds cwd to sys.path first, so it shadows the properly pip-installed
# packages and fails with a circular-import-looking ImportError. cd away
# so the real installed packages resolve.
cd /tmp
# This dev build of firedrake has no __version__ attribute (same situation
# as hydrogym - see the ensure_hydrogym.sh gotcha), use importlib.metadata.
python -c "import firedrake, importlib.metadata; print(f'Firedrake: {importlib.metadata.version(\"firedrake\")}')"

# Deactivate
deactivate

# Bake the same LD_LIBRARY_PATH fix into the venv's own (uv-generated)
# bin/activate, not just activate_firedrake.sh below: a plain
# `source bin/activate` would otherwise resolve mpi4py/h5py/petsc4py's
# libmpi.so.40/libhdf5/libfftw3_mpi to the HPCX/NVHPC copies (see the
# LD_LIBRARY_PATH comment near the top of this script) - this "works" by
# accident today only because OpenMPI promises libmpi.so ABI compatibility
# across versions, not because it's actually correct, so fix it at the
# source rather than relying on that.
cat >> "${VENV_DIR}/bin/activate" << 'EOF'

# Prepend GNU-toolchain MPI/HDF5/FFTW dirs so compiled extensions in this
# venv (mpi4py, h5py, petsc4py) resolve their runtime libs correctly
# instead of falling back to HPCX/NVHPC's ABI-incompatible copies - see
# the firedrake feature's install.sh for the full explanation.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}"
EOF

# Create activation helper script
cat > "${VENV_DIR}/activate_firedrake.sh" << 'EOF'
#!/bin/bash
# Source this to activate the Firedrake environment with PETSc
source /opt/venvs/firedrake/bin/activate
export PETSC_DIR=/opt/petsc
export PETSC_ARCH=arch-firedrake-default
export PATH=${PETSC_DIR}/${PETSC_ARCH}/bin:${PATH}
# /usr/lib/x86_64-linux-gnu and the GNU-toolchain HDF5/FFTW dirs must come
# before HPCX/NVHPC's in LD_LIBRARY_PATH, or mpi4py/h5py/petsc4py resolve
# libmpi.so.40 (and libfftw3_mpi/libhdf5) to an ABI-incompatible copy at
# import time (see the petsc and firedrake features' install.sh for detail).
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${PETSC_DIR}/${PETSC_ARCH}/lib:${LD_LIBRARY_PATH}
echo "Firedrake environment activated with PETSc"
EOF
chmod +x "${VENV_DIR}/activate_firedrake.sh"

echo "=== Firedrake + HydroGym CPU Feature Installation Complete ==="
echo "Venv: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/activate_firedrake.sh"
echo ""
echo "To use:"
echo "  source ${VENV_DIR}/activate_firedrake.sh"
echo "  python -c \"import firedrake; import hydrogym; print('OK')\""