#!/bin/bash
# PETSc Feature Install Script
# Builds PETSc 3.24.0 with OpenBLAS for Firedrake compatibility
# Uses GCC + OpenMPI from base

set -euo pipefail

# Feature options - the devcontainer CLI injects option values as env vars
# named by concatenating the camelCase option id in uppercase with no
# separator (e.g. "petscVersion" -> PETSCVERSION, not PETSC_VERSION) - read
# from those, or a configured option silently falls back to the default.
PETSC_VERSION="${PETSCVERSION:-v3.24.0}"
WITH_FIREDRAKE="${WITHFIREDRAKE:-true}"

echo "=== PETSc Feature Installation ==="
echo "Version: ${PETSC_VERSION}"
echo "Firedrake compatibility: ${WITH_FIREDRAKE}"

# flex is needed by --download-ptscotch (PTScotch's build system requires it)
# and isn't part of base/Dockerfile.base's apt package list.
apt-get update && apt-get install -y --no-install-recommends flex && rm -rf /var/lib/apt/lists/*

# Use GCC and apt-OpenMPI from base, via explicit paths - bare mpicc/mpicxx/
# mpifort/mpif77 resolve to NVHPC's compilers instead, since HPCX is first
# on PATH (see base/Dockerfile.base's GNU-toolchain library stack comment).
export CC=/usr/bin/mpicc.openmpi
export CXX=/usr/bin/mpicxx.openmpi
export FC=/usr/bin/mpifort.openmpi
export F77=/usr/bin/mpif77.openmpi
export MPIEXEC=/usr/bin/mpiexec.openmpi

# Base library paths (GNU-toolchain stack, ABI-compatible with the CC/CXX/FC/F77 above)
export HDF5_HOME="/opt/maia_deps_gnu/hdf5-1.14.5"
export FFTW_HOME="/opt/maia_deps_gnu/fftw-3.3.10"

# The container's default LD_LIBRARY_PATH puts HPCX's libmpi.so.40 (OpenMPI
# 5.x/ompi5) and the NVHPC-toolchain HDF5/FFTW builds (/opt/maia_deps, no
# "_gnu") ahead of their GNU-toolchain counterparts - a binary compiled
# against the CC/CXX/FC/F77 and HDF5_HOME/FFTW_HOME above will still
# dynamically resolve libmpi.so.40/libfftw3_mpi.so.3 to the wrong
# (ABI-incompatible) copy at runtime unless the correct dirs come first
# (same trap as maia-cpu's postCreateCommand.sh; `ldd` on the resulting
# libpetsc.so confirms this without the override - libmpi resolved to
# HPCX's copy, and libfftw3_mpi to the NVHPC one, despite --with-fftw-dir
# correctly pointing at the GNU one at configure time - DT_RUNPATH from the
# linker is searched *after* LD_LIBRARY_PATH, so it loses to whatever's
# already in LD_LIBRARY_PATH). The versioned MPI runtime libs (libmpi.so.40
# etc) live directly under /usr/lib/x86_64-linux-gnu, NOT the openmpi/lib
# subdir (that subdir only has unversioned link-time symlinks).
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${HDF5_HOME}/lib:${FFTW_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Set as a real shell variable (not just a literal ./configure argument
# below) since it's referenced later in this script (build/verify steps);
# with `set -u` those references would otherwise fail as unbound.
export PETSC_ARCH="arch-firedrake-default"

# Install OpenBLAS first (needed for PETSc)
echo "Building OpenBLAS..."
cd /tmp
git clone --branch v0.3.27 --depth 1 https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=HASWELL USE_OPENMP=1
make install PREFIX=/opt/openblas-0.3.27
export OPENBLAS_HOME="/opt/openblas-0.3.27"
cd /tmp && rm -rf OpenBLAS

# Download and build PETSc
PETSC_DIR="/opt/petsc"
echo "Cloning PETSc ${PETSC_VERSION}..."
if [[ -d "${PETSC_DIR}" ]]; then
    echo "PETSc directory exists, updating..."
    cd "${PETSC_DIR}"
    git fetch origin
else
    # gitlab.com's git-smart-http sometimes rejects this base image's git
    # client with "RPC failed; HTTP 403" / "expected flush after ref
    # listing" - reproduced on two unrelated networks, plain HTTPS GETs to
    # gitlab.com work fine, so it's specifically a protocol v2 negotiation
    # issue, not a real block. protocol.version=0 works around it.
    if ! git clone https://gitlab.com/petsc/petsc.git "${PETSC_DIR}"; then
        echo "Clone failed, retrying with protocol.version=0 (see comment above)..."
        rm -rf "${PETSC_DIR}"
        git -c protocol.version=0 clone https://gitlab.com/petsc/petsc.git "${PETSC_DIR}"
    fi
    cd "${PETSC_DIR}"
fi

git checkout "${PETSC_VERSION}"

# Configure PETSc
echo "Configuring PETSc..."
CONFIGURE_OPTS=(
    --with-cc="${CC}"
    --with-cxx="${CXX}"
    --with-fc="${FC}"
    --with-mpiexec="${MPIEXEC}"
    --with-c2html=0
    --with-debugging=0
    --with-fortran-bindings=0
    --with-shared-libraries=1
    --with-strict-petscerrorcode
    --COPTFLAGS="-O3 -march=native -mtune=native -fPIC"
    --CXXOPTFLAGS="-O3 -march=native -mtune=native -fPIC"
    --FOPTFLAGS="-O3 -march=native -mtune=native -fPIC"
    --download-bison
    --with-fftw=1
    --with-hdf5=1
    --with-hwloc=1
    --with-zlib=1
    --with-blas-lapack-dir="${OPENBLAS_HOME}"
    --with-hdf5-dir="${HDF5_HOME}"
    --with-fftw-dir="${FFTW_HOME}"
    --download-metis
    --download-mumps
    --download-netcdf
    --download-ptscotch
    --download-scalapack
    --download-suitesparse
    --download-superlu_dist
    --download-hypre
    PETSC_ARCH="${PETSC_ARCH}"
)

# Add Firedrake-specific options if enabled
if [[ "${WITH_FIREDRAKE}" == "true" ]]; then
    echo "Adding Firedrake-compatible options..."
    # Firedrake needs these additional packages
    CONFIGURE_OPTS+=(
        --download-parmetis
        --download-chaco
    )
fi

./configure "${CONFIGURE_OPTS[@]}"

# Build PETSc
echo "Building PETSc..."
make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}" all -j$(nproc)

# Verify build
if [[ -f "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so" ]]; then
    echo "=== PETSc build successful ==="
    echo "PETSc lib: ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so"
else
    echo "ERROR: PETSc build failed"
    exit 1
fi

# Verify libpetsc.so actually resolves its MPI/FFTW deps to the
# GNU-toolchain copies (see the LD_LIBRARY_PATH comment above) rather than
# silently falling back to HPCX/NVHPC ones - `ldd` reflects the resolution
# that will actually happen at runtime, which is a stronger check than "it
# built" given how easy this is to get silently wrong.
echo "Verifying libpetsc.so links against the correct (GNU-toolchain) MPI/FFTW..."
LDD_OUT=$(ldd "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so")
echo "${LDD_OUT}" | grep -iE "mpi|fftw"
if echo "${LDD_OUT}" | grep -qE "(hpcx|comm_libs).*libmpi\.so|/opt/maia_deps/fftw"; then
    echo "ERROR: libpetsc.so resolved MPI/FFTW to the NVHPC-toolchain copy - ABI mismatch"
    exit 1
fi
echo "libpetsc.so MPI/FFTW resolution OK"

echo "=== PETSc Feature Installation Complete ==="
echo "PETSc_DIR: ${PETSC_DIR}"
echo "PETSC_ARCH: arch-firedrake-default"
echo "PETSC_LIB: ${PETSC_DIR}/arch-firedrake-default/lib"
echo ""
echo "For Firedrake, set these environment variables:"
echo "  export PETSC_DIR=${PETSC_DIR}"
echo "  export PETSC_ARCH=arch-firedrake-default"