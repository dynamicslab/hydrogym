#!/bin/bash
# PETSc Feature Post-Create Command
# Verifies PETSc installation and sets up environment for Firedrake

set -euo pipefail

echo "=== PETSc Post-Create Verification ==="

PETSC_DIR="/opt/petsc"
PETSC_ARCH="arch-firedrake-default"

if [[ ! -d "${PETSC_DIR}" ]]; then
    echo "WARNING: PETSc not installed. Run the feature install or rebuild container."
    exit 0
fi

if [[ -f "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so" ]]; then
    echo "PETSc library verified: ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so"
else
    echo "WARNING: PETSc library not found at expected location"
    exit 0
fi

# Confirm libpetsc.so still resolves MPI/FFTW to the GNU-toolchain copies
# (not HPCX/NVHPC ones) at actual runtime - see install.sh's LD_LIBRARY_PATH
# comment for why this is easy to get silently wrong.
LDD_OUT=$(LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}" ldd "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so")
if echo "${LDD_OUT}" | grep -qE "(hpcx|comm_libs).*libmpi\.so|/opt/maia_deps/fftw"; then
    echo "WARNING: libpetsc.so resolves MPI/FFTW to the NVHPC-toolchain copy - ABI mismatch"
else
    echo "PETSc MPI/FFTW resolution OK"
fi

# Set up environment for Firedrake
echo ""
echo "For Firedrake, add to your shell profile or activate script:"
echo "  export PETSC_DIR=${PETSC_DIR}"
echo "  export PETSC_ARCH=${PETSC_ARCH}"
echo "  export PATH=\${PETSC_DIR}/\${PETSC_ARCH}/bin:\${PATH}"
echo "  export LD_LIBRARY_PATH=\${PETSC_DIR}/\${PETSC_ARCH}/lib:\${LD_LIBRARY_PATH}"

echo ""
echo "=== PETSc Post-Create Complete ==="