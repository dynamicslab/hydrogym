#!/bin/bash
# NekRS Feature Post-Create Command
#
# Builds NekRS from the in-repo third_party/nekrs submodule (pinned to
# v26.0) into third_party/nekrs/install (= $NEKRS_HOME, see
# devcontainer-feature.json containerEnv), then the HydroGym MPMD driver
# examples/nekrs/mpmd/nekrs-mpmd into $NEKRS_HOME/bin (both via
# build_nekrs.sh, shared with the ROCm image), then a hydrogym[nekrs] venv at
# /opt/venvs/nekrs.
#
# Toolchain: HPC-X Open MPI 5 (CUDA-aware, first on PATH - see
# base/Dockerfile.base) with the GNU compilers underneath it: NekRS's JIT
# (OCCA_CXX) and its Nek5000 Fortran parts are written for GCC/gfortran, the
# HPC-X wrappers default to nvc/nvc++/nvfortran. mpi4py in the venv is built from source against the
# same wrappers, so controller and solver ranks share one libmpi in an MPMD
# launch (a mismatch hangs or crashes at MPI_Init).
#
# Everything NekRS needs (OCCA, HYPRE, gslib, Nek5000, parRSB) is vendored in
# its source tree, so this step needs no network apart from the pip installs.
# Idempotent: skips the NekRS build if $NEKRS_HOME/bin/nekrs exists (delete
# third_party/nekrs/install to force a rebuild).

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/nekrs.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
CUDAARCH="${CUDAARCH:-80;86}"
GPUMPI="${GPUMPI:-true}"
BUILDTYPE="${BUILDTYPE:-RelWithDebInfo}"

echo "=== NekRS Post-Create Setup ==="

NEKRS_SRC=/workspace/third_party/nekrs
NEKRS_HOME="${NEKRS_HOME:-${NEKRS_SRC}/install}"
FEATURE_DIR=/workspace/.devcontainer/features/nekrs

if [[ ! -f "${NEKRS_SRC}/CMakeLists.txt" ]]; then
    echo "WARNING: NekRS source not found at ${NEKRS_SRC}."
    echo "Run 'git submodule update --init third_party/nekrs' in the hydrogym repo on the host, then rebuild/reopen the container."
    exit 0
fi

if [[ -x "${NEKRS_HOME}/bin/nekrs" && -x "${NEKRS_HOME}/bin/nekrs-mpmd" ]]; then
    echo "NekRS already installed at ${NEKRS_HOME} - skipping build."
else
    NEKRS_SRC="${NEKRS_SRC}" NEKRS_HOME="${NEKRS_HOME}" MPMD_SRC=/workspace/examples/nekrs/mpmd \
    BACKEND=CUDA GPUARCH="${CUDAARCH}" GPUMPI="${GPUMPI}" BUILDTYPE="${BUILDTYPE}" \
    MPI_BIN="${HPCX_ROOT}/bin" MPI_CC=gcc MPI_CXX=g++ MPI_FC=gfortran \
        bash "${FEATURE_DIR}/build_nekrs.sh"
fi

bash /workspace/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/nekrs nekrs

# mpi4py may have come in as a prebuilt wheel or built against another MPI -
# rebuild it against HPC-X unless HPC-X's libmpi is the one it
# actually loads.
# shellcheck disable=SC1091
source /opt/venvs/nekrs/bin/activate
if ! python -c "from mpi4py import MPI; import sys; sys.exit('${HPCX_ROOT}/lib/libmpi' not in open('/proc/self/maps').read())" 2>/dev/null; then
    echo "Rebuilding mpi4py against HPC-X..."
    MPICC="${NEKRS_HOME}/mpi/bin/mpicc" pip install --no-cache-dir --no-binary mpi4py --force-reinstall --no-deps "mpi4py>=4.0.0"
fi
python -c "from mpi4py import MPI; print('mpi4py:', MPI.Get_library_version().splitlines()[0])"
deactivate

echo ""
echo "=== NekRS Post-Create Complete ==="
echo "NEKRS_HOME=${NEKRS_HOME}  (nekrs, nekrs-mpmd in \$NEKRS_HOME/bin)"
echo "Python: /opt/venvs/nekrs  -  example: cd examples/nekrs/channel && python ../fetch_case.py && ./run.sh --policy zero --steps 20"
