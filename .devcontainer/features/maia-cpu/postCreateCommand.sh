#!/bin/bash
# MAIA CPU Feature Post-Create Command
#
# Runs after the container is created, once the host bind mount (see
# devcontainer.json "workspaceMount") has attached /workspace/wipmaiaml.
# This is where the actual configure/build happens (see install.sh for why
# it can't happen at image-build time). Idempotent: skips the build if a
# matching binary already exists.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/maia-cpu.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
BUILD_TYPE="${BUILD_TYPE:-production}"
ENABLE_COMPONENTS="${ENABLE_COMPONENTS:-}"
DISABLE_COMPONENTS="${DISABLE_COMPONENTS:-}"

# The container's default LD_LIBRARY_PATH puts HPCX's libmpi.so ahead of
# apt/GNU OpenMPI's - fine for maia-gpu, but a GNU-toolchain binary linked
# against apt's libmpi_cxx.so.40 will resolve the wrong libmpi.so at runtime
# and fail with "undefined symbol: ompi_mpi_errors_throw_exceptions" (an
# ABI mismatch, not a missing library). Put the matching GNU-toolchain libs
# first for this script's own build/test run, and print the same override
# below for anyone running the binary manually afterward.
# Note: the versioned runtime libs (libmpi.so.40, libmpi_cxx.so.40) live
# directly under /usr/lib/x86_64-linux-gnu, NOT the openmpi/lib subdir
# (that subdir only has unversioned link-time symlinks, used by
# LIBRARY_DIRS below at build time) - HPCX's libmpi.so.40 would otherwise
# get found first via LD_LIBRARY_PATH, ahead of the default system path.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${MAIA_LIB_HOME_GNU:-/opt/maia_deps_gnu}/hdf5-1.14.5/lib:${MAIA_LIB_HOME_GNU:-/opt/maia_deps_gnu}/parallel-netcdf-1.14.0/lib:${MAIA_LIB_HOME_GNU:-/opt/maia_deps_gnu}/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}"

echo "=== MAIA CPU Post-Create Setup ==="

MAIA_DIR="/workspace/wipmaiaml"
BUILD_DIR="${MAIA_DIR}/build_gnu_${BUILD_TYPE}"

# wipmaiaml isn't public yet, so it only shows up here via the host bind
# mount. Once it's public, replace this guard with a `git clone` fallback
# (or clone it in install.sh at build time like the other features do).
if [[ ! -d "${MAIA_DIR}" ]]; then
    echo "WARNING: MAIA source not found at ${MAIA_DIR}."
    echo "Check that the host bind mount in devcontainer.json points at a valid wipmaiaml checkout."
    exit 0
fi

cd "${MAIA_DIR}"

if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
    echo "MAIA CPU binaries already present at ${BUILD_DIR} - skipping build."
else
    # A previous incomplete attempt (e.g. one that failed at the link step)
    # can leave a stale CMakeCache.txt behind pointing at old library paths -
    # CMake reuses cached paths rather than re-reading the host config on a
    # plain re-run, so a partial build dir must not be trusted.
    rm -rf "${BUILD_DIR}"

    echo "Initializing git submodules..."
    git submodule update --init --depth 1 -- include/Eigen include/cantera include/doctest include/hypre include/mfem include/sundials 2>/dev/null || true

    HOST_CONFIG="${MAIA_DIR}/auxiliary/hosts/LocalCPU.cmake"
    echo "Creating GNU host config: ${HOST_CONFIG}"

    # configure.py's host detection (cmake/GetHost.cmake) is purely based on
    # the container's OS hostname pattern-matched against a hardcoded list of
    # known clusters - it does NOT read MAIA_HOST/HOST env vars, contrary to
    # what the manual build docs (CLAUDE.md) might suggest. Since this
    # container's hostname is always "LocalGPU" (set in devcontainer.json's
    # runArgs, shared across every solver feature), auto-detection would
    # always resolve to the NVHPC-only LocalGPU profile regardless of which
    # feature is active. MAIA_HOST_FILE is the actual override (see
    # cmake/Configure.cmake) - it points straight at a host file and skips
    # hostname detection entirely.
    export MAIA_HOST_FILE="${HOST_CONFIG}"

    cat > "${HOST_CONFIG}" << 'EOF'
# LocalCPU - GCC + OpenMPI host configuration for MAIA CPU builds
set(MAIA_HOST "LocalCPU")
set(HOST "LocalCPU")
set(HOST_SUPPORTED_COMPILERS "GNU")
set(HOST_DEFAULT_COMPILER "GNU")

set(HOST_COMPILER_EXE_GNU "/usr/bin/g++")

# Use base-installed libraries
set(INCLUDE_DIRS
    "/opt/maia_deps_gnu/parallel-netcdf-1.14.0/include"
    "/opt/maia_deps_gnu/fftw-3.3.10/include"
    "/opt/maia_deps_gnu/hdf5-1.14.5/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
)

set(LIBRARY_DIRS
    "/opt/maia_deps_gnu/parallel-netcdf-1.14.0/lib"
    "/opt/maia_deps_gnu/fftw-3.3.10/lib"
    "/opt/maia_deps_gnu/hdf5-1.14.5/lib"
    "/usr/lib/x86_64-linux-gnu/openmpi/lib"
)

set(LIBRARY_NAMES "pnetcdf" "fftw3_mpi" "fftw3" "hdf5_hl" "hdf5" "sz" "aec" "z" "dl" "mpi_cxx" "mpi")

set(HOST_CONFIGURE "export LC_ALL=C" "export LANG=C")
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_PRESERVE_ENV "=")
EOF

    COMPILER_CONFIG="${MAIA_DIR}/auxiliary/compilers/GNU.cmake"
    if [[ ! -f "${COMPILER_CONFIG}" ]]; then
        echo "Creating GNU compiler config..."
        cat > "${COMPILER_CONFIG}" << 'EOF'
# GNU Compiler Configuration for MAIA
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran")

# Build types
set(CMAKE_BUILD_TYPE_DEBUG "debug")
set(CMAKE_BUILD_TYPE_PRODUCTION "production")
set(CMAKE_BUILD_TYPE_EXTREME "extreme")

# Default build type
set(DEFAULT_BUILD_TYPE "production")

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fPIC -std=c++20")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fPIC")

set(CMAKE_C_FLAGS_PRODUCTION "-O3 -march=native -mtune=native -fPIC")
set(CMAKE_CXX_FLAGS_PRODUCTION "-O3 -march=native -mtune=native -fPIC -std=c++20")
set(CMAKE_Fortran_FLAGS_PRODUCTION "-O3 -march=native -mtune=native -fPIC")

set(CMAKE_C_FLAGS_EXTREME "-Ofast -march=native -mtune=native -fPIC -funroll-loops")
set(CMAKE_CXX_FLAGS_EXTREME "-Ofast -march=native -mtune=native -fPIC -std=c++20 -funroll-loops")
set(CMAKE_Fortran_FLAGS_EXTREME "-Ofast -march=native -mtune=native -fPIC -funroll-loops")

# OpenMP
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_Fortran_FLAGS "-fopenmp")

# PSTL (multicore only for GNU)
set(HAVE_PSTL "YES")
set(PSTL_MULTICORE_FLAGS "-stdpar=multicore")
EOF
    fi

    CONFIGURE_CMD=(
        python3 configure.py gnu "${BUILD_TYPE}"
        --with-hdf5
        --disable-updateGitSubmodules
        --compile-commands
    )

    if [[ -n "${ENABLE_COMPONENTS}" ]]; then
        IFS=',' read -ra COMPS <<< "${ENABLE_COMPONENTS}"
        for comp in "${COMPS[@]}"; do
            CONFIGURE_CMD+=(--enable-"${comp}")
        done
    fi

    if [[ -n "${DISABLE_COMPONENTS}" ]]; then
        IFS=',' read -ra COMPS <<< "${DISABLE_COMPONENTS}"
        for comp in "${COMPS[@]}"; do
            CONFIGURE_CMD+=(--disable-"${comp}")
        done
    fi

    echo "Running configure.py for GNU..."
    "${CONFIGURE_CMD[@]}"

    echo "Building MAIA CPU (solver + tests)..."
    make -j16 -C "${BUILD_DIR}" maia test_maia

    if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
        echo "=== MAIA CPU build successful ==="
    else
        echo "ERROR: MAIA CPU build failed - binaries not found"
        exit 1
    fi

    echo "Running MAIA CPU tests..."
    (cd "${BUILD_DIR}" && ./bin/test_maia)
fi

# Solver binary alone is useless without something that can drive it -
# make sure hydrogym[maia] is importable from a dedicated venv.
bash /workspace/hydrogym/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/maia-cpu maia

# Set up compile_commands.json symlink (suffixed so it doesn't clash with the GPU build's)
if [[ -f "${BUILD_DIR}/compile_commands.json" && ! -L "${MAIA_DIR}/compile_commands.json.gnu" ]]; then
    ln -sf "${BUILD_DIR}/compile_commands.json" "${MAIA_DIR}/compile_commands.json.gnu"
    echo "Created compile_commands.json.gnu symlink"
fi

echo ""
echo "=== MAIA CPU Post-Create Complete ==="
echo "Binary: ${BUILD_DIR}/bin/maia"
echo "Build dir: ${BUILD_DIR}"
echo ""
echo "To run this binary manually, the GNU-toolchain libs must come before"
echo "HPCX's in LD_LIBRARY_PATH (otherwise it picks up an ABI-incompatible"
echo "libmpi.so and fails with 'undefined symbol: ompi_mpi_errors_throw_exceptions'):"
echo "  export LD_LIBRARY_PATH=\"/usr/lib/x86_64-linux-gnu:\${MAIA_LIB_HOME_GNU}/hdf5-1.14.5/lib:\${MAIA_LIB_HOME_GNU}/parallel-netcdf-1.14.0/lib:\${MAIA_LIB_HOME_GNU}/fftw-3.3.10/lib:\${LD_LIBRARY_PATH}\""
