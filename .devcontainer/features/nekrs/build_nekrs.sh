#!/bin/bash
# Configure, build and install NekRS (CUDA, HIP or CPU) plus HydroGym's
# nekrs-mpmd driver. Shared by features/nekrs/postCreateCommand.sh (CUDA, at
# container start), rocm/Dockerfile.hydrogym-rocm (HIP) and the release images
# in release/ (CUDA, CPU), the last two at image build time.
#
# Environment:
#   NEKRS_SRC     NekRS source tree (the third_party/nekrs submodule)
#   NEKRS_HOME    install prefix
#   MPMD_SRC      examples/nekrs/mpmd (driver.cpp, build.sh)
#   BACKEND       CUDA | HIP | CPU  (CPU: OCCA serial backend and CPU HYPRE;
#                 run with --backend SERIAL)
#   GPUARCH       CMAKE_CUDA_ARCHITECTURES ("80;86") or CMAKE_HIP_ARCHITECTURES (gfx942);
#                 not used for CPU
#   CUDA_VERSION  CUDA only: toolkit version below NVHPC's CUDA_HOME (13.2, 12.9)
#   GPUMPI        true | false  (NEKRS_GPU_MPI)
#   BUILDTYPE     CMAKE_BUILD_TYPE
#   MPI_BIN       directory with the MPI compiler wrappers (mpicc, mpicxx, mpif77)
#   MPI_CC, MPI_CXX, MPI_FC  compilers underneath those wrappers
#   NPROC         make jobs (default: compute_parallelism.sh in the containers, else nproc)
#   NEKRS_RESUME  1: keep an existing, configured build directory and continue its
#                 make (an interrupted build, e.g. in a time-limited batch job)
#
# The MPI wrappers are wrapped once more in $NEKRS_HOME/mpi/bin with the
# underlying compilers fixed (OMPI_CC/OMPI_CXX/OMPI_FC): NekRS records its
# compilers at configure time and calls them again at run time (udf and OCCA
# JIT builds), so they have to work without those variables in the caller's
# environment - and setting them container-wide would also change what plain
# mpicc means for every other solver in the container. The MPICH names
# (MPICH_CC/MPICH_CXX/MPICH_FC) are set as well, for MPICH-based MPIs (e.g.
# Cray MPICH) on clusters - see examples/nekrs/INSTALL_HPC.md.

set -euo pipefail

: "${NEKRS_SRC:?}" "${NEKRS_HOME:?}" "${MPMD_SRC:?}" "${BACKEND:?}" "${MPI_BIN:?}"
[[ "${BACKEND}" == "CPU" ]] || : "${GPUARCH:?}"
: "${MPI_CC:?}" "${MPI_CXX:?}" "${MPI_FC:?}"
GPUMPI="${GPUMPI:-true}"
BUILDTYPE="${BUILDTYPE:-RelWithDebInfo}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"
BUILD_DIR="${NEKRS_SRC}/build"

WRAP_DIR="${NEKRS_HOME}/mpi/bin"
mkdir -p "${WRAP_DIR}"
for spec in "mpicc:CC=${MPI_CC}" "mpicxx:CXX=${MPI_CXX}" "mpif77:FC=${MPI_FC}" "mpifort:FC=${MPI_FC}"; do
    name="${spec%%:*}" var="${spec#*:}"
    extra=""
    [[ "${var}" == FC=* ]] && extra=" MPICH_F77=${MPI_FC}"   # (older MPICH mpif77)
    printf '#!/bin/sh\nexport OMPI_%s MPICH_%s%s\nexec %s/%s "$@"\n' "${var}" "${var}" "${extra}" "${MPI_BIN}" "${name}" \
        > "${WRAP_DIR}/${name}"
    chmod 755 "${WRAP_DIR}/${name}"
done
export CC="${WRAP_DIR}/mpicc" CXX="${WRAP_DIR}/mpicxx" FC="${WRAP_DIR}/mpif77"
unset CFLAGS CXXFLAGS FFLAGS LDFLAGS CPPFLAGS

# Patches to the pinned NekRS (see the header of each file), paths relative to
# the NekRS root after stripping one leading component. Idempotent: skipped
# if the reverse patch applies. git apply also works on a plain source tree
# that isn't a git checkout (the image build).
for patch in "${PATCH_DIR}"/*.patch; do
    if git -C "${NEKRS_SRC}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "$(basename "${patch}") already applied."
    elif git -C "${NEKRS_SRC}" apply "${patch}"; then
        echo "Applied $(basename "${patch}")"
    else
        echo "ERROR: could not apply ${patch} to ${NEKRS_SRC} (NekRS pin changed?)."
        exit 1
    fi
done

gpu_mpi=OFF
[[ "${GPUMPI}" == "true" && "${BACKEND}" != "CPU" ]] && gpu_mpi=ON
hypre_gpu=ON
[[ "${BACKEND}" == "CPU" ]] && hypre_gpu=OFF

backend_args=()
case "${BACKEND}" in
    CUDA)
        # find_package(CUDAToolkit) looks in CUDA_HOME, but NVHPC ships
        # cuBLAS/cuSPARSE/cuRAND/cuSOLVER (needed by GPU HYPRE) separately
        # under math_libs/.
        # NVHPC's cuda_multi layout keeps one toolkit per version below
        # CUDA_HOME (cuda/13.2, cuda/12.9); CUDA_VERSION picks one.
        # A plain CUDA toolkit (cluster module) has them in CUDA_HOME itself.
        CUDA_ROOT="${CUDA_HOME:?CUDA_HOME not set}"
        MATH_LIBS="${NVHPC_ROOT:-}/math_libs"
        if [[ -n "${CUDA_VERSION:-}" && -d "${CUDA_HOME}/${CUDA_VERSION}" ]]; then
            CUDA_ROOT="${CUDA_HOME}/${CUDA_VERSION}"
            MATH_LIBS="${MATH_LIBS}/${CUDA_VERSION}"
        fi
        [[ -n "${NVHPC_ROOT:-}" && -d "${MATH_LIBS}/include" ]] || MATH_LIBS="${CUDA_ROOT}"
        # The NVHPC image's CPATH points at the default CUDA's math_libs (and
        # NCCL/NVSHMEM, not needed here), which shadows the chosen version's
        # headers in every compile; its LD_LIBRARY_PATH likewise, so the
        # libraries are found through RPATH instead
        # (CMAKE_INSTALL_RPATH_USE_LINK_PATH below).
        export CPATH="${MATH_LIBS}/include"
        # HYPRE's configure guesses math_libs next to CUDA_HOME, i.e. the
        # default version (patches/NekRS-26.0_hypre-vendor-math-path.patch).
        export HYPRE_VENDOR_MATH_PATH="${MATH_LIBS}"
        # OCCA JIT-compiles the kernels with $OCCA_CUDA_COMPILER (default:
        # nvcc on PATH, i.e. NVHPC's default CUDA); the feature points that at
        # this wrapper so kernels and libraries come from the same toolkit
        # (not a symlink: nvcc finds its headers relative to the path it was
        # called by).
        mkdir -p "${NEKRS_HOME}/cuda/bin"
        rm -f "${NEKRS_HOME}/cuda/bin/nvcc"
        printf '#!/bin/sh\nexec %s/bin/nvcc "$@"\n' "${CUDA_ROOT}" > "${NEKRS_HOME}/cuda/bin/nvcc"
        chmod 755 "${NEKRS_HOME}/cuda/bin/nvcc"
        backend_args=(
            -DOCCA_ENABLE_CUDA=ON -DOCCA_ENABLE_HIP=OFF
            -DCMAKE_CUDA_ARCHITECTURES="${GPUARCH}"
            -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc"
            -DCUDAToolkit_ROOT="${CUDA_ROOT}"
            -DCMAKE_PREFIX_PATH="${MATH_LIBS}"
            -DCMAKE_INCLUDE_PATH="${MATH_LIBS}/include"
            -DCMAKE_LIBRARY_PATH="${MATH_LIBS}/lib64"
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
        )
        ;;
    HIP)
        backend_args=(
            -DOCCA_ENABLE_CUDA=OFF -DOCCA_ENABLE_HIP=ON
            -DCMAKE_HIP_ARCHITECTURES="${GPUARCH}"
            -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}"
        )
        ;;
    CPU)
        backend_args=(-DOCCA_ENABLE_CUDA=OFF -DOCCA_ENABLE_HIP=OFF)
        ;;
    *)
        echo "ERROR: BACKEND must be CUDA, HIP or CPU, got '${BACKEND}'"; exit 1 ;;
esac

if [[ "${NEKRS_RESUME:-0}" == "1" && -f "${BUILD_DIR}/Makefile" ]]; then
    echo "NEKRS_RESUME=1: continuing the build in ${BUILD_DIR}"
else
rm -rf "${BUILD_DIR}"
cmake -S "${NEKRS_SRC}" -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE="${BUILDTYPE}" \
    -DCMAKE_INSTALL_PREFIX="${NEKRS_HOME}" \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_Fortran_COMPILER="${FC}" \
    "${backend_args[@]}" \
    -DOCCA_ENABLE_DPCPP=OFF -DOCCA_ENABLE_OPENCL=OFF \
    -DENABLE_HYPRE_GPU="${hypre_gpu}" -DENABLE_ADIOS=OFF -DENABLE_CVODE=OFF \
    -DNEKRS_GPU_MPI="${gpu_mpi}" \
    2>&1 | tee "${NEKRS_SRC}/cmake.log"
fi
if ! make -j"${NPROC:-$(compute_parallelism.sh 2>/dev/null || nproc)}" -C "${BUILD_DIR}" install > "${NEKRS_SRC}/make.log" 2>&1; then
    grep -n -B2 -A8 -m5 -E "error:|Error [0-9]" "${NEKRS_SRC}/make.log" || tail -80 "${NEKRS_SRC}/make.log"
    echo "ERROR: NekRS build failed - full log in ${NEKRS_SRC}/make.log"
    exit 1
fi
if [[ ! -x "${NEKRS_HOME}/bin/nekrs" ]]; then
    echo "ERROR: NekRS build failed - ${NEKRS_HOME}/bin/nekrs not found"
    exit 1
fi
echo "=== NekRS build successful: ${NEKRS_HOME} ==="

# HydroGym's MPMD driver (NekRS's own driver with a communicator split so the
# Python controller can share MPI_COMM_WORLD).
(cd "${MPMD_SRC}" && PATH="${WRAP_DIR}:${PATH}" NEKRS_HOME="${NEKRS_HOME}" ./build.sh \
    && install -m 755 nekrs-mpmd "${NEKRS_HOME}/bin/" && rm nekrs-mpmd)
