#!/usr/bin/env bash
# Build the HydroGym Apptainer image for NVIDIA GH200 nodes (aarch64: Grace
# CPU + Hopper GPU, cc90) in steps, on an aarch64 machine with Apptainer
# (>= 1.3, `apptainer build --fakeroot`), e.g. the login node of a GH200
# cluster. The arm64 counterpart of build_release_images.sh +
# release/Dockerfile.hydrogym-cuda; definition files in release/apptainer/.
#
# Apptainer cannot resume a build, and the solver compiles are long, so they
# run as `apptainer exec` of the build environment into ordinary directories
# (make continues an interrupted build: RESUME=1), and only the short steps
# are `apptainer build`s:
#
#   env       BUILD_ENV: the build environment (NVIDIA HPC SDK 26.5, HPC-X,
#             m-AIA libraries), hydrogym-cuda-build.def      [internet]
#   venv      /opt/venvs/hydrogym from uv.lock: torch (CPU), JAX (CUDA 12),
#             JAX-Fluids, mpi4py built against HPC-X           [internet]
#   nek5000   Nek5000 case binaries
#   maia      m-AIA, one build per MAIA_PRESETS entry (default hopper = cc90)
#   nekrs     NekRS (CUDA, CUDA_ARCH, default 90) + nekrs-mpmd
#   pack      BUILD_ROOT/solvers.tar: the above + HPC-X and NVHPC runtime libraries
#   image     WORK/hydrogym-cuda-base.sif, hydrogym-cuda.def   [internet: apt]
#   hydrogym  WORK/hydrogym-<version>-<revision>-gh200.sif: base + HydroGym
#             (hydrogym.def; a new HydroGym revision only repeats this step)
#   all       all of these, in this order
#
# The build context (CTX) is staged on a machine with the HydroGym checkout
# (its solver submodules initialised) and copied over:
#   STAGE_DIR=<dir> STAGE_ONLY=1 HYDROGYM_REF=<ref> .devcontainer/scripts/build_release_images.sh
#   git rev-parse --short=9 <ref> > <dir>/HYDROGYM_REVISION
#
# Usage:
#   CTX=<context> WORK=<dir> .devcontainer/scripts/build_apptainer_arm64.sh all
#   RESUME=1 ... build_apptainer_arm64.sh maia nekrs      # e.g. in a batch job
#
# Environment:
#   CTX         build context (default WORK/ctx)
#   WORK        the SIF files (default: current directory)
#   BUILD_ROOT  build environment, solver build trees, installed trees,
#               solvers.tar (default WORK; a fast local disk helps a lot)
#   BUILD_ENV   build environment: a sandbox directory (default
#               BUILD_ROOT/build-env) or, if it ends in .sif, a SIF file
#   JOBS        make -j (default 16); MAIA_JOBS, NEKRS_JOBS (default JOBS)
#   MAIA_PRESETS  m-AIA --enable-pstl presets (default hopper)
#   CUDA_ARCH   NekRS/HYPRE CMAKE_CUDA_ARCHITECTURES (default 90)
#   RESUME      1: maia/nekrs continue the existing build trees
#   PACK_WITHOUT_MAIA  1: pack (and so the image) without m-AIA, e.g. an interim image
#   VERSION     image version (1.0.2); HYDROGYM_REVISION (default: CTX/HYDROGYM_REVISION)
#   APPTAINER_CACHEDIR, APPTAINER_TMPDIR  as usual (a few tens of GB in the tmpdir)

set -euo pipefail

WORK="$(realpath -m "${WORK:-${PWD}}")"
CTX="$(realpath -m "${CTX:-${WORK}/ctx}")"
BUILD_ROOT="$(realpath -m "${BUILD_ROOT:-${WORK}}")"
BUILD_ENV="${BUILD_ENV:-${BUILD_ROOT}/build-env}"
JOBS="${JOBS:-16}"
MAIA_JOBS="${MAIA_JOBS:-${JOBS}}"
NEKRS_JOBS="${NEKRS_JOBS:-${JOBS}}"
MAIA_PRESETS="${MAIA_PRESETS:-hopper}"
CUDA_ARCH="${CUDA_ARCH:-90}"
RESUME="${RESUME:-0}"
VERSION="${VERSION:-1.0.2}"
DEFS="${CTX}/build/release/apptainer"
OPT="${BUILD_ROOT}/opt"
SRC="${BUILD_ROOT}/src"
NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
HPCX_HOME="${NVHPC_ROOT}/comm_libs/12.9/hpcx/hpcx-2.50"
BASE_SIF="${WORK}/hydrogym-cuda-base.sif"

[[ "$(uname -m)" == aarch64 ]] || { echo "ERROR: build on an aarch64 machine (this one is $(uname -m))"; exit 1; }
[[ -d "${DEFS}" ]] || { echo "ERROR: no build context in ${CTX} (see the header of this script)"; exit 1; }
[[ $# -ge 1 ]] || { sed -n '2,52p' "$0"; exit 1; }
mkdir -p "${WORK}" "${OPT}" "${SRC}"

# Run a command in the build environment; extra --bind options first.
in_env() {
    local binds=()
    while [[ "${1:-}" == --bind ]]; do binds+=("$1" "$2"); shift 2; done
    apptainer exec --cleanenv --bind "${BUILD_ROOT}" --bind "${CTX}" "${binds[@]}" \
        --env JOBS="${JOBS}",XDG_CACHE_HOME="${BUILD_ROOT}/cache" "${BUILD_ENV}" bash -c "set -euo pipefail; $1"
}

step_env() {
    local out="${BUILD_ENV}" args=(--fakeroot -F --build-arg CONTEXT="${CTX}" --build-arg JOBS="${JOBS}")
    [[ "${out}" == *.sif ]] || args+=(--sandbox)
    apptainer build "${args[@]}" "${out}" "${DEFS}/hydrogym-cuda-build.def"
}

step_venv() {
    rm -rf "${OPT}/venvs" && mkdir -p "${OPT}/venvs"
    in_env --bind "${OPT}/venvs:/opt/venvs" --bind "${CTX}/hydrogym:/opt/src/hydrogym-lock" \
        "MODE=deps HYDROGYM_SRC=/opt/src/hydrogym-lock MPI_CC=/opt/mpi-gnu/bin/mpicc JAX_SPEC='jax[cuda12]' \
            UV_CACHE_DIR=${BUILD_ROOT}/uv-cache bash ${CTX}/build/release/install_hydrogym_venv.sh"
}

step_nek5000() {
    rm -rf "${OPT}/nek5000" && cp -a "${CTX}/nek5000" "${OPT}/nek5000"
    in_env --bind "${OPT}/nek5000:/opt/nek5000" \
        "NEK_ROOT=/opt/nek5000 MPI_FC=/opt/mpi-gnu/bin/mpif77 MPI_CC=/opt/mpi-gnu/bin/mpicc MPI_BIN=/opt/mpi-gnu/bin \
            bash ${CTX}/build/features/nek5000/build_nek5000.sh && \
         find /opt/nek5000/cases -name '*.o' -delete && ls /opt/nek5000/cases/*/nek5000"
}

step_maia() {
    mkdir -p "${OPT}/maia"
    local preset src
    for preset in ${MAIA_PRESETS}; do
        src="${SRC}/maia-${preset}"
        if [[ "${RESUME}" != "1" || ! -d "${src}" ]]; then
            rm -rf "${src}" && cp -a "${CTX}/maia" "${src}"
        fi
        in_env --bind "${src}:/opt/src/m-AIA" --bind "${OPT}/maia:/opt/maia" \
            "MAIA_SRC=/opt/src/m-AIA MAIA_COMPILER=nvhpc MAIA_PSTL_PRESET=${preset} MAIA_JOBS=${MAIA_JOBS} \
                MAIA_RESUME=${RESUME} MAIA_HOST_TEMPLATE=${CTX}/build/release/MAIA_CONTAINER_NVHPC.cmake \
                bash ${CTX}/build/release/build_maia_container.sh"
    done
    ls -l "${OPT}"/maia/cc*/bin/maia
}

step_nekrs() {
    local src="${SRC}/nekrs"
    if [[ "${RESUME}" != "1" || ! -d "${src}" ]]; then
        rm -rf "${src}" "${OPT}/nekrs" && cp -a "${CTX}/nekrs" "${src}"
    fi
    mkdir -p "${OPT}/nekrs" "${BUILD_ROOT}/mpmd" && cp -a "${CTX}/hydrogym/examples/nekrs/mpmd/." "${BUILD_ROOT}/mpmd/"
    # No GPU driver at build time: OCCA's tools link against libcuda, so the
    # toolkit's stub stands in for it, on LD_LIBRARY_PATH only.
    in_env --bind "${src}:/opt/src/nekrs" --bind "${OPT}/nekrs:/opt/nekrs" \
        "stub=\$(mktemp -d) && ln -s \${CUDA_HOME}/12.9/lib64/stubs/libcuda.so \${stub}/libcuda.so.1 && \
         LD_LIBRARY_PATH=\${stub}:\${LD_LIBRARY_PATH} NEKRS_RESUME=${RESUME} \
         NEKRS_SRC=/opt/src/nekrs NEKRS_HOME=/opt/nekrs MPMD_SRC=${BUILD_ROOT}/mpmd NPROC=${NEKRS_JOBS} \
         BACKEND=CUDA GPUARCH='${CUDA_ARCH}' CUDA_VERSION=12.9 GPUMPI=true BUILDTYPE=RelWithDebInfo \
         MPI_BIN=\${MPI_HOME}/bin MPI_CC=gcc MPI_CXX=g++ MPI_FC=gfortran \
            bash ${CTX}/build/features/nekrs/build_nekrs.sh && \
         rm -rf \${stub} && grep -m1 CMAKE_CUDA_ARCHITECTURES /opt/src/nekrs/build/CMakeCache.txt && \
         ! grep -r -l -e \${stub} /opt/nekrs/bin /opt/nekrs/lib"
}

step_pack() {
    local tar="${BUILD_ROOT}/solvers.tar"
    local need="maia nekrs nek5000 venvs"
    # PACK_WITHOUT_MAIA=1: an image without m-AIA
    [[ "${PACK_WITHOUT_MAIA:-0}" == "1" ]] && need="nekrs nek5000 venvs" && mkdir -p "${OPT}/maia"
    for d in ${need}; do
        [[ -n "$(ls -A "${OPT}/${d}" 2>/dev/null)" ]] || { echo "ERROR: ${OPT}/${d} is empty - run the step first"; exit 1; }
    done
    in_env --bind "${OPT}/maia:/opt/maia" --bind "${OPT}/nekrs:/opt/nekrs" --bind "${OPT}/nek5000:/opt/nek5000" \
        --bind "${OPT}/venvs:/opt/venvs" \
        "cd / && tar -cf ${tar}.part --exclude='opt/nek5000/.git*' \
            ${HPCX_HOME#/} ${NVHPC_ROOT#/}/compilers/lib opt/mpi-gnu opt/maia_deps opt/maia opt/nekrs opt/nek5000 opt/venvs \
            usr/local/bin/uv usr/local/bin/compute_parallelism.sh usr/local/share/ca-certificates/extra"
    mv "${tar}.part" "${tar}"
    ls -l "${tar}"
}

step_image() {
    apptainer build --fakeroot -F --build-arg CONTEXT="${CTX}" --build-arg SOLVERS_TAR="${BUILD_ROOT}/solvers.tar" \
        --build-arg CUDA_ARCH="${CUDA_ARCH}" "${BASE_SIF}" "${DEFS}/hydrogym-cuda.def"
}

step_hydrogym() {
    local rev="${HYDROGYM_REVISION:-$(cat "${CTX}/HYDROGYM_REVISION" 2>/dev/null || echo unknown)}"
    local out="${WORK}/hydrogym-${VERSION}-${rev}-gh200.sif"
    tar -C "${CTX}/hydrogym" -cf "${BUILD_ROOT}/hydrogym.tar" .
    apptainer build --fakeroot -F --build-arg BASE="${BASE_SIF}" --build-arg HYDROGYM_TAR="${BUILD_ROOT}/hydrogym.tar" \
        --build-arg HYDROGYM_VERSION="${VERSION}" --build-arg HYDROGYM_REVISION="${rev}" "${out}" "${DEFS}/hydrogym.def"
    rm -f "${BUILD_ROOT}/hydrogym.tar"
    ln -sfn "$(basename "${out}")" "${WORK}/hydrogym-gh200.sif"
    ls -l "${out}"
}

steps=("$@")
[[ "${steps[*]}" == all ]] && steps=(env venv nek5000 maia nekrs pack image hydrogym)
for s in "${steps[@]}"; do
    declare -F "step_${s}" > /dev/null || { echo "ERROR: unknown step '${s}'"; exit 1; }
    t0=$(date +%s)
    echo "=== ${s}: start $(date -Iseconds) ==="
    "step_${s}"
    echo "=== ${s}: done in $(( ($(date +%s) - t0) / 60 )) min $(( ($(date +%s) - t0) % 60 )) s ==="
done
