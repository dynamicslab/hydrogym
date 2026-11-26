#!/bin/bash
# Build m-AIA for the release images.
#
# Environment:
#   MAIA_SRC         m-AIA source tree (third_party/m-AIA with include/Eigen
#                    and include/doctest filled in)
#   MAIA_COMPILER    nvhpc (GPU, stdpar) | gnu (CPU)
#   MAIA_HOST_TEMPLATE  m-AIA host file for the image (MAIA_CONTAINER_NVHPC.cmake
#                    or MAIA_CONTAINER_GNU.cmake next to this script)
#   MAIA_PSTL_PRESET nvhpc only: one of configure.py's --enable-pstl presets
#                    (turing, ampere, ada, hopper, blackwell, blackwell_consumer)
#   MAIA_BUILD_TYPE  production (default)
#   MAIA_JOBS        make -j (default: compute_parallelism.sh)
#   MAIA_NVHPC_TP    nvhpc only: CPU target (-tp=...) instead of -tp=native;
#                    default haswell on x86_64, neoverse-v2 (Grace) on aarch64
#   MAIA_RESUME      1: keep an existing build directory and continue its make
#                    (an interrupted build, e.g. in a time-limited batch job)
#
# Installs /opt/maia/<ccXX>/bin/maia (nvhpc: one binary per GPU architecture,
# each a single-architecture build; the maia wrapper on PATH picks one at run
# time) or /opt/maia/bin/maia (gnu).
#
# Portability: the image is built on another machine than it runs on, so the
# host-specific CPU target of the NVHPC compiler file (-tp=native) is replaced
# by haswell (AVX2) on x86_64 and by neoverse-v2 (NVIDIA Grace, GH200) on
# aarch64, and the GNU build gets -march=x86-64-v3 (AVX2): every x86_64 server
# and desktop CPU of the last ten years supports it.

set -euo pipefail

: "${MAIA_SRC:?}" "${MAIA_COMPILER:?}" "${MAIA_HOST_TEMPLATE:?}"
MAIA_BUILD_TYPE="${MAIA_BUILD_TYPE:-production}"

cd "${MAIA_SRC}"
host_file="${MAIA_SRC}/auxiliary/hosts/MAIACONTAINER.cmake"
cp "${MAIA_HOST_TEMPLATE}" "${host_file}"
export MAIA_HOST_FILE="${host_file}"

configure_args=(--with-hdf5 --disable-updateGitSubmodules)
case "${MAIA_COMPILER}" in
    nvhpc)
        if [[ -z "${MAIA_PSTL_PRESET:-}" ]]; then
            echo "No MAIA_PSTL_PRESET: nothing to build."
            mkdir -p /opt/maia
            exit 0
        fi
        tp="${MAIA_NVHPC_TP:-}"
        [[ -n "${tp}" ]] || case "$(uname -m)" in aarch64) tp=neoverse-v2 ;; *) tp=haswell ;; esac
        # (NVHPC.cmake also sets -tp=haswell for some host names, MAIACONTAINER
        # among them)
        sed -i -E "s/\"-tp=(native|haswell)\"/\"-tp=${tp}\"/g" auxiliary/compilers/NVHPC.cmake
        grep -q "\"-tp=${tp}\"" auxiliary/compilers/NVHPC.cmake \
            || { echo "ERROR: could not set -tp=${tp} in NVHPC.cmake (m-AIA pin changed?)"; exit 1; }
        case "${MAIA_PSTL_PRESET}" in
            turing) cc=cc75 ;; ampere) cc=cc80 ;; ada) cc=cc89 ;; hopper) cc=cc90 ;;
            blackwell) cc=cc100 ;; blackwell_consumer) cc=cc120 ;;
            *) echo "ERROR: unsupported MAIA_PSTL_PRESET '${MAIA_PSTL_PRESET}'"; exit 1 ;;
        esac
        prefix="/opt/maia/${cc}"
        configure_args+=(--enable-pstl "${MAIA_PSTL_PRESET}")
        ;;
    gnu)
        sed -i 's/^set(_cxx_flags_production "-O3"/set(_cxx_flags_production "-O3" "-march=x86-64-v3" "-mtune=generic"/' \
            auxiliary/compilers/GNU.cmake
        grep -q '"-march=x86-64-v3"' auxiliary/compilers/GNU.cmake \
            || { echo "ERROR: could not set -march=x86-64-v3 in GNU.cmake (m-AIA pin changed?)"; exit 1; }
        # Some of the pinned sources compare floats exactly on purpose
        # (lbsolverdxqy.cpp); GCC's production flags make that an error.
        sed -i 's/^\( *\)"-Wno-error=stack-protector"/\1"-Wno-error=stack-protector"\n\1"-Wno-error=float-equal"/' \
            auxiliary/compilers/GNU.cmake
        grep -q '"-Wno-error=float-equal"' auxiliary/compilers/GNU.cmake \
            || { echo "ERROR: could not add -Wno-error=float-equal to GNU.cmake (m-AIA pin changed?)"; exit 1; }
        prefix=/opt/maia
        ;;
    *)
        echo "ERROR: MAIA_COMPILER must be nvhpc or gnu, got '${MAIA_COMPILER}'"; exit 1 ;;
esac

build_dir="${MAIA_SRC}/build_${MAIA_COMPILER}_${MAIA_BUILD_TYPE}"
if [[ "${MAIA_RESUME:-0}" == "1" && -f "${build_dir}/Makefile" ]]; then
    echo "MAIA_RESUME=1: continuing the build in ${build_dir}"
else
    rm -rf "${build_dir}"
    python3 configure.py "${MAIA_COMPILER}" "${MAIA_BUILD_TYPE}" "${configure_args[@]}"
fi
# NVHPC (CUDA 12 device link): every object's device code is registered
# under a name made of its source file and a short random hash; two objects
# drawing the same hash fail the device link ("redefinition of
# '__cudaRegisterLinkedBinary_..."), at random. Then those objects are
# compiled again (new hashes) and the link repeated.
jobs="${MAIA_JOBS:-$(compute_parallelism.sh 2>/dev/null || nproc)}"
for attempt in 1 2 3 4; do
    if make -j"${jobs}" -C "${build_dir}" maia > "${MAIA_SRC}/make.log" 2>&1; then
        break
    fi
    ids="$(grep -o "redefinition of '__cudaRegisterLinkedBinary_[A-Za-z0-9_]*'" "${MAIA_SRC}/make.log" \
        | sed "s/.*__cudaRegisterLinkedBinary\(_[A-Za-z0-9_]*\)'/\1/" | sort -u || true)"
    if [[ -z "${ids}" || ${attempt} -eq 4 ]]; then
        grep -n -B2 -A8 -m5 -E "error:|Error [0-9]" "${MAIA_SRC}/make.log" || tail -80 "${MAIA_SRC}/make.log"
        echo "ERROR: m-AIA build failed"
        exit 1
    fi
    for id in ${ids}; do
        objs="$(grep -l -a -r --include='*.o' -- "${id}" "${build_dir}/src" || true)"
        echo "device-code name clash ${id} in: ${objs//$'\n'/ } - compiling them again"
        [[ -n "${objs}" ]] && rm -f ${objs}
    done
done
mkdir -p "${prefix}/bin"
install -m 755 "${build_dir}/bin/maia" "${prefix}/bin/maia"
echo "=== m-AIA build successful: ${prefix}/bin/maia ==="
