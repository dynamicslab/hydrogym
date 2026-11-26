#!/bin/bash
# Build m-AIA (HIP, hipstdpar) for the ROCm image and install it as
# /opt/maia/bin/maia. Called from rocm/Dockerfile.hydrogym-rocm.
#
# Environment:
#   MAIA_SRC    m-AIA source tree (third_party/m-AIA, with include/Eigen and
#               include/doctest filled in)
#   GPU_ARCH    offload target(s), ';'-separated (gfx942)
#   MAIA_MARCH  host CPU target instead of the compiler file's -march=native: the
#               image is built on a different machine than it runs on
#               (znver4 = the MI300A's Zen 4 cores)
#   MAIA_BUILD_TYPE  production (default)
#
# hipstdpar offloads std:: algorithms on plain host allocations, which needs
# XNACK (unified memory) at run time: the installed maia is a wrapper that
# sets HSA_XNACK=1 (no-op on the APU MI300A, required on MI300X/MI250X).

set -euo pipefail

: "${MAIA_SRC:?}" "${GPU_ARCH:?}"
MAIA_MARCH="${MAIA_MARCH:-znver4}"
MAIA_BUILD_TYPE="${MAIA_BUILD_TYPE:-production}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /opt/maia/wrap /opt/maia/bin
printf '#!/bin/sh\nexport OMPI_CXX=hipcc\nexec %s/bin/mpicxx "$@"\n' "${MPI_HOME:-/opt/ompi}" > /opt/maia/wrap/mpicxx
chmod 755 /opt/maia/wrap/mpicxx

offload=""
IFS=';' read -ra archs <<< "${GPU_ARCH}"
for a in "${archs[@]}"; do offload+="\"--offload-arch=${a}\" "; done
compiler_file="${MAIA_SRC}/auxiliary/compilers/AMDGPU_AAC6.cmake"
sed -i -e "s/\"-march=native\"/\"-march=${MAIA_MARCH}\"/g" -e 's/ *"-mtune=native"//g' \
    -e "s/^\( *\)\"--offload-arch=gfx942\"/\1${offload}/" "${compiler_file}"
grep -q -- "-march=${MAIA_MARCH}" "${compiler_file}" && grep -q -- "--offload-arch=${archs[0]}" "${compiler_file}" \
    || { echo "ERROR: could not adapt ${compiler_file} (m-AIA pin changed?)"; exit 1; }

cp "${HERE}/ROCMCONTAINER.cmake" "${MAIA_SRC}/auxiliary/hosts/ROCMCONTAINER.cmake"
export MAIA_HOST_FILE="${MAIA_SRC}/auxiliary/hosts/ROCMCONTAINER.cmake"

cd "${MAIA_SRC}"
build_dir="${MAIA_SRC}/build_amdgpu_aac6_${MAIA_BUILD_TYPE}"
rm -rf "${build_dir}"
python3 configure.py amdgpu_aac6 "${MAIA_BUILD_TYPE}" --with-hdf5 --disable-openmp --disable-updateGitSubmodules
if ! make -j"$(compute_parallelism.sh)" -C "${build_dir}" maia > "${MAIA_SRC}/make.log" 2>&1; then
    grep -n -B2 -A8 -m5 -E "error:|Error [0-9]" "${MAIA_SRC}/make.log" || tail -80 "${MAIA_SRC}/make.log"
    echo "ERROR: m-AIA build failed - full log in ${MAIA_SRC}/make.log"
    exit 1
fi
install -m 755 "${build_dir}/bin/maia" /opt/maia/bin/maia.bin
printf '#!/bin/sh\nexport HSA_XNACK="${HSA_XNACK:-1}"\nexec /opt/maia/bin/maia.bin "$@"\n' > /opt/maia/bin/maia
chmod 755 /opt/maia/bin/maia
echo "=== m-AIA build successful: /opt/maia/bin/maia ==="
