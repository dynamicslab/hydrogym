#!/usr/bin/env bash
# Build the self-contained HydroGym release images (x86_64):
#
#   all_gpus            NVIDIA cc75-cc120 in one image (six m-AIA builds):
#                       <REPO>-nvhpc-26.5_cuda-12.9, the released GPU image
#   cpu                 no GPU: <REPO>-cpu, the released CPU image
#
# Smaller GPU images with the builds of one GPU group only (not released):
#   turing_ampere_ada   NVIDIA cc75, cc80, cc86, cc89 (T4, RTX 20, A100, A30,
#                       RTX 30, A6000, RTX 40, L4, L40)
#   hopper_blackwell    NVIDIA cc90, cc100, cc120 (H100, H200, B100, B200,
#                       RTX 50)
#
# from .devcontainer/release/Dockerfile.hydrogym-{cuda,cpu}. The build context
# is staged from `git archive`, so uncommitted work never ends up in an image:
#   hydrogym/  HydroGym at HYDROGYM_REF (default HEAD), plus the NekRS channel
#              case files (examples/nekrs/channel/tcf.re2, rs6tcf0.f00004; from
#              the checkout, else downloaded from the Hugging Face Hub)
#   build/     .devcontainer of BUILD_REF (default HEAD): Dockerfiles and build
#              scripts
#   nekrs/, maia/ (with include/Eigen, include/doctest), nek5000/ (with the KTH
#              Nek5000 and Toolbox): the solver submodules at the commits
#              HYDROGYM_REF pins - they must be initialised in this checkout.
# HydroGym is the last layer of every image: a new HYDROGYM_REF with the same
# solver pins and lock file only rebuilds that layer.
#
# Usage:
#   .devcontainer/scripts/build_release_images.sh                       # GPU and CPU image
#   VARIANTS=cpu .devcontainer/scripts/build_release_images.sh
#   HYDROGYM_REF=release/1.0.2 VERSION=1.0.2 .devcontainer/scripts/build_release_images.sh
#
# Environment:
#   VARIANTS      all_gpus cpu (default, in this order); also
#                 turing_ampere_ada, hopper_blackwell
#   HYDROGYM_REF  HydroGym revision to package (HEAD)
#   BUILD_REF     revision of the Dockerfiles/build scripts (HEAD)
#   VERSION       image tag (1.0.2); every image is also tagged :latest
#                 unless LATEST=0
#   REPO          image name prefix (clagemann/hydrogym): all_gpus is
#                 <REPO>-nvhpc-26.5_cuda-12.9, cpu <REPO>-cpu
#   DOCKER        docker (or podman)
#   UBUNTU_MIRROR an Ubuntu mirror for the apt installs (if archive.ubuntu.com
#                 is unreachable), e.g. http://<country>.archive.ubuntu.com/ubuntu/
#   STAGE_DIR     build context (default: a temporary directory, ~1 GB)
#   STAGE_ONLY    1: only stage the build context in STAGE_DIR (e.g. to build
#                 on another machine: docker build -f STAGE_DIR/<dockerfile> ...)
#   BUILD_ARGS    extra docker build options, e.g. the make -j of the
#                 builds: "--build-arg JOBS=16 --build-arg MAIA_JOBS=4
#                 --build-arg NEKRS_JOBS=6" (the defaults; with the legacy
#                 docker builder, which runs the stages one after the other,
#                 MAIA_JOBS and NEKRS_JOBS can be as high as JOBS)
# Logs: .devcontainer/build-logs/release-<variant>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER="${DOCKER:-docker}"
VARIANTS="${VARIANTS:-all_gpus cpu}"
HYDROGYM_REF="${HYDROGYM_REF:-HEAD}"
BUILD_REF="${BUILD_REF:-HEAD}"
VERSION="${VERSION:-1.0.2}"
REPO="${REPO:-clagemann/hydrogym}"
LOG_DIR="${REPO_ROOT}/.devcontainer/build-logs"
mkdir -p "${LOG_DIR}"
if [[ -z "${STAGE_DIR:-}" ]]; then
    STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hydrogym-release-ctx.XXXXXX")"
    trap 'rm -rf "${STAGE_DIR}"' EXIT
fi

cd "${REPO_ROOT}"
rev="$(git rev-parse --short=9 "${HYDROGYM_REF}")"
build_rev="$(git rev-parse --short=9 "${BUILD_REF}")"
echo "=== build_release_images.sh: HydroGym ${HYDROGYM_REF} (${rev}), build files ${BUILD_REF} (${build_rev}), variants: ${VARIANTS} ==="

# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------
rm -rf "${STAGE_DIR:?}"/{hydrogym,build,nekrs,maia,nek5000}
mkdir -p "${STAGE_DIR}/hydrogym" "${STAGE_DIR}/build"
git archive "${HYDROGYM_REF}" | tar -x -C "${STAGE_DIR}/hydrogym"
git archive "${BUILD_REF}:.devcontainer" | tar -x -C "${STAGE_DIR}/build"

# Submodule SUB of PARENT at the commit PARENT_REV pins -> DEST.
stage_submodule() {
    local parent="$1" parent_rev="$2" sub="$3" dest="$4" commit
    commit="$(git -C "${parent}" rev-parse "${parent_rev}:${sub}")"
    if ! git -C "${parent}/${sub}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        echo "ERROR: ${parent}/${sub} commit ${commit} not available - run 'git submodule update --init' for it."
        exit 1
    fi
    mkdir -p "${dest}"
    git -C "${parent}/${sub}" archive "${commit}" | tar -x -C "${dest}"
}
stage_submodule . "${HYDROGYM_REF}" third_party/nekrs "${STAGE_DIR}/nekrs"
stage_submodule . "${HYDROGYM_REF}" third_party/m-AIA "${STAGE_DIR}/maia"
maia_commit="$(git rev-parse "${HYDROGYM_REF}:third_party/m-AIA")"
for sub in include/Eigen include/doctest; do
    stage_submodule third_party/m-AIA "${maia_commit}" "${sub}" "${STAGE_DIR}/maia/${sub}"
done
cp -a "${STAGE_DIR}/hydrogym/third_party/nek5000" "${STAGE_DIR}/nek5000"
for sub in Nek5000 Toolbox; do
    stage_submodule . "${HYDROGYM_REF}" "third_party/nek5000/solver/KTH_DRL_Framework/${sub}" \
        "${STAGE_DIR}/nek5000/solver/KTH_DRL_Framework/${sub}"
done
# The solver sources are separate context directories, not again in hydrogym/
# (the images link /opt/hydrogym/third_party/nek5000 to /opt/nek5000).
rm -rf "${STAGE_DIR}/hydrogym/third_party/nekrs" "${STAGE_DIR}/hydrogym/third_party/m-AIA" \
    "${STAGE_DIR}/hydrogym/third_party/nek5000"

cp .devcontainer/base/extra-ca-certs/*.crt "${STAGE_DIR}/build/base/extra-ca-certs/" 2>/dev/null || true
channel="${STAGE_DIR}/hydrogym/examples/nekrs/channel"
for f in tcf.re2 rs6tcf0.f00004; do
    [[ -f "examples/nekrs/channel/${f}" ]] && cp "examples/nekrs/channel/${f}" "${channel}/"
done
if [[ ! -f "${channel}/tcf.re2" || ! -f "${channel}/rs6tcf0.f00004" ]]; then
    echo "Fetching the NekRS channel case files from the Hugging Face Hub..."
    hf=https://huggingface.co/datasets/dynamicslab/HydroGym-environments/resolve/main/TCFlarge_3D_Re207
    curl -fsSL -o "${channel}/tcf.re2" "${hf}/tcf.re2" && \
        curl -fsSL -o "${channel}/rs6tcf0.f00004" "${hf}/restart_files/init_1/rs6tcf0.f00004" \
        || echo "WARNING: could not fetch them - the images will download them at run time (fetch_case.py)."
fi

# Git LFS files (Firedrake meshes, Nek5000 case files): `git archive` writes
# the data only where git-lfs is installed and the objects are local (git lfs
# pull); otherwise the context holds the LFS pointer files.
lfs_pointers="$(grep -rlI '^version https://git-lfs' "${STAGE_DIR}" 2>/dev/null || true)"
if [[ -n "${lfs_pointers}" ]]; then
    echo "ERROR: Git LFS pointer files in the build context (install git-lfs, run 'git lfs pull'):"
    echo "${lfs_pointers}" | head -20
    exit 1
fi

if [[ "${STAGE_ONLY:-0}" == "1" ]]; then
    echo "Build context staged in ${STAGE_DIR} (STAGE_ONLY=1, no build)."
    trap - EXIT
    exit 0
fi

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# BuildKit (docker buildx, podman) runs the m-AIA stages and the NekRS
# stage at the same time; the legacy docker builder one after the other.
progress=""
if [[ "${DOCKER}" == docker ]] && "${DOCKER}" buildx version >/dev/null 2>&1; then
    progress="--progress=plain"
fi
rc=0
summary=()
for variant in ${VARIANTS}; do
    args=(--build-arg HYDROGYM_VERSION="${VERSION}" --build-arg HYDROGYM_REVISION="${rev}")
    [[ -n "${UBUNTU_MIRROR:-}" ]] && args+=(--build-arg UBUNTU_MIRROR="${UBUNTU_MIRROR}")
    case "${variant}" in
        turing_ampere_ada)
            dockerfile=build/release/Dockerfile.hydrogym-cuda
            name="${REPO}-nvhpc-26.5_cuda-12.9_turing_ampere_ada"
            args+=(--build-arg CUDA_ARCH="75;80;86;89"
                   --build-arg MAIA_PRESET1=turing --build-arg MAIA_PRESET2=ampere --build-arg MAIA_PRESET3=ada
                   --build-arg IMAGE_TITLE="HydroGym (NVIDIA Turing, Ampere, Ada)")
            ;;
        hopper_blackwell)
            dockerfile=build/release/Dockerfile.hydrogym-cuda
            name="${REPO}-nvhpc-26.5_cuda-12.9_hopper_blackwell"
            args+=(--build-arg CUDA_ARCH="90;100;120"
                   --build-arg MAIA_PRESET1=hopper --build-arg MAIA_PRESET2=blackwell
                   --build-arg MAIA_PRESET3=blackwell_consumer
                   --build-arg IMAGE_TITLE="HydroGym (NVIDIA Hopper, Blackwell)")
            ;;
        all_gpus)
            dockerfile=build/release/Dockerfile.hydrogym-cuda
            name="${REPO}-nvhpc-26.5_cuda-12.9"
            args+=(--build-arg CUDA_ARCH="75;80;86;89;90;100;120"
                   --build-arg MAIA_PRESET1=turing --build-arg MAIA_PRESET2=ampere --build-arg MAIA_PRESET3=ada
                   --build-arg MAIA_PRESET4=hopper --build-arg MAIA_PRESET5=blackwell
                   --build-arg MAIA_PRESET6=blackwell_consumer
                   --build-arg IMAGE_TITLE="HydroGym (NVIDIA GPUs)")
            ;;
        cpu)
            dockerfile=build/release/Dockerfile.hydrogym-cpu
            name="${REPO}-cpu"
            ;;
        *)
            echo "ERROR: unknown variant '${variant}' (all_gpus, cpu, turing_ampere_ada, hopper_blackwell)"; exit 1 ;;
    esac
    log="${LOG_DIR}/release-${variant}.log"
    echo ""
    echo "=== ${variant}: ${name}:${VERSION} (log: ${log}) ==="
    start=$(date +%s)
    # shellcheck disable=SC2086
    if "${DOCKER}" build ${progress} -f "${STAGE_DIR}/${dockerfile}" "${args[@]}" ${BUILD_ARGS:-} \
        -t "${name}:${VERSION}" $([[ "${LATEST:-1}" == "1" ]] && echo "-t ${name}:latest") "${STAGE_DIR}" > "${log}" 2>&1; then
        status=OK
    else
        status="FAILED (see ${log})"; rc=1
        tail -40 "${log}"
    fi
    elapsed=$(( $(date +%s) - start ))
    summary+=("${variant}: ${status}  ${name}:${VERSION}  [$(( elapsed / 60 )) min]")
    echo "${summary[-1]}"
done

echo ""
echo "=== Summary ==="
printf '  %s\n' "${summary[@]}"
"${DOCKER}" images --filter "reference=${REPO}-*:${VERSION}" || true
exit "${rc}"
