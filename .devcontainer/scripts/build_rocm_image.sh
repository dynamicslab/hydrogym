#!/usr/bin/env bash
# Build the self-contained AMD/ROCm HydroGym image (NekRS, m-AIA, Nek5000,
# Firedrake, JAX-Fluids; .devcontainer/rocm/Dockerfile.hydrogym-rocm) from a
# committed revision.
#
# The build context is staged from `git archive` (REF, default HEAD) plus the
# solver submodules at the commits REF pins (third_party/nekrs,
# third_party/m-AIA with its include/Eigen and include/doctest, and the KTH
# Nek5000 and Toolbox below third_party/nek5000), so uncommitted work in the
# checkout never ends up in the image. Added on top: site CA certificates
# from .devcontainer/base/extra-ca-certs/*.crt (git-ignored) and the NekRS
# channel case files (examples/nekrs/channel/tcf.re2, rs6tcf0.f00004, fetched
# by examples/nekrs/fetch_case.py), so the image needs no network at run time.
#
# Usage:
#   .devcontainer/scripts/build_rocm_image.sh
#   GPU_ARCH="gfx90a;gfx942" TAG=hydrogym-rocm:test .devcontainer/scripts/build_rocm_image.sh
#   DOCKER=docker REF=kl/nekrs .devcontainer/scripts/build_rocm_image.sh
# Then for an offline cluster:
#   podman save -o hydrogym-rocm.tar <TAG>   ->  podman load -i hydrogym-rocm.tar
#
# Environment: DOCKER (podman), TAG, REF, GPU_ARCH (gfx942), GPU_MPI (true),
# ROCM_IMAGE (see the Dockerfile), STAGE_DIR (temporary build context,
# default under /tmp - needs ~2 GB).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER="${DOCKER:-podman}"
REF="${REF:-HEAD}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
GPU_MPI="${GPU_MPI:-true}"
# ROCm version for the default tag: from ROCM_IMAGE (e.g. ...:7.2.4-complete)
# or the Dockerfile's default.
rocm_image="${ROCM_IMAGE:-$(sed -n 's/^ARG ROCM_IMAGE=//p' "${REPO_ROOT}/.devcontainer/rocm/Dockerfile.hydrogym-rocm")}"
rocm_version="${rocm_image##*:}"; rocm_version="${rocm_version%%-*}"
TAG="${TAG:-hydrogym-rocm:rocm${rocm_version}-${GPU_ARCH//;/_}}"
STAGE_DIR="${STAGE_DIR:-$(mktemp -d /tmp/hydrogym-rocm-ctx.XXXXXX)}"
LOG_DIR="${REPO_ROOT}/.devcontainer/build-logs"
mkdir -p "${LOG_DIR}" "${STAGE_DIR}/hydrogym"
trap 'rm -rf "${STAGE_DIR}"' EXIT

cd "${REPO_ROOT}"
rev="$(git rev-parse --short "${REF}")"
echo "=== build_rocm_image.sh: ${TAG} from ${REF} (${rev}), GPU_ARCH=${GPU_ARCH} ==="
git archive "${REF}" | tar -x -C "${STAGE_DIR}/hydrogym"
# Submodule SUB (a path below the repo, possibly nested: PARENT/SUB with
# PARENT itself a submodule) at the commit its parent pins -> DEST.
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
stage_submodule . "${REF}" third_party/nekrs "${STAGE_DIR}/nekrs"
stage_submodule . "${REF}" third_party/m-AIA "${STAGE_DIR}/maia"
maia_commit="$(git rev-parse "${REF}:third_party/m-AIA")"
for sub in include/Eigen include/doctest; do
    stage_submodule third_party/m-AIA "${maia_commit}" "${sub}" "${STAGE_DIR}/maia/${sub}"
done
cp -a "${STAGE_DIR}/hydrogym/third_party/nek5000" "${STAGE_DIR}/nek5000"
for sub in Nek5000 Toolbox; do
    stage_submodule . "${REF}" "third_party/nek5000/solver/KTH_DRL_Framework/${sub}" \
        "${STAGE_DIR}/nek5000/solver/KTH_DRL_Framework/${sub}"
done

cp .devcontainer/base/extra-ca-certs/*.crt "${STAGE_DIR}/hydrogym/.devcontainer/base/extra-ca-certs/" 2>/dev/null || true
for f in tcf.re2 rs6tcf0.f00004; do
    if [[ -f "examples/nekrs/channel/${f}" ]]; then
        cp "examples/nekrs/channel/${f}" "${STAGE_DIR}/hydrogym/examples/nekrs/channel/"
    else
        echo "WARNING: examples/nekrs/channel/${f} missing (run examples/nekrs/fetch_case.py) - the image will need network to fetch it."
    fi
done

build_args=(--build-arg GPU_ARCH="${GPU_ARCH}" --build-arg GPU_MPI="${GPU_MPI}")
[[ -n "${ROCM_IMAGE:-}" ]] && build_args+=(--build-arg ROCM_IMAGE="${ROCM_IMAGE}")

"${DOCKER}" build -f "${STAGE_DIR}/hydrogym/.devcontainer/rocm/Dockerfile.hydrogym-rocm" \
    "${build_args[@]}" --label "org.opencontainers.image.revision=${rev}" \
    -t "${TAG}" "${STAGE_DIR}" 2>&1 | tee "${LOG_DIR}/hydrogym-rocm.log"
exit "${PIPESTATUS[0]}"
