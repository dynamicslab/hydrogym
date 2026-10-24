#!/usr/bin/env bash
# Build the self-contained AMD/ROCm HydroGym + NekRS image
# (.devcontainer/rocm/Dockerfile.nekrs-rocm) from a committed revision.
#
# The build context is staged from `git archive` (REF, default HEAD) plus the
# third_party/nekrs submodule at the commit REF pins, so uncommitted work in
# the checkout never ends up in the image. Added on top: site CA certificates
# from .devcontainer/base/extra-ca-certs/*.crt (git-ignored) and the NekRS
# channel case files (examples/nekrs/channel/tcf.re2, rs6tcf0.f00004, fetched
# by examples/nekrs/fetch_case.py), so the image needs no network at run time.
#
# Usage:
#   .devcontainer/scripts/build_rocm_image.sh
#   GPU_ARCH="gfx90a;gfx942" TAG=hydrogym-nekrs-rocm:test .devcontainer/scripts/build_rocm_image.sh
#   DOCKER=docker REF=kl/nekrs .devcontainer/scripts/build_rocm_image.sh
# Then for an offline cluster:
#   podman save -o hydrogym-nekrs-rocm.tar <TAG>   ->  podman load -i hydrogym-nekrs-rocm.tar
#
# Environment: DOCKER (podman), TAG, REF, GPU_ARCH (gfx942), GPU_MPI (true),
# ROCM_IMAGE (see the Dockerfile), STAGE_DIR (temporary build context,
# default under /tmp - needs ~1 GB).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER="${DOCKER:-podman}"
REF="${REF:-HEAD}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
GPU_MPI="${GPU_MPI:-true}"
# ROCm version for the default tag: from ROCM_IMAGE (e.g. ...:7.2.4-complete)
# or the Dockerfile's default.
rocm_image="${ROCM_IMAGE:-$(sed -n 's/^ARG ROCM_IMAGE=//p' "${REPO_ROOT}/.devcontainer/rocm/Dockerfile.nekrs-rocm")}"
rocm_version="${rocm_image##*:}"; rocm_version="${rocm_version%%-*}"
TAG="${TAG:-hydrogym-nekrs-rocm:26.0-rocm${rocm_version}-${GPU_ARCH//;/_}}"
STAGE_DIR="${STAGE_DIR:-$(mktemp -d /tmp/hydrogym-rocm-ctx.XXXXXX)}"
LOG_DIR="${REPO_ROOT}/.devcontainer/build-logs"
mkdir -p "${LOG_DIR}" "${STAGE_DIR}/hydrogym" "${STAGE_DIR}/nekrs"
trap 'rm -rf "${STAGE_DIR}"' EXIT

cd "${REPO_ROOT}"
rev="$(git rev-parse --short "${REF}")"
echo "=== build_rocm_image.sh: ${TAG} from ${REF} (${rev}), GPU_ARCH=${GPU_ARCH} ==="
git archive "${REF}" | tar -x -C "${STAGE_DIR}/hydrogym"
nekrs_commit="$(git rev-parse "${REF}:third_party/nekrs")"
if ! git -C third_party/nekrs cat-file -e "${nekrs_commit}^{commit}" 2>/dev/null; then
    echo "ERROR: third_party/nekrs commit ${nekrs_commit} not available - run 'git submodule update --init third_party/nekrs'."
    exit 1
fi
git -C third_party/nekrs archive "${nekrs_commit}" | tar -x -C "${STAGE_DIR}/nekrs"

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

"${DOCKER}" build -f "${STAGE_DIR}/hydrogym/.devcontainer/rocm/Dockerfile.nekrs-rocm" \
    "${build_args[@]}" --label "org.opencontainers.image.revision=${rev}" \
    -t "${TAG}" "${STAGE_DIR}" 2>&1 | tee "${LOG_DIR}/nekrs-rocm.log"
exit "${PIPESTATUS[0]}"
