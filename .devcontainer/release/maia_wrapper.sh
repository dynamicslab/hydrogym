#!/bin/bash
# maia: start the m-AIA build of this image that matches the GPU.
#
# The image holds one single-architecture m-AIA build per GPU architecture,
# /opt/maia/<ccXY>/bin/maia. This wrapper picks the one for the visible GPU
# (CUDA driver API, so CUDA_VISIBLE_DEVICES applies; nvidia-smi as fallback):
# the exact compute capability, else the highest build of the same major
# version below it (e.g. cc80 for cc86/cc87; device code of a cc8.0 build runs
# on every 8.x GPU).
#
#   MAIA_GPU_CC=cc80 maia ...   use that build, without detection
#   maia --list-builds          show the builds of this image
set -euo pipefail

BUILDS=/opt/maia
available() { for d in "${BUILDS}"/cc*/bin/maia; do [[ -x "$d" ]] && basename "$(dirname "$(dirname "$d")")"; done | sort -V || true; }

if [[ "${1:-}" == "--list-builds" ]]; then
    available
    exit 0
fi

cc="${MAIA_GPU_CC:-}"
if [[ -z "${cc}" ]]; then
    status=0
    cc="$("${BUILDS}/libexec/gpu_cc" 2>/dev/null)" || status=$?
    if [[ ${status} -eq 2 ]]; then
        echo "maia: the visible GPUs have different compute capabilities (${cc}); restrict CUDA_VISIBLE_DEVICES or set MAIA_GPU_CC." >&2
        exit 1
    elif [[ ${status} -ne 0 || -z "${cc}" ]]; then
        cc=""
        # nvidia-smi only if the driver library itself could not be loaded
        # (it ignores CUDA_VISIBLE_DEVICES)
        if [[ ${status} -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
            cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d ' .')" || cap=""
            [[ -n "${cap}" ]] && cc="cc${cap}"
        fi
    fi
fi
if [[ -z "${cc}" ]]; then
    echo "maia: no NVIDIA GPU visible. Start the container with GPU access (docker run --gpus all, apptainer --nv)," >&2
    echo "      or set MAIA_GPU_CC to one of: $(available | tr '\n' ' ')" >&2
    exit 1
fi
cc="cc${cc#cc}"

bin=""
if [[ -x "${BUILDS}/${cc}/bin/maia" ]]; then
    bin="${BUILDS}/${cc}/bin/maia"
else
    num="${cc#cc}" major="${cc#cc}"; major="${major%?}"
    for b in $(available | sort -rV); do
        bnum="${b#cc}" bmajor="${b#cc}"; bmajor="${bmajor%?}"
        if [[ "${bmajor}" == "${major}" && "${bnum}" -le "${num}" ]]; then
            bin="${BUILDS}/${b}/bin/maia"
            break
        fi
    done
fi
if [[ -z "${bin}" ]]; then
    echo "maia: this image has no m-AIA build for a ${cc} GPU (builds: $(available | tr '\n' ' '))." >&2
    echo "      MAIA_GPU_CC selects one of them by hand." >&2
    exit 1
fi
exec "${bin}" "$@"
