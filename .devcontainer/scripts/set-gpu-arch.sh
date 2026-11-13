#!/usr/bin/env bash
# Set the GPU architecture preset (pstlPreset) across every devcontainer
# config that builds maia-gpu, instead of hand-editing each one. See
# .devcontainer/README.md for the full preset -> GPU table.
#
# Usage:
#   ./set-gpu-arch.sh              # auto-detect from nvidia-smi compute_cap
#   ./set-gpu-arch.sh ampere       # set explicitly (skips auto-detection)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVCONTAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VALID_PRESETS=(volta turing ampere ada hopper blackwell blackwell_consumer multicore HOST)

detect_preset() {
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
    if [[ -z "${cc}" ]]; then
        echo "ERROR: nvidia-smi not found, or returned no GPU - pass a preset explicitly." >&2
        echo "  ./set-gpu-arch.sh <${VALID_PRESETS[*]// /|}>" >&2
        exit 1
    fi
    # nvidia-smi's compute_cap is the raw "major.minor" pair, not the
    # marketing arch name - map the values this repo's base image (CUDA
    # 13.2 / NVHPC 26.5) actually ships presets for. 8.7 is Jetson
    # Orin's Ampere variant.
    case "${cc}" in
        7.0) echo volta ;;
        7.5) echo turing ;;
        8.0|8.6|8.7) echo ampere ;;
        8.9) echo ada ;;
        9.0) echo hopper ;;
        10.0) echo blackwell ;;
        12.0) echo blackwell_consumer ;;
        *)
            echo "ERROR: unrecognized compute capability '${cc}' - pass a preset explicitly:" >&2
            echo "  ./set-gpu-arch.sh <${VALID_PRESETS[*]// /|}>" >&2
            exit 1
            ;;
    esac
}

PRESET="${1:-}"
if [[ -z "${PRESET}" ]]; then
    PRESET="$(detect_preset)"
    echo "Detected compute capability -> pstlPreset=${PRESET}"
fi

valid=0
for p in "${VALID_PRESETS[@]}"; do
    [[ "${p}" == "${PRESET}" ]] && valid=1
done
if [[ "${valid}" -ne 1 ]]; then
    echo "ERROR: '${PRESET}' is not a valid pstlPreset. Valid values: ${VALID_PRESETS[*]}" >&2
    exit 1
fi

# The only configs build_all_containers.sh (and a normal devcontainer open)
# actually build with the maia-gpu feature - CPU-only configs have no
# pstlPreset to set, and devcontainer-template.json is a reference/copy
# source, not something built directly.
FILES=(devcontainer.json maia-gpu-test.devcontainer.json full-gpu-stack.devcontainer.json)
for f in "${FILES[@]}"; do
    path="${DEVCONTAINER_DIR}/${f}"
    if [[ ! -f "${path}" ]]; then
        echo "WARNING: ${path} not found - skipping"
        continue
    fi
    sed -i \
        -e "s/\"PSTL_PRESET\": \"[^\"]*\"/\"PSTL_PRESET\": \"${PRESET}\"/" \
        -e "s/\"pstlPreset\": \"[^\"]*\"/\"pstlPreset\": \"${PRESET}\"/" \
        "${path}"
    echo "Updated ${f} -> ${PRESET}"
done

echo ""
echo "Done. pstlPreset is now '${PRESET}' in: ${FILES[*]}"
