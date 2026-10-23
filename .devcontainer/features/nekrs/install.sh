#!/bin/bash
# NekRS Feature Install Script
#
# Runs during the image build, before /workspace is bind-mounted, so the
# NekRS source (third_party/nekrs submodule) isn't visible yet - the build
# happens in postCreateCommand.sh. This script only persists the options.

set -euo pipefail

CUDAARCH="${CUDAARCH:-80;86}"
GPUMPI="${GPUMPI:-true}"
BUILDTYPE="${BUILDTYPE:-RelWithDebInfo}"
CUDAVERSION="${CUDAVERSION:-13.2}"

echo "=== NekRS Feature Installation ==="
echo "CUDA architectures: ${CUDAARCH}  GPU-aware MPI: ${GPUMPI}  build type: ${BUILDTYPE}"

CONFIG_DIR=/opt/maia-feature-config
mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/nekrs.env" <<EOT
CUDAARCH="${CUDAARCH}"
GPUMPI=${GPUMPI}
BUILDTYPE=${BUILDTYPE}
CUDAVERSION=${CUDAVERSION}
EOT

echo "Options saved to ${CONFIG_DIR}/nekrs.env"

echo "=== NekRS feature configured (build deferred to postCreateCommand) ==="
