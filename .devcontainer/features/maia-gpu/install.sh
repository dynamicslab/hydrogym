#!/bin/bash
# MAIA GPU Feature Install Script
#
# NOTE: this runs during `docker build` (image build time), *before* the
# workspace bind mount is attached - so /workspace/third_party/m-AIA does not
# exist yet here, even though it's a real submodule of this repo (it arrives
# later via the host bind mount configured in devcontainer.json, same as the
# rest of the workspace). The actual configure/build therefore happens in
# postCreateCommand.sh instead, once the mount is live. All this script does
# is persist the chosen feature options for that later step.
#
# CAVEAT: third_party/m-AIA points at RWTH's public m-AIA mirror
# (git.rwth-aachen.de/aia/m-AIA/m-AIA), not the private wipmaiaml dev tree -
# it does not yet have the RL-relevant features documented in
# .devcontainer/README.md (LB jet-actuation BCs 2007/2008, the MPMD
# flow-control channel). Swap in a full-featured checkout once one is
# public, or point this submodule at it.

set -euo pipefail

# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. option "pstlPreset" -> env var PSTLPRESET, not PSTL_PRESET) - read
# from those, not the underscored names, or a configured option silently
# falls back to the hardcoded default below.
PSTL_PRESET="${PSTLPRESET:-ada}"
BUILD_TYPE="${BUILDTYPE:-production}"
ENABLE_COMPONENTS="${ENABLECOMPONENTS:-}"
DISABLE_COMPONENTS="${DISABLECOMPONENTS:-}"

echo "=== MAIA GPU Feature Installation ==="
echo "PSTL Preset: ${PSTL_PRESET}"
echo "Build Type: ${BUILD_TYPE}"
echo "Enable Components: ${ENABLE_COMPONENTS}"
echo "Disable Components: ${DISABLE_COMPONENTS}"

CONFIG_DIR=/opt/maia-feature-config
mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/maia-gpu.env" <<EOF
PSTL_PRESET=${PSTL_PRESET}
BUILD_TYPE=${BUILD_TYPE}
ENABLE_COMPONENTS=${ENABLE_COMPONENTS}
DISABLE_COMPONENTS=${DISABLE_COMPONENTS}
EOF

echo "Options saved to ${CONFIG_DIR}/maia-gpu.env"
echo "=== MAIA GPU feature configured (build deferred to postCreateCommand) ==="
