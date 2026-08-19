#!/bin/bash
# MAIA CPU Feature Install Script
#
# NOTE: this runs during `docker build` (image build time), *before* the
# workspace bind mount is attached - so /workspace/third_party/m-AIA does not
# exist yet here, even though it's a real submodule of this repo (it arrives
# later via the host bind mount configured in devcontainer.json, same as the
# rest of the workspace). The actual configure/build therefore happens in
# postCreateCommand.sh instead, once the mount is live. All this script does
# is persist the chosen feature options for that later step.
#
# CAVEAT: third_party/m-AIA points at the private wipmaiaml dev tree
# (git.rwth-aachen.de/aia/MAIA/Solver.git, branch wipmaiaml) for full
# RL-feature support (LB jet-actuation BCs 2007/2008, the MPMD flow-control
# channel) - see .devcontainer/README.md. This repo requires RWTH GitLab
# access; without it, `git submodule update --init` on third_party/m-AIA
# fails and maia-gpu/maia-cpu can't be built.

set -euo pipefail

# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. option "buildType" -> env var BUILDTYPE, not BUILD_TYPE) - read
# from those, not the underscored names, or a configured option silently
# falls back to the hardcoded default below.
BUILD_TYPE="${BUILDTYPE:-production}"
ENABLE_COMPONENTS="${ENABLECOMPONENTS:-}"
DISABLE_COMPONENTS="${DISABLECOMPONENTS:-}"

echo "=== MAIA CPU Feature Installation ==="
echo "Build Type: ${BUILD_TYPE}"
echo "Enable Components: ${ENABLE_COMPONENTS}"
echo "Disable Components: ${DISABLE_COMPONENTS}"

CONFIG_DIR=/opt/maia-feature-config
mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/maia-cpu.env" <<EOF
BUILD_TYPE=${BUILD_TYPE}
ENABLE_COMPONENTS=${ENABLE_COMPONENTS}
DISABLE_COMPONENTS=${DISABLE_COMPONENTS}
EOF

echo "Options saved to ${CONFIG_DIR}/maia-cpu.env"
echo "=== MAIA CPU feature configured (build deferred to postCreateCommand) ==="
