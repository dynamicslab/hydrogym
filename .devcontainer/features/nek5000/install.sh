#!/bin/bash
# Nek5000 Feature Install Script
#
# NOTE: this runs during `docker build` (image build time), *before* the
# workspace bind mount is attached, so /workspace/hydrogym isn't visible
# here yet - and the actual case source now lives in-repo, under
# third_party/nek5000/ (git submodules for Nek5000 core + KTH Toolbox, plus
# the four case sources), not as something this script can fetch on its
# own. The build therefore happens in postCreateCommand.sh instead, once
# the mount is live and (assuming `git submodule update --init` has been
# run on the host) the submodule content is populated. This script just
# persists the chosen feature options for that later step.

set -euo pipefail

CASES="${CASES:-mini_channel,large_channel,naca0012_200k,small_wing}"

echo "=== Nek5000 Feature Installation ==="
echo "Cases: ${CASES}"

CONFIG_DIR=/opt/maia-feature-config
mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/nek5000.env" <<EOF
CASES=${CASES}
EOF

echo "Options saved to ${CONFIG_DIR}/nek5000.env"
echo "=== Nek5000 feature configured (build deferred to postCreateCommand) ==="
