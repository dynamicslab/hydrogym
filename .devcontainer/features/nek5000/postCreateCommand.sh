#!/bin/bash
# Nek5000 Feature Post-Create Command
#
# Builds each requested case's nek5000 binary from the in-repo
# third_party/nek5000/ tree (build_nek5000.sh, shared with the ROCm image)
# with the apt/GNU OpenMPI toolchain, then a hydrogym[nek] venv at
# /opt/venvs/nek5000.
#
# Runs after the container is created, once the host bind mount (see
# devcontainer.json "workspaceMount") has attached /workspace.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/nek5000.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
CASES="${CASES:-mini_channel,large_channel,naca0012_200k,small_wing}"

echo "=== Nek5000 Post-Create Setup ==="

NEK_ROOT="/workspace/third_party/nek5000"
CASES_DIR="${NEK_ROOT}/cases"

NEK_ROOT="${NEK_ROOT}" CASES="${CASES}" MPI_FC=/usr/bin/mpifort.openmpi MPI_CC=/usr/bin/mpicc.openmpi MPI_BIN=/usr/bin \
    bash /workspace/.devcontainer/features/nek5000/build_nek5000.sh

# Solver binaries alone are useless without something that can drive them -
# make sure hydrogym[nek] is importable from a dedicated venv.
bash /workspace/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/nek5000 nek

echo ""
echo "=== Nek5000 Post-Create Complete ==="
echo "Case binaries: ${CASES_DIR}/<case>/nek5000"
echo "Add a case's directory to PATH, or pass --nek-path explicitly, to use it from hydrogym.nek."
