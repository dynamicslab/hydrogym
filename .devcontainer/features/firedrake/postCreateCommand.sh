#!/bin/bash
# Firedrake + HydroGym CPU Feature Post-Create Command
# Installs HydroGym (against the live /workspace/hydrogym mount, which
# isn't attached yet at build time - see install.sh) and verifies the
# environment.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/firedrake.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
HYDROGYM_EXTRAS="${HYDROGYM_EXTRAS:-maia,firedrake,nek}"

echo "=== Firedrake + HydroGym CPU Post-Create Setup ==="

VENV_DIR="/opt/venvs/firedrake"
HYDROGYM_DIR="/workspace/hydrogym"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "WARNING: Firedrake venv not found at ${VENV_DIR}"
    echo "Please install the 'firedrake' feature (requires 'petsc' feature)"
    exit 0
fi

source "${VENV_DIR}/activate_firedrake.sh"

# Lifecycle commands can run with cwd inside /workspace/hydrogym (the repo
# root, which itself contains a `hydrogym/` package subdirectory) - `python
# -c` adds cwd to sys.path first, so `import hydrogym` can resolve to that
# raw, never-actually-pip-installed source tree and "succeed" without
# hydrogym ever really being installed (see install.sh's identical
# firedrake/pyop2 shadowing issue for the same root cause). cd away so the
# check below reflects whether it's genuinely installed.
cd /tmp

if ! python -c "import hydrogym" 2>/dev/null; then
    if [[ -d "${HYDROGYM_DIR}" ]]; then
        echo "Installing HydroGym with extras: ${HYDROGYM_EXTRAS}..."
        (cd "${HYDROGYM_DIR}" && git lfs pull 2>/dev/null || true)
        pip install -e "${HYDROGYM_DIR}[${HYDROGYM_EXTRAS}]"
    else
        echo "WARNING: ${HYDROGYM_DIR} not found - check the workspace bind mount in devcontainer.json."
    fi
fi

# Test activation and imports
echo "Testing Firedrake environment..."
# This dev build of firedrake has no __version__ attribute (same situation
# as hydrogym below), use importlib.metadata.
python -c "import firedrake, importlib.metadata; print(f'Firedrake: {importlib.metadata.version(\"firedrake\")}')"
python -c "import hydrogym, importlib.metadata; print(f'HydroGym: {importlib.metadata.version(\"hydrogym\")}')"
python -c "import petsc4py; import mpi4py; import h5py; print('Core dependencies OK')"

deactivate

echo ""
echo "=== Firedrake + HydroGym CPU Post-Create Complete ==="
echo "Activate environment with:"
echo "  source ${VENV_DIR}/activate_firedrake.sh"
echo ""
echo "Then you can run HydroGym with Firedrake backend:"
echo "  python -c \"import hydrogym; env = hydrogym.FiredrakeEnv(...)\""