#!/bin/bash
# HydroGym GPU Feature Post-Create Command
# Installs HydroGym (against the live /workspace mount, which
# isn't attached yet at build time - see install.sh) and verifies the
# environment.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/hydrogym-gpu.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
HYDROGYM_EXTRAS="${HYDROGYM_EXTRAS:-maia,jax}"

echo "=== HydroGym GPU Post-Create Setup ==="

VENV_DIR="/opt/venvs/ml"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "WARNING: ML venv not found at ${VENV_DIR}"
    echo "Please install the 'python-ml' and 'hydrogym-gpu' features"
    exit 0
fi

# ensure_hydrogym.sh (used by maia-gpu/maia-cpu/nek5000) checks
# importlib.metadata rather than a bare `import hydrogym` - a bare import
# check run with cwd=/workspace (this lifecycle command's default cwd,
# which contains a directory literally named "hydrogym", the bind-mounted
# repo) false-positives via Python's implicit namespace packages, since
# `python -c` puts cwd on sys.path. That false positive previously made
# this script skip the real `pip install`, leaving hydrogym never actually
# installed here despite every check appearing to pass.
bash /workspace/.devcontainer/scripts/ensure_hydrogym.sh "${VENV_DIR}" "${HYDROGYM_EXTRAS}"

source "${VENV_DIR}/activate_hydrogym_gpu.sh"

echo "Testing HydroGym GPU environment..."
python -c "
import hydrogym, importlib.metadata
print(f'HydroGym: {importlib.metadata.version(\"hydrogym\")}')
"

python -c "
import jaxfluids
print(f'JAX-Fluids: {jaxfluids.__version__ if hasattr(jaxfluids, \"__version__\") else \"installed\"}')
"

python -c "
import jax
import torch
print(f'JAX devices: {jax.devices()}')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"

deactivate

echo ""
echo "=== HydroGym GPU Post-Create Complete ==="
echo "Activate environment with:"
echo "  source ${VENV_DIR}/activate_hydrogym_gpu.sh"
echo ""
echo "This environment provides:"
echo "  - HydroGym with MAIA + JAX backends"
echo "  - JAX-Fluids CFD solver"
echo "  - PyTorch + stable-baselines3 for RL"
echo "  - JAX with CUDA 12 (pip-managed)"