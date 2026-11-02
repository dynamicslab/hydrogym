#!/bin/bash
# Python ML Stack Feature Post-Create Command
# Verifies PyTorch/JAX/SB3 installation

set -euo pipefail

echo "=== Python ML Stack Post-Create Verification ==="

VENV_DIR="/opt/venvs/ml"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "WARNING: ML venv not found at ${VENV_DIR}"
    echo "Please install the 'python-ml' feature"
    exit 0
fi

# Test activation and imports
echo "Testing ML environment..."
source "${VENV_DIR}/activate_ml.sh"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "
import jax
print(f'JAX: {jax.__version__}')
print(f'Devices: {jax.devices()}')
print(f'Backend: {jax.default_backend()}')
"

python -c "
import stable_baselines3
print(f'stable-baselines3: {stable_baselines3.__version__}')
"

deactivate

echo ""
echo "=== Python ML Stack Post-Create Complete ==="
echo "Activate environment with:"
echo "  source ${VENV_DIR}/activate_ml.sh"
echo ""
echo "Note: This venv uses pip-managed CUDA 12.8 for PyTorch/JAX."
echo "System CUDA 13.2 (NVHPC) is separate and used by MAIA GPU."