#!/bin/bash
# JAX-Fluids Standalone Feature Post-Create Command
# Verifies JAX-Fluids installation

set -euo pipefail

echo "=== JAX-Fluids Post-Create Verification ==="

VENV_DIR="/opt/venvs/ml"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "WARNING: ML venv not found at ${VENV_DIR}"
    echo "Please install the 'python-ml' and 'jax-fluids' features"
    exit 0
fi

# Test activation and import
echo "Testing JAX-Fluids environment..."
source "${VENV_DIR}/activate_jaxfluids.sh"

python -c "
import jaxfluids
print(f'JAX-Fluids: {jaxfluids.__version__ if hasattr(jaxfluids, \"__version__\") else \"installed\"}')
"

deactivate

echo ""
echo "=== JAX-Fluids Post-Create Complete ==="
echo "Activate environment with:"
echo "  source ${VENV_DIR}/activate_jaxfluids.sh"
echo ""
echo "This environment provides:"
echo "  - JAX-Fluids CFD solver"
echo "  - JAX with CUDA 12 (pip-managed)"
echo "  - PyTorch + stable-baselines3 available"