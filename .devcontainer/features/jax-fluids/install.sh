#!/bin/bash
# JAX-Fluids Standalone Feature Install Script
# Installs JAX-Fluids in the python-ml venv
# Requires python-ml feature to be installed first

set -euo pipefail

# Feature options
# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. "jaxfluidsRepo" -> JAXFLUIDSREPO, not JAXFLUIDS_REPO) - read from
# those, or a configured option silently falls back to the default.
JAXFLUIDS_REPO="${JAXFLUIDSREPO:-https://github.com/tumaer/JAXFLUIDS.git}"
JAXFLUIDS_BRANCH="${JAXFLUIDSBRANCH:-main}"

echo "=== JAX-Fluids Standalone Feature Installation ==="
echo "Repository: ${JAXFLUIDS_REPO}"
echo "Branch: ${JAXFLUIDS_BRANCH}"

# Use the python-ml venv
VENV_DIR="/opt/venvs/ml"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Python ML venv not found at ${VENV_DIR}"
    echo "Please install the 'python-ml' feature first"
    exit 1
fi

# Activate the ML venv
source "${VENV_DIR}/bin/activate"

# Install JAX-Fluids
echo "Installing JAX-Fluids from ${JAXFLUIDS_REPO} (branch: ${JAXFLUIDS_BRANCH})..."
pip install -e "git+${JAXFLUIDS_REPO}@${JAXFLUIDS_BRANCH}#egg=jaxfluids"

# Verify installation
echo "Verifying JAX-Fluids installation..."
python -c "
import jaxfluids
print(f'JAX-Fluids: {jaxfluids.__version__ if hasattr(jaxfluids, \"__version__\") else \"installed\"}')
print(f'Location: {jaxfluids.__file__}')
"

python -c "
import jax
import torch
print(f'JAX devices: {jax.devices()}')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"

# Deactivate
deactivate

# Create activation helper script
cat > "${VENV_DIR}/activate_jaxfluids.sh" << 'EOF'
#!/bin/bash
# Source this to activate the JAX-Fluids environment
source /opt/venvs/ml/bin/activate
echo "JAX-Fluids environment activated"
EOF
chmod +x "${VENV_DIR}/activate_jaxfluids.sh"

echo "=== JAX-Fluids Standalone Feature Installation Complete ==="
echo "Venv: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/activate_jaxfluids.sh"
echo ""
echo "To use:"
echo "  source ${VENV_DIR}/activate_jaxfluids.sh"
echo "  python -c \"import jaxfluids; print('OK')\""