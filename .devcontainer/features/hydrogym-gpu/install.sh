#!/bin/bash
# HydroGym GPU Feature Install Script
# Installs JAX-Fluids and HydroGym[maia,jax] in the python-ml venv
# Requires python-ml feature to be installed first

set -euo pipefail

# Feature options
# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. "hydrogymExtras" -> HYDROGYMEXTRAS, not HYDROGYM_EXTRAS) - read
# from those, or a configured option silently falls back to the default.
HYDROGYM_EXTRAS="${HYDROGYMEXTRAS:-maia,jax}"
JAXFLUIDS_REPO="${JAXFLUIDSREPO:-https://github.com/tumaer/JAXFLUIDS.git}"

echo "=== HydroGym GPU Feature Installation ==="
echo "HydroGym extras: ${HYDROGYM_EXTRAS}"
echo "JAX-Fluids repo: ${JAXFLUIDS_REPO}"

# Persisted for postCreateCommand.sh, which installs HydroGym itself once
# /workspace is live (see note near the end of this script).
mkdir -p /opt/maia-feature-config
cat > /opt/maia-feature-config/hydrogym-gpu.env <<EOF
HYDROGYM_EXTRAS=${HYDROGYM_EXTRAS}
EOF

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
echo "Installing JAX-Fluids from ${JAXFLUIDS_REPO}..."
pip install -e "git+${JAXFLUIDS_REPO}#egg=jaxfluids"

# HydroGym itself is installed in postCreateCommand instead of here: this
# devcontainer bind-mounts the actual hydrogym repo to /workspace
# at container runtime (see devcontainer.json "workspaceMount"), but that
# mount isn't attached yet during this build-time install step - cloning a
# second copy here would just get shadowed by the real mount later and
# waste build time.

# Verify installations
echo "Verifying installations..."
python -c "
import jaxfluids
print(f'JAX-Fluids: {jaxfluids.__version__ if hasattr(jaxfluids, \"__version__\") else \"installed\"}')
"

python -c "
import jax
import torch
print('JAX + PyTorch available')
print(f'JAX devices: {jax.devices()}')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"

# Deactivate
deactivate

# Create activation helper script
cat > "${VENV_DIR}/activate_hydrogym_gpu.sh" << 'EOF'
#!/bin/bash
# Source this to activate the HydroGym GPU environment
source /opt/venvs/ml/bin/activate
# JAX/PyTorch manage their own CUDA via pip
echo "HydroGym GPU environment activated"
echo "JAX-Fluids, HydroGym[maia,jax], PyTorch, JAX ready"
EOF
chmod +x "${VENV_DIR}/activate_hydrogym_gpu.sh"

echo "=== HydroGym GPU Feature Installation Complete ==="
echo "Venv: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/activate_hydrogym_gpu.sh"
echo ""
echo "To use:"
echo "  source ${VENV_DIR}/activate_hydrogym_gpu.sh"
echo "  python -c \"import hydrogym; import jaxfluids; print('OK')\""