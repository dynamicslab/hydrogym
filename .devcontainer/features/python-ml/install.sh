#!/bin/bash
# Python ML Stack Feature Install Script
# Installs PyTorch (CUDA 12.8), JAX (CUDA 12), stable-baselines3 in venv
# Uses pip-managed CUDA independent of system CUDA 13.2

set -euo pipefail

# Feature options
# The devcontainer CLI injects option values as env vars named by
# concatenating the camelCase option id in uppercase with no separator
# (e.g. "pytorchCuda" -> PYTORCHCUDA, not PYTORCH_CUDA) - read from those,
# or a configured option silently falls back to the default.
PYTORCH_CUDA="${PYTORCHCUDA:-cu128}"
JAX_CUDA="${JAXCUDA:-cuda12}"

echo "=== Python ML Stack Feature Installation ==="
echo "PyTorch CUDA: ${PYTORCH_CUDA}"
echo "JAX CUDA: ${JAX_CUDA}"

# Create venv with --system-site-packages to reuse base mpi4py, h5py
VENV_DIR="/opt/venvs/ml"
echo "Creating venv at ${VENV_DIR} with --system-site-packages..."
uv venv "${VENV_DIR}" --system-site-packages --seed

# Activate venv
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA 12.8
echo "Installing PyTorch (${PYTORCH_CUDA})..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

# 2. Install JAX with CUDA 12
echo "Installing JAX (${JAX_CUDA})..."
pip install "jax[${JAX_CUDA}]"

# 3. Install stable-baselines3 with extra dependencies
echo "Installing stable-baselines3[extra]..."
pip install stable-baselines3[extra]

# 4. Install common ML utilities
echo "Installing ML utilities..."
pip install gymnasium tensorboard optuna wandb

# Verify installations
echo "Verifying installations..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "
import jax
print(f'JAX: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'JAX backend: {jax.default_backend()}')
"

python -c "
import stable_baselines3
print(f'stable-baselines3: {stable_baselines3.__version__}')
"

python -c "
import gymnasium
import tensorboard
import optuna
print('ML utilities OK')
"

# Deactivate
deactivate

# Create activation helper script
cat > "${VENV_DIR}/activate_ml.sh" << 'EOF'
#!/bin/bash
# Source this to activate the Python ML environment
source /opt/venvs/ml/bin/activate
# Note: JAX/PyTorch manage their own CUDA libraries via pip
# System CUDA 13.2 (NVHPC) remains available for MAIA GPU
echo "Python ML environment activated"
echo "PyTorch/JAX use pip-managed CUDA 12.8"
echo "System CUDA 13.2 (NVHPC) available for MAIA GPU"
EOF
chmod +x "${VENV_DIR}/activate_ml.sh"

echo "=== Python ML Stack Feature Installation Complete ==="
echo "Venv: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/activate_ml.sh"
echo ""
echo "To use:"
echo "  source ${VENV_DIR}/activate_ml.sh"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo "  python -c \"import jax; print(jax.devices())\""