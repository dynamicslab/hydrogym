# nix/backends/jax/default.nix
#
# JAX backend Nix dev shell.
#
# Python dep list mirrors pyproject.toml [tool.poetry.extras].jax:
#   jax, jaxlib, chex, navix, gymnax, tree-math, flax, omegaconf, toml
# Plus numpy/scipy/pandas which are core HydroGym deps.
#
# jaxlib comes from the official jax[cuda12] PyPI wheel installed by
# pip on shell entry. The wheel bundles cuBLAS/cuDNN/cuFFT/cuSPARSE/NCCL
# (NVHPC SDK 26.1 ships everything except cuDNN, so we let the wheel own
# the JAX-side CUDA stack rather than mixing sources). NVHPC's nvcc and
# nvfortran remain on PATH for native-code workflows.

{ pkgs, lib, mkBackendShell, cudaTarget }:

mkBackendShell {
  name = "jax-cuda-${cudaTarget.name}";
  python = pkgs.python312;
  inherit cudaTarget;

  pythonDeps = pyPkgs: with pyPkgs; [
    # Core HydroGym runtime
    numpy scipy pandas gymnasium
    huggingface-hub
    # Config + serialization
    omegaconf toml
    # Pip itself, so the full JAX ecosystem can install on shell entry.
    # chex + flax are pip-installed (the nixpkgs flax pulls in a broken
    # einops→jupyter→qtconsole→ipython-genutils chain on Py 3.12).
    pip setuptools wheel
    # SB3 + friends are common in JAX RL workflows but not required
    # by the JAX extras list; users add them with `pip install`.
  ];

  extraInputs = [
    pkgs.git
    pkgs.nvhpc-sdk    # exposed via overlay
  ];

  extraShellHook = ''
    # The mkBackendShell venv is already activated; pip installs land there.
    # JAX 0.10 is the first stable line whose bundled ptxas/libdevice ship
    # sm_120 kernels (Blackwell consumer / RTX 5090). H100 sm_90 and
    # B100/B200 sm_100 are also covered. Pinned to keep the pip layer
    # deterministic alongside the Nix-pinned NVHPC and nixpkgs.
    if ! python -c "import jax" 2>/dev/null; then
      echo "Installing jax[cuda12] + JAX ecosystem from PyPI..."
      pip install --quiet \
        "jax[cuda12]==0.10.1" \
        "chex" \
        "flax" \
        "control" \
        "dmsuite" \
        "navix" \
        "gymnax" \
        "tree-math"
    fi
  '';
}
