# nix/backends/jax/default.nix
#
# JAX backend Nix dev shell.
#
# Python dep list mirrors pyproject.toml [tool.poetry.extras].jax:
#   jax, jaxlib, chex, navix, gymnax, tree-math, flax, omegaconf, toml
# Plus numpy/scipy/pandas which are core HydroGym deps.
#
# jaxlib comes from the official jax[cuda12-local] PyPI wheel installed by
# pip on shell entry — nixpkgs jaxlib is not built against CUDA 12.9 at
# time of writing. The wheel discovers system CUDA libs at runtime via
# LD_LIBRARY_PATH set by mkBackendShell.

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
    # jax + jaxlib (CUDA 12 local) installed into the user's home so the
    # editable HydroGym install can find them. Nix-provided python has pip.
    if ! python -c "import jax" 2>/dev/null; then
      echo "Installing jax[cuda12-local] + JAX ecosystem from PyPI..."
      pip install --user --quiet \
        "jax[cuda12-local]==0.4.34" \
        "chex" \
        "flax" \
        "control" \
        "dmsuite" \
        "navix" \
        "gymnax" \
        "tree-math"
    fi
    export PATH="$HOME/.local/bin:$PATH"
  '';
}
