# nix/backends/jaxfluids/default.nix
#
# JAX-Fluids backend Nix dev shell.
#
# Python dep list = pyproject.toml [tool.poetry.extras].jaxfluids
#   plus core HydroGym runtime plus jaxfluids_rl from the overlay.

{ pkgs, lib, mkBackendShell, cudaTarget }:

let
  jaxfluidsRl = pkgs.python312Packages.callPackage ../../overlays/jaxfluids-rl.nix { };
in
mkBackendShell {
  name = "jaxfluids-cuda-${cudaTarget.name}";
  python = pkgs.python312;
  inherit cudaTarget;

  pythonDeps = pyPkgs: with pyPkgs; [
    # Core HydroGym
    numpy scipy pandas gymnasium huggingface-hub control dmsuite
    # JAX-Fluids extras
    flax gitpython h5py optax
    # Required to install jax[cuda12-local] on shell entry
    pip setuptools wheel
    # Vendored
    jaxfluidsRl
  ];

  extraInputs = [
    pkgs.git
    pkgs.nvhpc-sdk
  ];

  extraShellHook = ''
    if ! python -c "import jax" 2>/dev/null; then
      echo "Installing jax[cuda12-local] from PyPI..."
      pip install --user --quiet "jax[cuda12-local]==0.4.34"
    fi
    export PATH="$HOME/.local/bin:$PATH"
  '';
}
