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
    numpy scipy pandas gymnasium huggingface-hub
    # JAX-Fluids extras (flax + optax come from pip — nixpkgs flax pulls
    # in a Py3.12-incompatible einops→jupyter→qtconsole chain)
    gitpython h5py
    # Required to install jax[cuda12-local] (+ flax, optax, control, dmsuite)
    # on shell entry
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
      echo "Installing jax[cuda12-local] + JAX ecosystem from PyPI..."
      pip install --user --quiet \
        "jax[cuda12-local]==0.4.34" \
        "flax" \
        "optax" \
        "control" \
        "dmsuite"
    fi
    export PATH="$HOME/.local/bin:$PATH"
  '';
}
