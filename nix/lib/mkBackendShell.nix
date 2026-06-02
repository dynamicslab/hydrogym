# nix/lib/mkBackendShell.nix
#
# Factory for per-backend Nix dev shells.
#
# Each backend's default.nix calls this with:
#   - name:            string, e.g. "jax-cuda-hopper-blackwell"
#   - python:          a Python interpreter derivation (e.g. pkgs.python312)
#   - pythonDeps:      function pyPkgs -> [pyPkgs.foo pyPkgs.bar ...]
#   - extraInputs:     [ pkgs.someTool ... ] non-Python build inputs
#   - cudaTarget:      attribute set from nix/lib/cudaTargets.nix, or null for CPU
#   - extraShellHook:  string appended to the standard shellHook
#
# The factory adds the standard HydroGym shellHook that pip-installs the
# /opt/hydrogym checkout in editable mode (--no-deps so Nix-provided
# deps aren't re-resolved).

{ pkgs, lib }:

{ name
, python
, pythonDeps
, extraInputs ? [ ]
, cudaTarget ? null
, extraShellHook ? ""
}:

let
  pythonEnv = python.withPackages pythonDeps;

  cudaSetup =
    if cudaTarget == null then ""
    else ''
      export CUDA_VISIBLE_DEVICES=''${CUDA_VISIBLE_DEVICES:-all}
      export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/cuda/12.9 $XLA_FLAGS"
      export LD_LIBRARY_PATH="${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/cuda/12.9/lib64:${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib:$LD_LIBRARY_PATH"
      export PATH="${pkgs.nvhpc-sdk}/bin:$PATH"
      echo "CUDA target: ${cudaTarget.name}  (XLA targets: ${cudaTarget.xlaCudaTargets})"
    '';

in
pkgs.mkShell {
  inherit name;

  buildInputs = [ pythonEnv ] ++ extraInputs;

  shellHook = ''
    set -e
    ${cudaSetup}

    # Editable install of HydroGym from the bind-mounted source.
    if [ -d /opt/hydrogym ] && [ ! -f /opt/hydrogym/.nix-shell-installed ]; then
      echo "Installing HydroGym in editable mode..."
      ${pythonEnv}/bin/pip install -e /opt/hydrogym --no-deps --quiet
      touch /opt/hydrogym/.nix-shell-installed
    fi

    export PS1="\[\e[1;34m\][${name}]\[\e[0m\] \w \$ "

    ${extraShellHook}
    set +e
  '';
}
