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
      # Leave CUDA_VISIBLE_DEVICES UNSET by default — its valid values are
      # device IDs/UUIDs, not "all". Inside docker --gpus all the
      # nvidia-container-toolkit injects the right UUIDs already, and on a
      # bare host an unset var means "all devices visible".
      export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/cuda/12.9 $XLA_FLAGS"
      export LD_LIBRARY_PATH="${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/cuda/12.9/lib64:${pkgs.nvhpc-sdk}/Linux_x86_64/26.1/comm_libs/12.9/nccl/lib:$LD_LIBRARY_PATH"
      export PATH="${pkgs.nvhpc-sdk}/bin:$PATH"
      # JAX/CUDA need libcuda.so.1 + libnvidia-ml.so.1 from the host driver.
      # Their dirs (/usr/lib/x86_64-linux-gnu, etc.) are full of OTHER system
      # libs that would shadow nix-store versions and SIGFPE the loader, so
      # we build a per-shell symlink farm containing ONLY those two and put
      # just that dir on LD_LIBRARY_PATH. Locations probed:
      #   /run/opengl-driver/lib    — NixOS
      #   /usr/lib/x86_64-linux-gnu — Debian/Ubuntu multiarch
      #   /usr/lib64                — RHEL/Fedora/openSUSE
      NV_LIB_DIR="$VENV_DIR/.nv-driver-libs"
      if [ ! -d "$NV_LIB_DIR" ]; then
        mkdir -p "$NV_LIB_DIR"
        for nvdriver in /run/opengl-driver/lib /usr/lib/x86_64-linux-gnu /usr/lib64; do
          if [ -e "$nvdriver/libcuda.so.1" ]; then
            ln -sf "$nvdriver/libcuda.so.1" "$NV_LIB_DIR/libcuda.so.1"
            [ -e "$nvdriver/libnvidia-ml.so.1" ] && \
              ln -sf "$nvdriver/libnvidia-ml.so.1" "$NV_LIB_DIR/libnvidia-ml.so.1"
            break
          fi
        done
      fi
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NV_LIB_DIR"
      echo "CUDA target: ${cudaTarget.name}  (XLA targets: ${cudaTarget.xlaCudaTargets})"
    '';

in
pkgs.mkShell {
  inherit name;

  buildInputs = [ pythonEnv ] ++ extraInputs;

  shellHook = ''
    set -e
    # PyPI wheels link against system shared libs the Nix-built Python
    # doesn't know about. Seed LD_LIBRARY_PATH with the common runtime deps
    # before any backend-specific CUDA setup. Order: NVHPC libs > C++/zlib
    # runtime > caller's existing path. Add more here as wheels demand them.
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:''${LD_LIBRARY_PATH:-}"

    # python.withPackages builds a read-only env in /nix/store, so pip
    # refuses `--user` installs and can't write `-e` editable links there.
    # Bootstrap a writable venv layered over the Nix env (--system-site-packages
    # inherits numpy/scipy/etc. so we don't reinstall them). Backends drop
    # their pure-PyPI deps into this venv via extraShellHook. VENV_DIR is
    # also referenced by cudaSetup below for the driver-libs symlink farm,
    # so it has to be set before the CUDA env block runs.
    VENV_DIR="''${HYDROGYM_VENV_DIR:-$HOME/.cache/hydrogym/${name}-venv}"
    if [ ! -d "$VENV_DIR" ]; then
      ${pythonEnv}/bin/python -m venv "$VENV_DIR"
      # `--system-site-packages` only inherits from the *unwrapped* python3
      # site-packages, not from python.withPackages's wrapped env. Use a
      # .pth file instead so the wrapped env's site-packages (omegaconf,
      # huggingface-hub, etc.) is searched AFTER the venv's own — that way
      # pip-upgrades inside the venv take precedence and we still get
      # Nix-provided packages for free when no pip version is present.
      echo "${pythonEnv}/${python.sitePackages}" \
        > "$VENV_DIR/${python.sitePackages}/nix-env.pth"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    ${cudaSetup}

    # Editable install of HydroGym from the bind-mounted source — goes into
    # the venv, not the read-only Nix env.
    if [ -d /opt/hydrogym ] && [ ! -f "$VENV_DIR/.hydrogym-installed" ]; then
      echo "Installing HydroGym in editable mode..."
      pip install -e /opt/hydrogym --no-deps --quiet
      touch "$VENV_DIR/.hydrogym-installed"
    fi

    export PS1="\[\e[1;34m\][${name}]\[\e[0m\] \w \$ "

    ${extraShellHook}
    set +e
  '';
}
