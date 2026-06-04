# nix/backends/jaxfluids/default.nix
#
# JAX-Fluids backend Nix dev shell.
#
# Python dep list = pyproject.toml [tool.poetry.extras].jaxfluids
#   plus core HydroGym runtime plus the upstream JAXFLUIDS umbrella
#   (jaxfluids, jaxfluids_rl, jaxfluids_thirdparty — HydroGym imports
#   from all three).
#
# JAXFLUIDS is installed from a pinned git rev rather than via a Nix
# overlay. The plan called for a submodule + buildPythonPackage, but
# jaxfluids_rl is not a standalone repo (sub-package of tumaer/JAXFLUIDS,
# shipped together via setup.py's find_packages), and the umbrella's
# install_requires drags jax/jaxlib/flax/optax which conflict with the
# pip-pinned cuda12 wheels. Following the same pip pattern we already
# use for jax/flax/chex avoids that conflict and matches the rest of
# the JAX ecosystem story.

{ pkgs, lib, mkBackendShell, cudaTarget }:

let
  # Pinned to the first main-branch commit that ships src/jaxfluids_rl/.
  # The v0.2.1 tag predates jaxfluids_rl, so we can't use a release tag.
  jaxfluidsRev = "9bb1e6c85371d445cbdaaaf9e5699495cff4b371";
in
mkBackendShell {
  name = "jaxfluids-cuda-${cudaTarget.name}";
  python = pkgs.python312;
  inherit cudaTarget;

  pythonDeps = pyPkgs: with pyPkgs; [
    # Pure-Python deps stay in Nix (no numpy C-ext, no ABI sensitivity).
    gymnasium huggingface-hub gitpython omegaconf toml
    # C-extension deps that link against numpy must come from pip: the
    # nixpkgs-24.05 builds target numpy 1.x and break against the numpy
    # 2.x that jax[cuda12]==0.10.1 requires. That covers numpy, scipy,
    # pandas, h5py, matplotlib, plus flax/optax (transitive numpy via JAX).
    # Required to bootstrap the pip layer:
    pip setuptools wheel
  ];

  extraInputs = [
    pkgs.git
    pkgs.nvhpc-sdk
  ];

  extraShellHook = ''
    # The mkBackendShell venv is already activated; pip installs land there.
    # Match the JAX backend's pin (0.10.x covers sm_90/sm_100/sm_120).
    if ! python -c "import jax" 2>/dev/null; then
      echo "Installing jax[cuda12] + JAX/JAXFLUIDS ecosystem from PyPI..."
      pip install --quiet \
        "jax[cuda12]==0.10.1" \
        "flax" \
        "optax" \
        "scipy" \
        "pandas" \
        "h5py" \
        "matplotlib" \
        "pyvista" \
        "control" \
        "dmsuite"
    fi
    if ! python -c "import jaxfluids_rl" 2>/dev/null; then
      echo "Installing JAXFLUIDS umbrella (jaxfluids + jaxfluids_rl + jaxfluids_thirdparty) from upstream main @ ${jaxfluidsRev}..."
      pip install --quiet --no-deps \
        "git+https://github.com/tumaer/JAXFLUIDS.git@${jaxfluidsRev}"
    fi
  '';
}
