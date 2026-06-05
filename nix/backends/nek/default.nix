# nix/backends/nek/default.nix
#
# Factory for case-parameterised NEK5000 dev shells.
#
# Each case produces an `.#nek-cpu-<case>` devshell with the nek5000
# binary (compiled for that case's SIZE file) plus an mpi4py-enabled
# Python on PATH, both linked against pkgs.mpich.
#
# NOTE: this factory is currently UNUSED — the root flake does not
# instantiate any case yet because case-specific files (SIZE, *.usr,
# *.par) for the canonical TCFmini_3D_Re180 environment live outside
# this repo. The factory and the nek5000 builder ship together so a
# follow-up PR (or downstream user) can drop case files at
# `nix/backends/nek/cases/<case>/{SIZE,*.usr,*.par}` and add a
# corresponding `mkNekShell` call in `flake.nix` — see `nix/README.md`
# for the recipe.
#
# Layered Python dep model mirrors the JAX-Fluids backend: pure-Python
# packages stay in Nix, numpy-C-extension packages come from pip (so
# they match whatever numpy version the rest of the venv resolves to).

{ pkgs, lib, mkBackendShell }:

{ case, sizeFile, usrFile, parFile, nek5000Src }:

let
  nek5000Bin = pkgs.callPackage ./nek5000.nix { } {
    inherit case sizeFile usrFile parFile nek5000Src;
  };
in
mkBackendShell {
  name = "nek-cpu-${case}";
  python = pkgs.python312;
  cudaTarget = null;          # CPU-only

  pythonDeps = pyPkgs: with pyPkgs; [
    # Pure-Python deps stay in Nix (no numpy C-ext, no ABI sensitivity).
    gymnasium huggingface-hub omegaconf toml
    # NEK extras from pyproject.toml [tool.poetry.extras].nek that are
    # in nixpkgs and safe to ship from Nix (pure Python or
    # numpy-version-agnostic). pymech is not in nixpkgs — pip below.
    pettingzoo
    # mpi4py MUST come from Nix so it's linked against the same
    # pkgs.mpich the nek5000 binary was built against.
    mpi4py
    # Required to bootstrap the pip layer:
    pip setuptools wheel
  ];

  extraInputs = [
    pkgs.git
    pkgs.mpich
    nek5000Bin
  ];

  extraShellHook = ''
    # C-extension + numpy-dependent packages come from pip (same pattern
    # as the JAX-Fluids backend — nixpkgs-24.05 builds these against
    # numpy 1.x and they break against pip-installed numpy 2.x). control
    # and dmsuite aren't in nixpkgs at all.
    if ! python -c "import stable_baselines3" 2>/dev/null; then
      echo "Installing NEK pip layer (numpy, scipy, pandas, SB3, pymech, ...)..."
      pip install --quiet \
        "numpy" \
        "scipy" \
        "pandas" \
        "pymech" \
        "stable-baselines3" \
        "supersuit" \
        "tensorboard" \
        "control" \
        "dmsuite"
    fi
    export NEK_CASE=${case}
    export NEK_SHARE=${nek5000Bin}/share/nek/${case}
    echo "NEK5000 case '${case}' ready. Binary: $(which nek5000)"
  '';
}
