# flake.nix — HydroGym deterministic packaging
#
# Outputs:
#   .#jax-cuda-hopper-blackwell           (Phase 1)
#   .#jax-cuda-turing-ampere              (Phase 1)
#   .#jaxfluids-cuda-hopper-blackwell     (Phase 2, added in Task 12)
#   .#jaxfluids-cuda-turing-ampere        (Phase 2, added in Task 12)
#   .#nek-cpu-MiniChannel                 (Phase 3, added in Task 17)
#
# Reproducibility anchor: flake.lock (committed).
#
# Usage (inside the hydrogym/base:dev container, with --gpus all):
#   nix develop .#jax-cuda-hopper-blackwell

{
  description = "HydroGym deterministic per-backend dev shells";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;       # NVHPC SDK is unfree
          overlays = [
            (final: prev: {
              nvhpc-sdk = final.callPackage ./nix/overlays/nvhpc-26.1.nix { };
            })
          ];
        };

        cudaTargets = import ./nix/lib/cudaTargets.nix;
        mkBackendShell = pkgs.callPackage ./nix/lib/mkBackendShell.nix { };

        mkJaxShell = cudaTarget:
          pkgs.callPackage ./nix/backends/jax/default.nix {
            inherit mkBackendShell cudaTarget;
          };
      in
      {
        devShells = {
          jax-cuda-hopper-blackwell = mkJaxShell cudaTargets.hopper-blackwell;
          jax-cuda-turing-ampere    = mkJaxShell cudaTargets.turing-ampere;
        };

        # Default shell falls back to the most common GPU family.
        devShells.default = self.devShells.${system}.jax-cuda-hopper-blackwell;
      });
}
