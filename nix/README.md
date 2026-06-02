# HydroGym Nix Packaging

This directory holds the deterministic, per-backend dev shells for
HydroGym, defined as Nix flake outputs at the repo root (`flake.nix`).
The `flake.lock` file is the reproducibility anchor — same lockfile +
same `--system x86_64-linux` ⇒ bit-identical solver environment.

## Layout

```
flake.nix                      # Root: composes everything, exports outputs
nix/
├── lib/
│   ├── cudaTargets.nix        # PTX/SASS tables: hopper-blackwell, turing-ampere
│   └── mkBackendShell.nix     # Factory for per-backend mkShell derivations
├── overlays/
│   ├── nvhpc-26.1.nix         # NVHPC SDK 26.1 (CUDA 12.9) FOD
│   └── jaxfluids-rl.nix       # buildPythonPackage for vendored jaxfluids_rl
└── backends/
    ├── jax/default.nix        # JAX dep set + XLA setup
    ├── jaxfluids/default.nix  # JAX-Fluids dep set + jaxfluids_rl
    └── nek/
        ├── default.nix        # NEK shell factory (case-parameterised)
        ├── nek5000.nix        # Builds nek5000 binary per case
        └── cases/MiniChannel/ # Case-specific SIZE/.usr/.par files
```

## Outputs

Use inside the `hydrogym/base:dev` container (built from
`.packaging/Dockerfile.base`) with `--gpus all` for the CUDA ones:

| Output | Compute | Notes |
|---|---|---|
| `.#jax-cuda-hopper-blackwell` | NVIDIA Hopper, Blackwell | Pinned: jax 0.4.34 |
| `.#jax-cuda-turing-ampere`    | NVIDIA Turing, Ampere    | Pinned: jax 0.4.34 |
| `.#jaxfluids-cuda-hopper-blackwell` | NVIDIA Hopper, Blackwell | + jaxfluids_rl from submodule |
| `.#jaxfluids-cuda-turing-ampere`    | NVIDIA Turing, Ampere    | + jaxfluids_rl from submodule |
| `.#nek-cpu-MiniChannel`             | CPU-only                 | nek5000 + mpi4py + mpich one closure |

## Adding a new GPU arch family

1. Add an entry to `nix/lib/cudaTargets.nix`. Pick `name`, `nvccGencode`,
   `nvhpcGpuArch`, `xlaCudaTargets` from the relevant NVHPC SDK docs.
2. Add output entries in `flake.nix` referencing the new target. The
   `mkBackendShell` factory and per-backend `default.nix` files need no
   changes.

## Adding a new NEK case

1. Drop `SIZE`, `<case>.usr`, `<case>.par` under
   `nix/backends/nek/cases/<case>/`.
2. Add an output entry in `flake.nix` invoking `mkNekShell { case = "<case>"; ... }`.

## Adding a new backend

1. Create `nix/backends/<backend>/default.nix` returning a function that
   takes `{ pkgs, mkBackendShell, cudaTarget }` (or omits `cudaTarget`
   for CPU backends) and returns the result of `mkBackendShell { ... }`.
2. Wire it into `flake.nix` outputs.
3. If the backend has a third-party dep not on PyPI, vendor it as a git
   submodule under `third_party/<dep>/` and add a `buildPythonPackage`
   overlay under `nix/overlays/`.

## Adding Firedrake

Phase 4 of the migration. Native Nix build: PETSc (real + complex),
SLEPc, MPICH, Firedrake. See follow-up spec.

## Why this layout

- Files that change together live together (per-backend bundles).
- The factory keeps boilerplate low: adding a GPU arch is one entry,
  not a copy-pasted flake output.
- The `flake.lock` is the only file that needs to change to update an
  input; backend definitions stay stable.
