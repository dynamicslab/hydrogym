# HydroGym Nix Packaging

This directory holds the deterministic, per-backend dev shells for
HydroGym, defined as Nix flake outputs at the repo root (`flake.nix`).
The `flake.lock` file is the reproducibility anchor — same lockfile +
same `--system x86_64-linux` ⇒ bit-identical solver environment.

## Layout

```
flake.nix                      # Root: composes everything, exports outputs
third_party/nek5000/           # Submodule, pinned to Nek5000 v19.0
nix/
├── lib/
│   ├── cudaTargets.nix        # PTX/SASS tables: hopper-blackwell, turing-ampere
│   └── mkBackendShell.nix     # Factory for per-backend mkShell derivations
├── overlays/
│   └── nvhpc-26.1.nix         # NVHPC SDK 26.1 (CUDA 12.9 + 13.1) FOD
└── backends/
    ├── jax/default.nix        # JAX dep set + XLA setup
    ├── jaxfluids/default.nix  # JAX-Fluids dep set + JAXFLUIDS umbrella
    │                          #   (pip-installed from a pinned git rev;
    │                          #    no Nix overlay — jaxfluids_rl is a
    │                          #    sub-package of tumaer/JAXFLUIDS, not
    │                          #    standalone)
    └── nek/
        ├── default.nix        # NEK shell factory (case-parameterised)
        ├── nek5000.nix        # Builds nek5000 binary per case
        └── cases/             # Drop SIZE/.usr/.par here per case
```

## Outputs

Use inside the `hydrogym/base:dev` container (built from
`.packaging/Dockerfile.base`) with `--gpus all` for the CUDA ones:

| Output | Compute | Notes |
|---|---|---|
| `.#jax-cuda-hopper-blackwell` | NVIDIA Hopper, Blackwell | Pinned: jax[cuda12] 0.10.1 |
| `.#jax-cuda-turing-ampere`    | NVIDIA Turing, Ampere    | Pinned: jax[cuda12] 0.10.1 |
| `.#jaxfluids-cuda-hopper-blackwell` | NVIDIA Hopper, Blackwell | + JAXFLUIDS pinned to main rev |
| `.#jaxfluids-cuda-turing-ampere`    | NVIDIA Turing, Ampere    | + JAXFLUIDS pinned to main rev |

The `nix/backends/nek/` builder + factory are wired up but **no NEK
shell is currently exposed in `flake.nix`** — case-specific files
(SIZE, `<case>.usr`, `<case>.par`) live outside this repo for now.
Drop a case in and follow the recipe below to expose it.

## Adding a new GPU arch family

1. Add an entry to `nix/lib/cudaTargets.nix`. Pick `name`, `nvccGencode`,
   `nvhpcGpuArch`, `xlaCudaTargets` from the relevant NVHPC SDK docs.
2. Add output entries in `flake.nix` referencing the new target. The
   `mkBackendShell` factory and per-backend `default.nix` files need no
   changes.

## Adding a new NEK case

1. Drop `SIZE`, `<case>.usr`, `<case>.par` under
   `nix/backends/nek/cases/<case>/` — by convention `<case>` matches the
   HuggingFace environment directory name (e.g. `TCFmini_3D_Re180`).
2. In `flake.nix`, add a `mkNekShell` factory next to the JAX ones and
   wire a per-case output:

   ```nix
   mkNekShell = { case, sizeFile, usrFile, parFile }:
     (pkgs.callPackage ./nix/backends/nek/default.nix {
       inherit mkBackendShell;
     }) {
       inherit case sizeFile usrFile parFile;
       nek5000Src = ./third_party/nek5000;
     };

   devShells = {
     # ... existing JAX shells ...
     nek-cpu-TCFmini_3D_Re180 = mkNekShell {
       case = "TCFmini_3D_Re180";
       sizeFile = ./nix/backends/nek/cases/TCFmini_3D_Re180/SIZE;
       usrFile  = ./nix/backends/nek/cases/TCFmini_3D_Re180/TCFmini_3D_Re180.usr;
       parFile  = ./nix/backends/nek/cases/TCFmini_3D_Re180/TCFmini_3D_Re180.par;
     };
   };
   ```
3. (Optional) Extend `.github/workflows/nix-build.yml` to build the new
   shell on CI.

## Adding a new backend

1. Create `nix/backends/<backend>/default.nix` returning a function that
   takes `{ pkgs, mkBackendShell, cudaTarget }` (or omits `cudaTarget`
   for CPU backends) and returns the result of `mkBackendShell { ... }`.
2. Wire it into `flake.nix` outputs.
3. For third-party deps not on PyPI, prefer pip-install from a pinned
   git URL in the backend's `extraShellHook` (matches the JAXFLUIDS
   pattern). Reach for a Nix `buildPythonPackage` overlay under
   `nix/overlays/` only if you need Nix-store caching or the package
   has C extensions sensitive to numpy ABI.

## Adding Firedrake

Phase 4 of the migration. Native Nix build: PETSc (real + complex),
SLEPc, MPICH, Firedrake. See follow-up spec.

## Why this layout

- Files that change together live together (per-backend bundles).
- The factory keeps boilerplate low: adding a GPU arch is one entry,
  not a copy-pasted flake output.
- The `flake.lock` is the only file that needs to change to update an
  input; backend definitions stay stable.
