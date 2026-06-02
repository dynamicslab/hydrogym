# Deterministic Nix-Flake Packaging for HydroGym

**Status:** Approved design — ready for implementation planning
**Date:** 2026-06-02
**Authors:** Ludger Paehler (with Claude Code)
**PR scope:** Phases 1–3 (JAX, JAX-Fluids, NEK + base image + flake skeleton). Firedrake → follow-up PR. MAIA → deferred.

## Motivation

The existing `.packaging/` directory contains four Firedrake-only Dockerfiles that chain through three Docker Hub images (`lpaehler/hydrogym-firedrake-env:stable` → `lpaehler/hydrogym-env:stable` → `…/hydrogym`). They are not reproducible: `firedrake-install` fetches dependencies dynamically at build time, PETSc is built with `-march=native` (non-portable), and the chained `:stable` tags make rollback impossible.

For MAIA, NEK, JAX, and JAX-Fluids, the repo contains no Dockerfiles at all. The example shell scripts under `examples/<backend>/getting_started/run_*_docker.sh` are misnamed — they actually do `module load <Easybuild module>` and source venvs under `/home/easybuild/venvs/...`, which only works on the specific HPC system the user develops on. Outside that environment, the repo has no reproducible packaging story for the four non-Firedrake backends.

This proposal replaces the current packaging with **one minimal base Docker image** plus a **single root-level `flake.nix`** exposing one Nix dev shell per (backend × compute target). The flake lock file is the reproducibility anchor.

## Goals

1. **Bit-identical environments** across maintainer machines, contributor laptops, CI, and HPC nodes (modulo CUDA driver ABI on the host).
2. **One source of truth** for what each backend needs — no drift between the README's `pip install` lists, the Dockerfiles, the Easybuild modules, and the venvs.
3. **Match the existing `clagemann/hydrogym-nvhpc-26.1_cuda-12.9_*` images** in GPU-target granularity (Hopper/Blackwell vs Turing/Ampere) for the GPU-capable backends.
4. **Co-located source edits:** Editing `hydrogym/*.py` on the host reflects immediately inside the dev shell — no flake rebuild required.

## Non-goals

- Replacing the Easybuild modules on the user's HPC system. Those scripts continue to work; the new flow is additive.
- ROCm support. Deferred to a later flake output.
- A NixOS-based runtime. The base image stays Ubuntu 22.04 for familiarity.
- MAIA flake outputs in this PR. MAIA is currently only distributed as an Easybuild module on a shared HPC, with no portable binary or accessible source. Deferred until that changes.

## Architecture

Three layers, with the flake lock file as the reproducibility anchor.

```
┌──────────────────────────────────────────────────────────┐
│ HOST                                                     │
│  - NVIDIA driver (>= driver ABI-compatible with CUDA 12.9)│
│  - NVIDIA Container Toolkit (nvidia-docker)              │
└────────────────────┬─────────────────────────────────────┘
                     │ docker run --gpus all
┌────────────────────▼─────────────────────────────────────┐
│ BASE DOCKER IMAGE  (hydrogym/base:<tag>)                 │
│  - Ubuntu 22.04                                          │
│  - Multi-user Nix installed via official installer       │
│    (pinned installer version baked into Dockerfile)      │
│  - HydroGym source at /opt/hydrogym                      │
│  - HF cache mount point: /root/.cache/hydrogym           │
│  - NO Python, NO solver libs, NO CUDA in the image       │
│  Built from: .packaging/Dockerfile.base                  │
└────────────────────┬─────────────────────────────────────┘
                     │ nix develop .#<output>
┌────────────────────▼─────────────────────────────────────┐
│ NIX DEV SHELL  (per backend × per compute target)        │
│  Reproducibility anchor: flake.lock                      │
│  Owns: Python 3.12, all solver Python deps, NVHPC SDK,  │
│        CUDA libs, NCCL, MPI (for MPMD backends)          │
│  Phase 1–3 outputs:                                      │
│    .#jax-cuda-hopper-blackwell                           │
│    .#jax-cuda-turing-ampere                              │
│    .#jaxfluids-cuda-hopper-blackwell                     │
│    .#jaxfluids-cuda-turing-ampere                        │
│    .#nek-cpu-<case>          (one per supported case)    │
└──────────────────────────────────────────────────────────┘
```

### Reproducibility contract

The base Docker image is **not** the anchor — it is an Ubuntu shell that happens to contain Nix and a HydroGym checkout. It can be rebuilt freely without breaking determinism, provided the Nix installer version is pinned in the Dockerfile.

The anchor is `flake.lock`. Given the same lockfile and the same `--system x86_64-linux`, every Nix shell evaluates to a bit-identical store path:

- NVHPC SDK 26.1 → `fetchurl` with SHA256 (assumes a public NVIDIA URL; verify before implementation).
- jaxlib → installed via the `jax[cuda12-local]` PyPI wheel with hash-pinning; `LD_LIBRARY_PATH` points at NVHPC's CUDA libs from the same closure.
- NEK5000 source → git submodule with rev pin, built into a Nix derivation per case.
- HydroGym source → bind-mounted from `/opt/hydrogym` into the dev shell; installed editable via `pip install -e .` in a shellHook on first entry.

### HydroGym installation policy

HydroGym lives in the **base image**, not the flake. The flake provides Python + all dependencies; the dev shell's `shellHook` runs `pip install -e /opt/hydrogym --no-deps` on first entry. This means:

- Editing `hydrogym/jax/envs/kolmogorov.py` on the host is reflected immediately in the running shell.
- Bumping a solver dependency (e.g. JAX version) requires a flake update, not an image rebuild.
- The base image's `/opt/hydrogym` is the default; users override via `-v $HOST_PATH:/opt/hydrogym` to work on a different checkout.

### GPU runtime contract

The flake brings NVHPC SDK 26.1 (containing CUDA 12.9, NCCL, math libs) into `/nix/store`. The host's NVIDIA driver is passed through via `--gpus all`. **Required host driver:** ABI-compatible with CUDA 12.9 (≈ driver 555.x or newer; pin exact minimum during phase 1 validation).

Two GPU output families correspond to GPU architecture (PTX/SASS targeting). The concrete `compute_XX` / `sm_XX` lists are determined by what NVHPC SDK 26.1 supports and finalised during phase 1; the families themselves are:

- `*-cuda-hopper-blackwell` — Hopper (datacenter, sm_90 family) and Blackwell (datacenter, sm_100 family).
- `*-cuda-turing-ampere` — Turing (sm_75) and Ampere (sm_80, sm_86).

Selection happens at flake-output time, not at runtime — the closures differ in the AOT-compiled kernels they ship. The split mirrors the existing `clagemann/hydrogym-nvhpc-26.1_cuda-12.9_{hopper_blackwell,turing_ampere}:latest` image tags one-for-one.

## Per-backend output matrix

| Backend | Day-one outputs | Phase | Compute target | Notes |
|---|---|---|---|---|
| jax | `jax-cuda-hopper-blackwell`, `jax-cuda-turing-ampere` | 1 | GPU | Pure-Python, no MPI. The prototype. |
| jaxfluids | `jaxfluids-cuda-hopper-blackwell`, `jaxfluids-cuda-turing-ampere` | 2 | GPU | Adds `jaxfluids_rl` as git submodule. |
| nek | `nek-cpu-<case>` per supported case | 3 | CPU | Case-parameterised; MPMD with mpi4py + nek5000 in one closure. |
| firedrake | `firedrake-cpu-real`, `firedrake-cpu-complex` | 4 (follow-up PR) | CPU | Native Nix packaging from scratch — PETSc/SLEPc/MPICH/Firedrake as derivations. |
| maia | deferred | — | — | Blocked on portable binary distribution. |

**Day-one outputs in this PR:** 4 GPU + N NEK case outputs.

## Repository layout

```
hydrogym-next/
├── .packaging/
│   ├── Dockerfile.base              # Ubuntu 22.04 + multi-user Nix + HydroGym checkout
│   └── README.md                    # How to build, tag, push the base image
├── flake.nix                        # Top-level: exports all .#<backend>-<target> outputs
├── flake.lock                       # The reproducibility anchor (committed)
├── nix/
│   ├── lib/
│   │   ├── mkBackendShell.nix       # Factory: takes backend + target, returns devShell
│   │   └── cudaTargets.nix          # Hopper/Blackwell vs Turing/Ampere arch flag tables
│   ├── overlays/
│   │   ├── nvhpc-26.1.nix           # NVHPC SDK 26.1 FOD (per arch family)
│   │   ├── jaxlib-cuda.nix          # jaxlib override binding to NVHPC's CUDA libs
│   │   └── jaxfluids-rl.nix         # buildPythonPackage for third_party/jaxfluids_rl
│   ├── backends/
│   │   ├── jax/default.nix          # JAX dep set: jax, chex, navix, gymnax, flax, tree-math
│   │   ├── jaxfluids/default.nix    # JAX-Fluids dep set + jaxfluids_rl from overlay
│   │   └── nek/
│   │       ├── default.nix          # NEK shell factory: mkNekShell { case = "MiniChannel"; ... }
│   │       ├── nek5000.nix          # Builds nek5000 binary from third_party/nek5000 submodule
│   │       └── cases/
│   │           └── MiniChannel/
│   │               ├── SIZE
│   │               ├── <case>.usr
│   │               └── <case>.par
│   └── README.md                    # Architecture doc; how to add a backend or arch family
└── third_party/
    ├── firedrake/                   # Existing submodule (unchanged in this PR; used by phase 4)
    ├── jaxfluids_rl/                # NEW submodule, rev-pinned for FOD reproducibility
    └── nek5000/                     # NEW submodule, rev-pinned
```

**Not deleted in this PR.** The four existing `.packaging/Dockerfile.{firedrake_env,hydrogym_env,hydrogym,devpod}` files stay in place until phase 5 (a follow-up PR after Firedrake's native Nix build lands). Firedrake users continue to pull `lpaehler/hydrogym-firedrake-env:stable` etc. from Docker Hub through phases 1–4. Removing the old packaging is the only irreversible step in the migration and gates explicitly on phase 4 being validated.

**Not touched in this PR:** `examples/<backend>/getting_started/run_*_docker.sh`. These remain HPC module-load scripts and are out of scope. A follow-up PR will add `run_*_nix.sh` siblings that use the dev shell.

### The `mkBackendShell` factory

The key abstraction. Each backend's `default.nix` returns a function that, given a compute target, produces a `pkgs.mkShell`:

```nix
# nix/backends/jax/default.nix (sketch)
{ pkgs, lib, mkBackendShell, cudaTarget }:

mkBackendShell {
  name = "jax-${cudaTarget.name}";
  cudaTarget = cudaTarget;
  pythonDeps = pyPkgs: with pyPkgs; [
    jax
    (jaxlib.override { cudaSupport = true; })
    chex navix gymnax tree-math flax
    omegaconf toml
    numpy scipy pandas
  ];
  extraInputs = [ pkgs.git ];
  shellHookExtra = ''
    export LD_LIBRARY_PATH="${pkgs.nvhpc-26_1}/Linux_x86_64/26.1/cuda/12.9/lib64:$LD_LIBRARY_PATH"
  '';
}
```

Adding a new GPU architecture family or a new NEK case is a single new entry in the top-level `flake.nix` outputs list, plus (for NEK) a new `cases/<name>/` directory. The factory keeps the boilerplate per backend low.

## Per-backend risk and mitigation

### JAX — the prototype

**Risk:** jaxlib's CUDA ABI must match NVHPC's CUDA 12.9.

**Mitigation:** Install jaxlib from the official `jax[cuda12-local]` wheel (which links against system CUDA libs at runtime, not bundled ones). The flake's `shellHook` sets `LD_LIBRARY_PATH` to NVHPC's CUDA 12.9 libs from the same closure. Verified during phase 1 by running `python -c "import jax; print(jax.devices())"` inside the dev shell and confirming a CUDA device is reported.

### JAX-Fluids — third-party `jaxfluids_rl`

**Risk:** `jaxfluids_rl` is not on PyPI in a Nix-friendly form.

**Mitigation:** Add `third_party/jaxfluids_rl` as a git submodule, rev-pinned. Package as `buildPythonPackage { src = ../../third_party/jaxfluids_rl; ... }` in `nix/overlays/jaxfluids-rl.nix`. Submodule rev bumps require a `nix flake update` to refresh the lock.

### NEK5000 — case-specific compile

**Risk:** Each case (e.g. `TCFmini_3D_Re180`) requires its own `SIZE` file and rebuild. The current Easybuild module `Nek5000/1.0-gompi-2024a-SystemCUDA-MiniChannel` encodes the case name in the module identifier — same approach is needed in the flake.

**Mitigation:** `mkNekShell { case = "MiniChannel"; sizeFile = ./cases/MiniChannel/SIZE; usrFile = ./cases/MiniChannel/MiniChannel.usr; ... }`. The flake's top level enumerates supported cases:

```nix
nekCases = [ "MiniChannel" /* ... add more */ ];
```

Each becomes a separate dev shell output. mpi4py and `nek5000` are both built against the same `pkgs.mpich` derivation, satisfying the MPMD invariant that the Python side and the solver binary share an MPI implementation.

### Firedrake (Phase 4, follow-up PR)

**Risk:** `firedrake-install` is opinionated and network-fetching; the existing Dockerfile passes ~15 `--download-*` flags to PETSc (chaco, fftw, hdf5, hwloc, hypre, metis, ml, mumps, mpich, netcdf, pastix, pnetcdf, ptscotch, scalapack, suitesparse, superlu_dist) and builds two scalar variants (real + complex).

**Mitigation (deferred to phase 4 design):** Build PETSc, SLEPc, MPICH, and Firedrake as proper Nix derivations from source. Two PETSc scalar types means two derivations. Multi-week effort; not blocking this PR.

### MAIA (deferred)

**Risk:** No portable binary distribution exists today.

**Mitigation:** None possible in this PR. Reopens when MAIA gains a public URL or a `requireFile`-compatible distribution channel.

## User workflow

```bash
# Build/pull the base image (no GPU needed for this step)
docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .

# Start container with GPU pass-through, repo bind-mounted for live edits
docker run --gpus all -it \
    -v $PWD:/opt/hydrogym \
    -v ~/.cache/hydrogym:/root/.cache/hydrogym \
    hydrogym/base:dev

# Inside container — enter the Nix dev shell for your backend × GPU arch
cd /opt/hydrogym
nix develop .#jax-cuda-hopper-blackwell

# Now python, jax, hydrogym.jax are available, all pinned via flake.lock
python -c "import jax; print(jax.devices())"
python examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py
```

For NEK (MPMD):
```bash
nix develop .#nek-cpu-MiniChannel
# Inside the shell, both mpi4py and nek5000 are on PATH, built against the same MPI
mpirun -np 1 python test_nek_direct.py : -np 10 nek5000
```

## Testing and CI

### Local validation order

1. `nix flake check` — flake evaluates; all derivation hashes are consistent.
2. `nix build .#<output>` per output, with the binary cache warm.
3. `docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .` succeeds.
4. Inside the container: `nix develop .#jax-cuda-hopper-blackwell` and `python -c "import hydrogym.jax, jax; print(jax.devices())"` reports a CUDA device.
5. End-to-end: run `examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py` to completion inside the dev shell.
6. Same flow for `.#jaxfluids-cuda-hopper-blackwell` (and the Turing/Ampere variant on the appropriate hardware).
7. NEK MPMD smoke test: `mpirun -np 1 python ... : -np N nek5000` inside `.#nek-cpu-MiniChannel`.

### CI additions

- **New workflow `nix-build.yml`** runs on every PR:
  - `nix flake check` — evaluates all outputs.
  - `nix build .#nek-cpu-MiniChannel` — the only output buildable on a GitHub-hosted runner (CPU-only, no NVHPC SDK).
  - GPU outputs are validated only at evaluation time (`nix flake show`); actual builds require a self-hosted GPU runner, which is out of scope for this PR.
- **Cachix binary cache** (e.g. `hydrogym.cachix.org`) for the heavy NVHPC SDK and any expensive native builds. Push happens from a maintainer-triggered workflow, not from PRs. Without this, every contributor pays the full multi-hour build on first entry.
- Existing CI (`ruff_lint`, `ruff_format`, `isort`, `codespell`, `build`) unchanged.

### Manual verification before merge

The author must demonstrate on real hardware:
- One Hopper or Blackwell GPU: `jax.devices()` reports CUDA, Kolmogorov example completes.
- One Turing or Ampere GPU: same.
- CPU-only host: NEK MiniChannel MPMD smoke test completes.

This evidence is required in the PR description before merge per the project's `verification-before-completion` discipline.

## Migration phasing

The four phases below are designed so each is independently shippable. Phases 1–3 are this PR; phases 4–5 are follow-ups.

| Phase | Scope | Done when |
|---|---|---|
| 1 | `Dockerfile.base` + `flake.nix` skeleton + `jax-cuda-hopper-blackwell` and `jax-cuda-turing-ampere` outputs | End-to-end Kolmogorov example runs in the dev shell on real GPU |
| 2 | `jaxfluids_rl` submodule + `jaxfluids-cuda-*` outputs | JAX-Fluids nozzle example runs |
| 3 | `nek5000` submodule + `nek-cpu-<case>` factory + `MiniChannel` case | NEK MPMD smoke test passes |
| 4 (follow-up PR) | Native Firedrake build (PETSc real + complex, SLEPc, MPICH, Firedrake) | Cylinder test from `test/test_cyl.py` passes inside `.#firedrake-cpu-real` |
| 5 (follow-up PR) | Delete old `.packaging/Dockerfile.{firedrake_env,hydrogym_env,hydrogym,devpod}`; update top-level README to point at the Nix flow | README install section no longer references `lpaehler/hydrogym-*:stable` |

**Irreversibility checkpoint:** phase 5 is the only step that removes existing infrastructure. It only happens after phase 4 lands and Firedrake users have a working migration path.

**Deferred indefinitely:** MAIA flake outputs.

## Open questions for implementation phase

These are not blockers for the design but must be resolved in the implementation plan (`writing-plans` next):

1. **Exact NVHPC 26.1 URL.** Verify `developer.download.nvidia.com/hpc-sdk/26.1/...` serves the tarball without auth. If not, document the `requireFile` fallback.
2. **Minimum host NVIDIA driver version** for CUDA 12.9 ABI. Pin during phase 1 validation; document in `.packaging/README.md`.
3. **Cachix namespace.** Whether to use `hydrogym.cachix.org` (new), piggy-back on an existing one, or skip Cachix initially and accept slow first builds.
4. **NEK5000 case list.** Which cases beyond `MiniChannel` ship in phase 3 (e.g. `TCFmini_3D_Re180` referenced in the example scripts)?
5. **jaxlib version pin.** Latest jaxlib compatible with CUDA 12.9 at design time; rev-bump policy.

## Acceptance criteria for the PR

- [ ] `.packaging/Dockerfile.base` exists and builds on a clean Ubuntu host with Docker + buildkit.
- [ ] `flake.nix` at repo root evaluates; `nix flake show` lists the four GPU outputs and at least one `nek-cpu-*` output.
- [ ] `flake.lock` is committed and pins all inputs to specific revisions/hashes.
- [ ] `nix develop .#jax-cuda-hopper-blackwell` enters a shell with python, jax, and `hydrogym.jax` importable.
- [ ] One JAX example, one JAX-Fluids example, one NEK MPMD example run end-to-end in the appropriate shells.
- [ ] `nix-build.yml` CI workflow added; passes on the PR.
- [ ] `nix/README.md` documents the architecture and how to add a backend or GPU arch.
- [ ] No existing CI workflow regresses.
- [ ] Old `.packaging/` Dockerfiles are **not** removed in this PR (phase 5).
- [ ] PR description includes evidence of manual GPU + NEK runs per the verification section.
