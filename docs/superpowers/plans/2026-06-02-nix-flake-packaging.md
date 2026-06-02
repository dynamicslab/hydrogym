# Deterministic Nix-Flake Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Firedrake-only `.packaging/` Dockerfiles with a minimal Ubuntu 22.04 + Nix base image plus per-backend × per-GPU-arch Nix flake outputs for JAX, JAX-Fluids, and NEK. Firedrake and MAIA are out of scope for this PR.

**Architecture:** One base Docker image (Ubuntu 22.04 + multi-user Nix + HydroGym checkout, no Python/CUDA). One root-level `flake.nix` whose `flake.lock` is the reproducibility anchor. Per-backend dev shells via an `mkBackendShell` factory. Builds two GPU arch families (`hopper-blackwell`, `turing-ampere`) for JAX and JAX-Fluids; CPU-only case-parameterised shells for NEK.

**Tech Stack:** Nix flakes, Ubuntu 22.04 base, Docker, NVHPC SDK 26.1 (CUDA 12.9), Python 3.12, JAX, jaxlib (CUDA 12 local), NEK5000 (Fortran/MPI), MPICH, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-02-nix-flake-packaging-design.md`

---

## File Structure

**New files (created in this plan):**

```
.packaging/Dockerfile.base                       # Ubuntu 22.04 + Nix + HydroGym
.packaging/README.md                             # How to build/tag/push the base image
flake.nix                                        # Top-level flake; one source of truth
flake.lock                                       # Generated; commit it
nix/README.md                                    # Architecture; how to add a backend
nix/lib/cudaTargets.nix                          # Arch flag tables (hopper-blackwell, turing-ampere)
nix/lib/mkBackendShell.nix                       # Factory for per-backend dev shells
nix/overlays/nvhpc-26.1.nix                      # NVHPC SDK 26.1 FOD derivation
nix/overlays/jaxlib-cuda.nix                     # jaxlib bound to NVHPC's CUDA libs
nix/overlays/jaxfluids-rl.nix                    # buildPythonPackage for jaxfluids_rl
nix/backends/jax/default.nix                     # JAX deps + shell hook
nix/backends/jaxfluids/default.nix               # JAX-Fluids deps + shell hook
nix/backends/nek/default.nix                     # NEK shell factory: mkNekShell { case = ...; }
nix/backends/nek/nek5000.nix                     # Builds nek5000 binary from submodule
nix/backends/nek/cases/MiniChannel/SIZE          # Case-specific SIZE file
nix/backends/nek/cases/MiniChannel/MiniChannel.usr
nix/backends/nek/cases/MiniChannel/MiniChannel.par
.github/workflows/nix-build.yml                  # New CI job for `nix flake check`
.gitmodules                                      # Modified to add jaxfluids_rl + nek5000
third_party/jaxfluids_rl/                        # New submodule (rev-pinned)
third_party/nek5000/                             # New submodule (rev-pinned)
```

**Files NOT touched in this PR** (per spec, removal is phase 5):
- `.packaging/Dockerfile.firedrake_env`
- `.packaging/Dockerfile.hydrogym_env`
- `.packaging/Dockerfile.hydrogym`
- `.packaging/Dockerfile.devpod`
- `examples/*/run_*_docker.sh` (HPC module scripts — out of scope)

---

## Phase 1 — Base image + flake skeleton + JAX outputs

### Task 1: Repository housekeeping for new directories

**Files:**
- Create: `nix/.gitkeep`
- Create: `nix/lib/.gitkeep`
- Create: `nix/overlays/.gitkeep`
- Create: `nix/backends/.gitkeep`

- [ ] **Step 1: Create the empty directory structure**

```bash
mkdir -p nix/lib nix/overlays nix/backends/jax nix/backends/jaxfluids nix/backends/nek/cases
touch nix/.gitkeep nix/lib/.gitkeep nix/overlays/.gitkeep nix/backends/.gitkeep
```

- [ ] **Step 2: Verify**

```bash
find nix -type d
```

Expected output:
```
nix
nix/lib
nix/overlays
nix/backends
nix/backends/jax
nix/backends/jaxfluids
nix/backends/nek
nix/backends/nek/cases
```

- [ ] **Step 3: Commit**

```bash
git add nix/
git commit -m "chore: scaffold nix/ directory layout"
```

---

### Task 2: Pin Nix installer version in base Dockerfile

**Files:**
- Create: `.packaging/Dockerfile.base`

- [ ] **Step 1: Determine the pinned Nix installer release**

Open https://github.com/NixOS/nix/releases in a browser and pick the latest stable release tagged `2.x` that has a `nix-2.x.x-x86_64-linux.tar.xz` artifact. Record the version and the SHA256 of that tarball.

Worked example values (replace with the values you record):
- Version: `2.24.10`
- Tarball URL: `https://releases.nixos.org/nix/nix-2.24.10/nix-2.24.10-x86_64-linux.tar.xz`
- SHA256 (compute with `curl -L <url> | sha256sum`): `0000000000000000000000000000000000000000000000000000000000000000`

- [ ] **Step 2: Write `.packaging/Dockerfile.base`**

Replace `__NIX_VERSION__` and `__NIX_SHA256__` with the values from Step 1.

```dockerfile
# syntax=docker/dockerfile:1.7

# Base image for HydroGym: Ubuntu 22.04 with multi-user Nix installed.
# The image is NOT the reproducibility anchor — flake.lock is. This image
# can be rebuilt freely; only the pinned Nix installer version matters.

FROM ubuntu:22.04

ARG NIX_VERSION=__NIX_VERSION__
ARG NIX_SHA256=__NIX_SHA256__

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Minimal apt set: enough to bootstrap Nix and run typical dev tooling.
# Anything solver-specific (Python, CUDA, MPI) belongs in the flake, not here.
RUN apt-get update \
 && apt-get -y install --no-install-recommends \
        ca-certificates curl xz-utils sudo locales \
        git openssh-client vim tini \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# Create unprivileged user `hydrogym`. Multi-user Nix daemon runs as root.
RUN useradd -m -s /bin/bash -G sudo hydrogym \
 && echo "hydrogym ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install multi-user Nix from a pinned tarball (offline-friendly, hash-verified).
RUN curl -fsSL -o /tmp/nix.tar.xz \
        "https://releases.nixos.org/nix/nix-${NIX_VERSION}/nix-${NIX_VERSION}-x86_64-linux.tar.xz" \
 && echo "${NIX_SHA256}  /tmp/nix.tar.xz" | sha256sum -c - \
 && mkdir -p /tmp/nix-install \
 && tar -xJf /tmp/nix.tar.xz -C /tmp/nix-install --strip-components=1 \
 && /tmp/nix-install/install --daemon --yes --no-channel-add \
 && rm -rf /tmp/nix.tar.xz /tmp/nix-install

# Enable flakes and the new `nix` CLI by default.
RUN mkdir -p /etc/nix \
 && printf 'experimental-features = nix-command flakes\nbuild-users-group = nixbld\n' \
      > /etc/nix/nix.conf

# HydroGym source is baked at /opt/hydrogym; users override at run time with
# `-v $PWD:/opt/hydrogym`. The clone targets the dynamicslab repository at
# build time; rebuilds pick up new commits.
RUN git clone https://github.com/dynamicslab/hydrogym.git /opt/hydrogym \
 && chown -R hydrogym:hydrogym /opt/hydrogym

# HF cache mount point. Users mount their host cache here for offline use.
RUN install -d -o hydrogym -g hydrogym /home/hydrogym/.cache/hydrogym

USER hydrogym
WORKDIR /opt/hydrogym

# Source the Nix profile so `nix` is on PATH in interactive shells.
RUN echo '. /etc/profile.d/nix.sh' >> /home/hydrogym/.bashrc

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
```

- [ ] **Step 3: Validate the Dockerfile syntax**

```bash
docker buildx build --check -f .packaging/Dockerfile.base .
```

Expected: `Check complete, no warnings found.` (or no errors; warnings are acceptable).

- [ ] **Step 4: Build the image to confirm it actually builds (no GPU needed)**

```bash
docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .
```

Expected: image builds; final line `Successfully tagged hydrogym/base:dev` or buildkit equivalent.

- [ ] **Step 5: Smoke-test the image — Nix is on PATH and works**

```bash
docker run --rm hydrogym/base:dev bash -lc 'nix --version && nix store ping'
```

Expected:
```
nix (Nix) 2.x.x
Store URL: daemon
Trusted: 0
Version: 2.x.x
```

- [ ] **Step 6: Commit**

```bash
git add .packaging/Dockerfile.base
git commit -m "feat(packaging): add Ubuntu 22.04 + Nix base image"
```

---

### Task 3: Write `.packaging/README.md`

**Files:**
- Create: `.packaging/README.md`

- [ ] **Step 1: Write the README**

```markdown
# HydroGym Packaging

This directory holds the Docker base image used by the Nix-flake-based
packaging system. The base image is intentionally minimal:

- Ubuntu 22.04
- Multi-user Nix daemon (version pinned in `Dockerfile.base`)
- HydroGym source cloned at `/opt/hydrogym`
- HF cache mount point at `/home/hydrogym/.cache/hydrogym`

No Python, no CUDA, no solver libraries live in this image. All of that
is provided by the Nix flake at the repo root — `flake.lock` is the
reproducibility anchor.

## Build

```bash
docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .
```

## Run (GPU)

```bash
docker run --gpus all -it \
    -v $PWD:/opt/hydrogym \
    -v ~/.cache/hydrogym:/home/hydrogym/.cache/hydrogym \
    hydrogym/base:dev
```

Inside the container:

```bash
cd /opt/hydrogym
nix develop .#jax-cuda-hopper-blackwell    # or another flake output
```

## GPU driver requirement

The flake's GPU outputs target CUDA 12.9 via NVHPC SDK 26.1. Your host
NVIDIA driver must be ABI-compatible with CUDA 12.9 (≈ driver 555 or
newer; verify with `nvidia-smi` and the
[CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/)).

## Legacy Firedrake images

The four `Dockerfile.{firedrake_env,hydrogym_env,hydrogym,devpod}` files
in this directory are the previous Firedrake-only packaging. They stay
in place until Firedrake gains a native Nix build (phase 4 of the
migration, follow-up PR).
```

- [ ] **Step 2: Commit**

```bash
git add .packaging/README.md
git commit -m "docs(packaging): document base image build and run"
```

---

### Task 4: Create `nix/lib/cudaTargets.nix`

**Files:**
- Create: `nix/lib/cudaTargets.nix`

- [ ] **Step 1: Write the cudaTargets file**

The exact PTX/SASS list per family is finalised during Step 3 of this task by reading the NVHPC SDK 26.1 release notes for `-gpu=` flags. Start with the values below; adjust if the release notes contradict them.

```nix
# nix/lib/cudaTargets.nix
#
# GPU architecture target tables consumed by mkBackendShell.
# Each target produces a separate dev-shell output; the closures differ
# in the AOT-compiled CUDA kernels they ship. The split mirrors the
# existing clagemann/hydrogym-nvhpc-26.1_cuda-12.9_{hopper_blackwell,
# turing_ampere}:latest image tags.
{
  hopper-blackwell = {
    name = "hopper-blackwell";
    # nvcc --gpu-architecture flags (compute capability list, comma-separated)
    nvccGencode = "compute_90,compute_100";
    # NVHPC's -gpu= compiler flag
    nvhpcGpuArch = "cc90,cc100";
    # JAX's XLA_FLAGS target list (deviceless AOT compile)
    xlaCudaTargets = "sm_90,sm_100";
  };

  turing-ampere = {
    name = "turing-ampere";
    nvccGencode = "compute_75,compute_80,compute_86";
    nvhpcGpuArch = "cc75,cc80,cc86";
    xlaCudaTargets = "sm_75,sm_80,sm_86";
  };
}
```

- [ ] **Step 2: Lint Nix syntax**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix-instantiate --parse nix/lib/cudaTargets.nix'
```

Expected: prints an AST, no errors.

- [ ] **Step 3: Verify the arch flags against NVHPC SDK 26.1 docs**

Open `https://docs.nvidia.com/hpc-sdk/archive/26.1/hpc-sdk-release-notes/index.html`. Confirm:
- `cc90` (Hopper) and `cc100` (Blackwell) are both supported by `-gpu=`.
- `cc75` / `cc80` / `cc86` are still supported (they were in earlier releases).

If `cc100` is not in the release, drop it and document the limitation in `nix/README.md` (Task 18).

- [ ] **Step 4: Commit**

```bash
git add nix/lib/cudaTargets.nix
git commit -m "feat(nix): add CUDA target tables for two arch families"
```

---

### Task 5: Create `nix/overlays/nvhpc-26.1.nix`

**Files:**
- Create: `nix/overlays/nvhpc-26.1.nix`

- [ ] **Step 1: Determine NVHPC SDK 26.1 URL and SHA256**

The plan assumes NVIDIA serves the SDK without auth. Verify by running:

```bash
curl -fsI "https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_12.9.tar.gz" | head -5
```

If the response is `HTTP/2 200` (or 301 → 200), the URL is public. Record the canonical URL and compute the SHA256:

```bash
curl -fsSL -o /tmp/nvhpc.tar.gz <URL>
sha256sum /tmp/nvhpc.tar.gz
```

If the URL is **not** public (auth required, 404, etc.), stop. The implementation must switch to `requireFile` — see the fallback at the bottom of this task.

Worked example (replace with real values):
- URL: `https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_12.9.tar.gz`
- SHA256: `0000000000000000000000000000000000000000000000000000000000000000`

- [ ] **Step 2: Write the overlay**

```nix
# nix/overlays/nvhpc-26.1.nix
#
# NVHPC SDK 26.1 (containing CUDA 12.9, NCCL, math libs, nvcc, nvfortran,
# nvc/nvc++) as a fixed-output derivation. This is the largest input to
# any GPU flake output — multi-GB in the store. Build it once per host
# and re-use via the Nix binary cache.

{ stdenv, fetchurl, lib, autoPatchelfHook, zlib, glibc, libGL }:

stdenv.mkDerivation rec {
  pname = "nvhpc-sdk";
  version = "26.1";

  src = fetchurl {
    url = "https://developer.download.nvidia.com/hpc-sdk/${version}/nvhpc_2026_261_Linux_x86_64_cuda_12.9.tar.gz";
    sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  # Runtime libs the patched ELFs need to resolve at load time.
  buildInputs = [ zlib glibc libGL ];

  # The tarball contains an install script; we extract its payload directly
  # to avoid running the installer (which prompts and writes outside store).
  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    # Vendor layout: install/{Linux_x86_64,...}. Move the whole tree.
    cp -r install/* "$out/"

    # Symlink the canonical binaries to $out/bin for PATH consumption.
    mkdir -p "$out/bin"
    for tool in nvcc nvc nvc++ nvfortran nvprof nsys ncu; do
      if [ -e "$out/Linux_x86_64/${version}/compilers/bin/$tool" ]; then
        ln -s "../Linux_x86_64/${version}/compilers/bin/$tool" "$out/bin/$tool"
      fi
    done

    runHook postInstall
  '';

  meta = with lib; {
    description = "NVIDIA HPC SDK 26.1 (CUDA 12.9)";
    homepage = "https://developer.nvidia.com/hpc-sdk";
    license = licenses.unfree;
    platforms = [ "x86_64-linux" ];
    # Vendor blob: distribution is governed by NVIDIA's EULA.
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
```

**Fallback (if SDK URL is not public):** replace the `src = fetchurl { ... }` block with:

```nix
src = requireFile {
  name = "nvhpc_2026_261_Linux_x86_64_cuda_12.9.tar.gz";
  sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
  message = ''
    Download the NVHPC SDK 26.1 (CUDA 12.9) tarball from
    https://developer.nvidia.com/nvidia-hpc-sdk-261-downloads
    and add it to the Nix store:
      nix-store --add-fixed sha256 ./nvhpc_2026_261_Linux_x86_64_cuda_12.9.tar.gz
  '';
};
```

- [ ] **Step 3: Confirm the tarball internal layout**

Run inside the base image:

```bash
docker run --rm -v /tmp:/host hydrogym/base:dev bash -lc \
  'tar -tzf /host/nvhpc.tar.gz | head -20'
```

Verify the listed paths match the `install/Linux_x86_64/${version}/compilers/bin/...` assumption in the `installPhase`. If they differ, adjust the `cp -r install/*` and binary-symlink loop accordingly.

- [ ] **Step 4: Commit**

```bash
git add nix/overlays/nvhpc-26.1.nix
git commit -m "feat(nix): add NVHPC SDK 26.1 overlay"
```

---

### Task 6: Create `nix/lib/mkBackendShell.nix`

**Files:**
- Create: `nix/lib/mkBackendShell.nix`

- [ ] **Step 1: Write the factory**

```nix
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
```

- [ ] **Step 2: Lint Nix syntax**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix-instantiate --parse nix/lib/mkBackendShell.nix'
```

Expected: prints an AST, no errors.

- [ ] **Step 3: Commit**

```bash
git add nix/lib/mkBackendShell.nix
git commit -m "feat(nix): add mkBackendShell factory"
```

---

### Task 7: Create `nix/backends/jax/default.nix`

**Files:**
- Create: `nix/backends/jax/default.nix`

- [ ] **Step 1: Write the JAX backend module**

The Python dep list comes directly from `pyproject.toml`'s `[tool.poetry.extras].jax`: `jax, jaxlib, chex, navix, gymnax, tree-math, flax, omegaconf, toml`. The factory takes a cudaTarget so two outputs share one definition.

```nix
# nix/backends/jax/default.nix
#
# JAX backend Nix dev shell.
#
# Python dep list mirrors pyproject.toml [tool.poetry.extras].jax:
#   jax, jaxlib, chex, navix, gymnax, tree-math, flax, omegaconf, toml
# Plus numpy/scipy/pandas which are core HydroGym deps.
#
# jaxlib comes from the official jax[cuda12-local] PyPI wheel installed by
# pip on shell entry — nixpkgs jaxlib is not built against CUDA 12.9 at
# time of writing. The wheel discovers system CUDA libs at runtime via
# LD_LIBRARY_PATH set by mkBackendShell.

{ pkgs, lib, mkBackendShell, cudaTarget }:

mkBackendShell {
  name = "jax-cuda-${cudaTarget.name}";
  python = pkgs.python312;
  inherit cudaTarget;

  pythonDeps = pyPkgs: with pyPkgs; [
    # Core HydroGym runtime
    numpy scipy pandas gymnasium
    huggingface-hub control dmsuite
    # JAX extra
    chex flax
    # Config + serialization
    omegaconf toml
    # Pip itself, so jax[cuda12-local] can install on shell entry
    pip setuptools wheel
    # SB3 + friends are common in JAX RL workflows but not required
    # by the JAX extras list; users add them with `pip install`.
  ];

  extraInputs = [
    pkgs.git
    pkgs.nvhpc-sdk    # exposed via overlay
  ];

  extraShellHook = ''
    # jax + jaxlib (CUDA 12 local) installed into the user's home so the
    # editable HydroGym install can find them. Nix-provided python has pip.
    if ! python -c "import jax" 2>/dev/null; then
      echo "Installing jax[cuda12-local] from PyPI..."
      pip install --user --quiet \
        "jax[cuda12-local]==0.4.34" \
        "navix" \
        "gymnax" \
        "tree-math"
    fi
    export PATH="$HOME/.local/bin:$PATH"
  '';
}
```

The `jax==0.4.34` pin is the latest at design time that's compatible with CUDA 12.9. Update during implementation if a newer release exists when the work happens; record the chosen version in `nix/README.md`.

- [ ] **Step 2: Lint Nix syntax**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix-instantiate --parse nix/backends/jax/default.nix'
```

Expected: prints an AST, no errors.

- [ ] **Step 3: Commit**

```bash
git add nix/backends/jax/default.nix
git commit -m "feat(nix): add JAX backend dev shell module"
```

---

### Task 8: Create the root `flake.nix` with JAX outputs

**Files:**
- Create: `flake.nix`

- [ ] **Step 1: Write the flake**

```nix
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
```

- [ ] **Step 2: Generate `flake.lock` (inside the base container)**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake lock'
```

Expected: `nix/flake.lock` is created, listing `nixpkgs` and `flake-utils` pinned to specific revs.

- [ ] **Step 3: Verify the flake evaluates**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake show --no-write-lock-file'
```

Expected output (substring):
```
└───devShells
    └───x86_64-linux
        ├───default: development environment 'jax-cuda-hopper-blackwell'
        ├───jax-cuda-hopper-blackwell: development environment 'jax-cuda-hopper-blackwell'
        └───jax-cuda-turing-ampere: development environment 'jax-cuda-turing-ampere'
```

- [ ] **Step 4: Run `nix flake check`**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake check --no-build'
```

Expected: exits 0, no errors. (`--no-build` keeps it fast — we don't pull the NVHPC SDK here; Task 9 does the actual build.)

- [ ] **Step 5: Commit**

```bash
git add flake.nix flake.lock
git commit -m "feat(nix): add root flake with JAX devshell outputs"
```

---

### Task 9: Build and smoke-test the JAX Hopper/Blackwell devshell

**Files:**
- (no file changes — validation task)

- [ ] **Step 1: Build the JAX Hopper/Blackwell devshell**

This pulls the NVHPC SDK tarball (multi-GB) and assembles the closure. First build is slow (no binary cache yet). Inside the container:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix build .#jax-cuda-hopper-blackwell.inputDerivation --no-link'
```

Expected: build completes, no errors. (We build `.inputDerivation` because `mkShell` derivations have no `out` payload.)

- [ ] **Step 2: Enter the devshell and check `jax.devices()`**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix develop .#jax-cuda-hopper-blackwell -c python -c "import jax; print(jax.devices())"'
```

Expected: prints a list including a `CudaDevice(id=0, ...)` entry. If only CPU devices are listed, the LD_LIBRARY_PATH / XLA setup is wrong — debug before continuing.

- [ ] **Step 3: Run the Kolmogorov example end-to-end**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc '
    cd /opt/hydrogym
    nix develop .#jax-cuda-hopper-blackwell -c \
      python examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py minimize_tke --num-steps 5
  '
```

Expected: step table prints; no NaN/Inf; exit code 0.

- [ ] **Step 4: Record the run output in the PR description**

Save the terminal output. The PR description must include this evidence per the spec's verification gate.

- [ ] **Step 5: (No commit — validation task.)**

---

### Task 10: Build and smoke-test the JAX Turing/Ampere devshell

**Files:**
- (no file changes — validation task)

- [ ] **Step 1: On a host with Turing or Ampere GPU, build the shell**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix build .#jax-cuda-turing-ampere.inputDerivation --no-link'
```

Expected: build completes; closure shares the NVHPC SDK store path with Hopper/Blackwell (most of the build is cached).

- [ ] **Step 2: Verify `jax.devices()` and run Kolmogorov**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc '
    cd /opt/hydrogym
    nix develop .#jax-cuda-turing-ampere -c \
      python -c "import jax; print(jax.devices())"
    nix develop .#jax-cuda-turing-ampere -c \
      python examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py minimize_tke --num-steps 5
  '
```

Expected: CUDA device listed; Kolmogorov example completes.

- [ ] **Step 3: Record output in PR description.**

---

### Task 11: Add `nix-build.yml` GitHub Actions workflow (JAX-only checks for now)

**Files:**
- Create: `.github/workflows/nix-build.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Nix Flake

on:
  push:
  pull_request:

permissions:
  contents: read

concurrency:
  group: nix-build-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  flake-check:
    name: nix flake check (eval only)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: cachix/install-nix-action@v27
        with:
          nix_path: nixpkgs=channel:nixos-24.05
          extra_nix_config: |
            experimental-features = nix-command flakes

      - name: nix flake show
        run: nix flake show --no-write-lock-file

      - name: nix flake check (evaluation only, no GPU builds)
        run: nix flake check --no-build --no-write-lock-file

  # GPU-output builds require a self-hosted GPU runner; not part of this
  # workflow. The Phase 3 task adds a CPU `nek-cpu-MiniChannel` build
  # step here once that output exists.
```

- [ ] **Step 2: Validate workflow syntax**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'python -c "import yaml; yaml.safe_load(open(\"/work/.github/workflows/nix-build.yml\"))"'
```

Expected: exits 0, no parse error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/nix-build.yml
git commit -m "ci: add Nix flake check workflow"
```

- [ ] **Step 4: Push the branch and confirm the workflow runs**

After pushing this commit to the PR branch, watch the GitHub Actions tab. Expected: `Nix Flake → flake-check (eval only)` job runs and passes within ~3 minutes.

---

## Phase 2 — JAX-Fluids outputs

### Task 12: Add `jaxfluids_rl` submodule and overlay

**Files:**
- Modify: `.gitmodules`
- Create: `third_party/jaxfluids_rl/` (as submodule)
- Create: `nix/overlays/jaxfluids-rl.nix`

- [ ] **Step 1: Determine the upstream jaxfluids_rl repo and rev**

The `hydrogym/jaxfluids/env_core.py` imports `from jaxfluids_rl.jxf_env import JAXFluidsEnv`. Find the source repository:

```bash
grep -rn "jaxfluids_rl" /home/lpaehler/Work/ReinforcementLearning/hydrogym-next/examples/jaxfluids/ 2>/dev/null
grep -rn "jaxfluids_rl" /home/lpaehler/Work/ReinforcementLearning/hydrogym-next/hydrogym/jaxfluids/ 2>/dev/null
```

If the repo URL is documented in `examples/jaxfluids/getting_started/README.md`, use that. Otherwise ask the maintainer (Ludger) for the correct git URL. Record:

- URL: `<URL>`
- Pinned commit SHA: `<SHA>`

- [ ] **Step 2: Add the submodule**

```bash
git submodule add <URL> third_party/jaxfluids_rl
cd third_party/jaxfluids_rl
git checkout <SHA>
cd ../..
```

- [ ] **Step 3: Write the overlay**

```nix
# nix/overlays/jaxfluids-rl.nix
#
# jaxfluids_rl is not on PyPI in a Nix-friendly form. We vendor it as
# a git submodule (third_party/jaxfluids_rl) and build it as a Python
# package against the same Python interpreter the rest of the JAX-Fluids
# shell uses.

{ buildPythonPackage, lib }:

buildPythonPackage {
  pname = "jaxfluids-rl";
  # Version comes from the submodule; pin in the source itself.
  version = "0.1.0";
  format = "pyproject";
  src = ../../third_party/jaxfluids_rl;
  doCheck = false;     # Upstream tests assume GPU; out of scope for build-time.

  meta = with lib; {
    description = "RL adapter for JAX-Fluids";
    license = licenses.gpl3Only;     # Verify against upstream LICENSE
    platforms = [ "x86_64-linux" ];
  };
}
```

If the submodule has a `setup.py` instead of `pyproject.toml`, change `format = "pyproject"` to `format = "setuptools"`.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/jaxfluids_rl nix/overlays/jaxfluids-rl.nix
git commit -m "feat(nix): vendor jaxfluids_rl as submodule"
```

---

### Task 13: Create `nix/backends/jaxfluids/default.nix`

**Files:**
- Create: `nix/backends/jaxfluids/default.nix`

- [ ] **Step 1: Write the backend module**

Python dep list from `[tool.poetry.extras].jaxfluids`: `jax, jaxlib, flax, gitpython, h5py, optax`. Note: `pyvista` is marked `optional = true` in pyproject but is NOT in the jaxfluids extras list — we match the extras list exactly, not the optional-deps list.

```nix
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
```

- [ ] **Step 2: Wire it into `flake.nix`**

Edit `flake.nix`. After the `mkJaxShell = ...;` line, add:

```nix
        mkJaxFluidsShell = cudaTarget:
          pkgs.callPackage ./nix/backends/jaxfluids/default.nix {
            inherit mkBackendShell cudaTarget;
          };
```

And in the `devShells = { ... };` block, add two entries:

```nix
          jaxfluids-cuda-hopper-blackwell = mkJaxFluidsShell cudaTargets.hopper-blackwell;
          jaxfluids-cuda-turing-ampere    = mkJaxFluidsShell cudaTargets.turing-ampere;
```

- [ ] **Step 3: Re-lock the flake and re-check**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake lock && nix flake check --no-build'
```

Expected: lock file updates (no diff if submodule paths unchanged); flake check passes.

- [ ] **Step 4: Verify `nix flake show` lists the new outputs**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake show --no-write-lock-file'
```

Expected: both `jaxfluids-cuda-hopper-blackwell` and `jaxfluids-cuda-turing-ampere` listed under `devShells.x86_64-linux`.

- [ ] **Step 5: Commit**

```bash
git add nix/backends/jaxfluids/default.nix flake.nix flake.lock
git commit -m "feat(nix): add JAX-Fluids devshell outputs"
```

---

### Task 14: Build and smoke-test JAX-Fluids devshell (Hopper/Blackwell)

**Files:**
- (no file changes — validation task)

- [ ] **Step 1: Build the closure**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix build .#jaxfluids-cuda-hopper-blackwell.inputDerivation --no-link'
```

Expected: build completes; reuses NVHPC SDK store path from Task 9.

- [ ] **Step 2: Run the JAX-Fluids nozzle example**

Identify the smallest runnable JAX-Fluids example from `examples/jaxfluids/`:

```bash
ls /home/lpaehler/Work/ReinforcementLearning/hydrogym-next/examples/jaxfluids/
```

If `test_jaxfluids_env.py` is the entry point, run:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc '
    cd /opt/hydrogym
    nix develop .#jaxfluids-cuda-hopper-blackwell -c \
      python examples/jaxfluids/test_jaxfluids_env.py --num-steps 3
  '
```

Expected: exit code 0; per-step output prints. If the example takes CLI flags different from `--num-steps`, adjust accordingly.

- [ ] **Step 3: Record output in PR description.**

---

### Task 15: (Optional) Repeat Task 14 on Turing/Ampere if hardware available

**Files:**
- (no file changes — validation task)

- [ ] **Step 1: On a Turing or Ampere host, run:**

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc '
    cd /opt/hydrogym
    nix develop .#jaxfluids-cuda-turing-ampere -c \
      python examples/jaxfluids/test_jaxfluids_env.py --num-steps 3
  '
```

If Turing/Ampere hardware is not accessible, skip this task; the Hopper/Blackwell validation plus the Turing/Ampere JAX validation (Task 10) together are sufficient evidence that the arch-family mechanism works.

---

## Phase 3 — NEK5000 outputs

### Task 16: Add `nek5000` submodule

**Files:**
- Modify: `.gitmodules`
- Create: `third_party/nek5000/` (as submodule)

- [ ] **Step 1: Record the NEK5000 repo and rev**

The upstream is `https://github.com/Nek5000/Nek5000`. Find the rev that matches the HydroGym integration:

```bash
grep -rn "nek5000" /home/lpaehler/Work/ReinforcementLearning/hydrogym-next/hydrogym/nek/ | head -20
grep -rn "Nek5000" /home/lpaehler/Work/ReinforcementLearning/hydrogym-next/examples/nek/ | head -10
```

If no specific rev is documented, use the latest tagged release. Record:
- URL: `https://github.com/Nek5000/Nek5000`
- Pinned commit SHA: `<SHA>`

- [ ] **Step 2: Add the submodule**

```bash
git submodule add https://github.com/Nek5000/Nek5000 third_party/nek5000
cd third_party/nek5000
git checkout <SHA>
cd ../..
```

- [ ] **Step 3: Commit**

```bash
git add .gitmodules third_party/nek5000
git commit -m "feat(nix): vendor Nek5000 as submodule"
```

---

### Task 17: Create the NEK5000 builder and MiniChannel case

**Files:**
- Create: `nix/backends/nek/nek5000.nix`
- Create: `nix/backends/nek/default.nix`
- Create: `nix/backends/nek/cases/MiniChannel/SIZE`
- Create: `nix/backends/nek/cases/MiniChannel/MiniChannel.usr`
- Create: `nix/backends/nek/cases/MiniChannel/MiniChannel.par`

- [ ] **Step 1: Find the existing MiniChannel case files**

The Easybuild module `Nek5000/1.0-gompi-2024a-SystemCUDA-MiniChannel` references a `MiniChannel` case. Locate the SIZE/.usr/.par files on the user's HPC (or in HuggingFace `dynamicslab/HydroGym-environments`):

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'pip install --user huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id=\"dynamicslab/HydroGym-environments\", \
        repo_type=\"dataset\", \
        allow_patterns=[\"TCFmini*MiniChannel*/**\", \"**/SIZE\", \"**/*.usr\", \"**/*.par\"], \
        local_dir=\"/work/.nek-case-tmp\")"'
```

Then copy the case files in:

```bash
mkdir -p nix/backends/nek/cases/MiniChannel
cp .nek-case-tmp/<path>/SIZE nix/backends/nek/cases/MiniChannel/
cp .nek-case-tmp/<path>/MiniChannel.usr nix/backends/nek/cases/MiniChannel/
cp .nek-case-tmp/<path>/MiniChannel.par nix/backends/nek/cases/MiniChannel/
rm -rf .nek-case-tmp
```

If the files are not on HuggingFace, ask the maintainer for them.

- [ ] **Step 2: Write `nix/backends/nek/nek5000.nix`**

```nix
# nix/backends/nek/nek5000.nix
#
# Builds a case-specific nek5000 binary. NEK5000's build is case-aware:
# the SIZE file determines static array sizes, so each case produces its
# own binary. We isolate that under one derivation per case.
#
# The Python side (mpi4py) and the nek5000 binary MUST be built against
# the same MPI implementation; the backend's default.nix wires both
# against pkgs.mpich.

{ stdenv, lib, gfortran, mpich, makefile-genmap ? null }:

{ case, sizeFile, usrFile, parFile, nek5000Src }:

stdenv.mkDerivation {
  pname = "nek5000-${case}";
  version = "git";

  src = nek5000Src;

  nativeBuildInputs = [ gfortran ];
  buildInputs = [ mpich ];

  # Stage the case-specific files into a build directory and invoke makenek.
  buildPhase = ''
    runHook preBuild
    mkdir -p build/${case}
    cp -r $src/. build/
    cp ${sizeFile} build/${case}/SIZE
    cp ${usrFile}  build/${case}/${case}.usr
    cp ${parFile}  build/${case}/${case}.par

    pushd build/${case}
    export SOURCE_ROOT="$PWD/../../core"
    # makenek picks up MPI compilers from PATH.
    ${nek5000Src}/bin/makenek ${case} || ./makenek ${case}
    popd
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/share/nek/${case}
    cp build/${case}/nek5000 $out/bin/nek5000
    cp build/${case}/${case}.* $out/share/nek/${case}/
    cp build/${case}/SIZE     $out/share/nek/${case}/
    runHook postInstall
  '';

  meta = with lib; {
    description = "Nek5000 spectral element solver (case: ${case})";
    homepage = "https://nek5000.mcs.anl.gov/";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}
```

- [ ] **Step 3: Write `nix/backends/nek/default.nix`**

```nix
# nix/backends/nek/default.nix
#
# Factory for case-parameterised NEK5000 dev shells.
#
# Each case produces:
#   .#nek-cpu-<case>
# with both `mpirun`, `python` (mpi4py-enabled), and the case-specific
# `nek5000` binary on PATH, all linked against pkgs.mpich.

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
  cudaTarget = null;       # CPU-only

  pythonDeps = pyPkgs: with pyPkgs; [
    # Core HydroGym
    numpy scipy pandas gymnasium huggingface-hub control dmsuite
    # nek extras from pyproject.toml [tool.poetry.extras].nek
    mpi4py omegaconf pymech pettingzoo stable-baselines3 supersuit tensorboard
    pip setuptools wheel
  ];

  extraInputs = [
    pkgs.git
    pkgs.mpich
    nek5000Bin
  ];

  extraShellHook = ''
    export NEK_CASE=${case}
    export NEK_SHARE=${nek5000Bin}/share/nek/${case}
    echo "NEK5000 case '${case}' ready. Binary: $(which nek5000)"
  '';
}
```

- [ ] **Step 4: Wire NEK into `flake.nix`**

After the existing `mkJaxFluidsShell` block in `flake.nix`, add:

```nix
        mkNekShell = { case, sizeFile, usrFile, parFile }:
          (pkgs.callPackage ./nix/backends/nek/default.nix {
            inherit mkBackendShell;
          }) {
            inherit case sizeFile usrFile parFile;
            nek5000Src = ./third_party/nek5000;
          };
```

In the `devShells = { ... };` block, add:

```nix
          nek-cpu-MiniChannel = mkNekShell {
            case = "MiniChannel";
            sizeFile = ./nix/backends/nek/cases/MiniChannel/SIZE;
            usrFile  = ./nix/backends/nek/cases/MiniChannel/MiniChannel.usr;
            parFile  = ./nix/backends/nek/cases/MiniChannel/MiniChannel.par;
          };
```

- [ ] **Step 5: Re-lock and re-check**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake lock && nix flake check --no-build && nix flake show --no-write-lock-file'
```

Expected: `nek-cpu-MiniChannel` appears under `devShells.x86_64-linux`.

- [ ] **Step 6: Build it (CPU-only, no GPU required)**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix build .#nek-cpu-MiniChannel.inputDerivation --no-link'
```

Expected: build completes; `nek5000` binary appears in the closure.

- [ ] **Step 7: MPMD smoke test inside the shell**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc '
    cd /opt/hydrogym
    nix develop .#nek-cpu-MiniChannel -c bash -c "
      which nek5000
      python -c \"from mpi4py import MPI; print(MPI.Get_library_version())\"
      mpirun -np 1 python examples/nek/getting_started/1_nekenv_single/test_nek_direct.py --env MiniChannel --steps 5 --nproc 1 : -np 1 nek5000
    "
  '
```

Expected: `which nek5000` resolves to a `/nix/store/...` path; mpi4py prints an MPICH version string matching `pkgs.mpich`; the MPMD smoke test runs `5` steps and exits 0.

If the example invocation differs from this pattern (different CLI flags), inspect `examples/nek/getting_started/1_nekenv_single/run_nekenv_docker.sh` and adapt the command.

- [ ] **Step 8: Extend the CI workflow to build NEK on `ubuntu-latest`**

Edit `.github/workflows/nix-build.yml`. After the `nix flake check` step, add:

```yaml
      - name: nix build nek-cpu-MiniChannel
        run: nix build .#nek-cpu-MiniChannel.inputDerivation --no-link
```

- [ ] **Step 9: Commit**

```bash
git add nix/backends/nek/ flake.nix flake.lock .github/workflows/nix-build.yml
git commit -m "feat(nix): add NEK5000 MiniChannel devshell + CI build"
```

---

### Task 18: Write `nix/README.md`

**Files:**
- Create: `nix/README.md`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add nix/README.md
git commit -m "docs(nix): document Nix packaging layout and extension points"
```

---

### Task 19: Update top-level `README.md` (additive only)

**Files:**
- Modify: `README.md` (top-level)

- [ ] **Step 1: Read the current README's Quick Start section**

```bash
sed -n '28,50p' README.md
```

Confirm the `## Quick Start with Docker (Recommended)` heading and the `docker pull clagemann/...` block exist.

- [ ] **Step 2: Insert a new section AFTER the existing Docker quick start**

The existing `clagemann/...` instructions must stay in place. We add a sibling section below them. Open `README.md` and find the line:

```
## Available Environments
```

Insert this section immediately before that line:

```markdown
## Deterministic builds with Nix (new)

For bit-reproducible environments — pinned via a single `flake.lock` —
HydroGym ships a minimal base Docker image plus per-backend Nix dev
shells. Available outputs:

- `.#jax-cuda-hopper-blackwell` / `.#jax-cuda-turing-ampere`
- `.#jaxfluids-cuda-hopper-blackwell` / `.#jaxfluids-cuda-turing-ampere`
- `.#nek-cpu-MiniChannel`

Firedrake and MAIA are not yet migrated — keep using the
`lpaehler/hydrogym-*:stable` and `clagemann/hydrogym-*` images above.

```bash
# Build the base image (one-time)
docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .

# Run with GPU pass-through
docker run --gpus all -it -v $PWD:/opt/hydrogym hydrogym/base:dev

# Inside the container — enter the dev shell for your backend × GPU arch
cd /opt/hydrogym
nix develop .#jax-cuda-hopper-blackwell
python examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py
```

See [`nix/README.md`](nix/README.md) for the architecture and
[`.packaging/README.md`](.packaging/README.md) for image build details.

```

- [ ] **Step 3: Verify the new section renders correctly**

```bash
head -120 README.md | tail -40
```

Confirm the new section appears between the `docker pull` block and `## Available Environments`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add Nix-based packaging quick start to README"
```

---

## Phase 4 — Final verification and PR preparation

### Task 20: Run the full acceptance checklist

**Files:**
- (no file changes — verification task)

Run each acceptance criterion from the spec end-to-end. Record the outcome in the PR description.

- [ ] **Step 1: `Dockerfile.base` builds**

```bash
docker build -f .packaging/Dockerfile.base -t hydrogym/base:dev .
```

Expected: exit 0.

- [ ] **Step 2: `nix flake show` lists all 5 outputs**

```bash
docker run --rm -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'nix flake show --no-write-lock-file'
```

Expected: lists `jax-cuda-hopper-blackwell`, `jax-cuda-turing-ampere`, `jaxfluids-cuda-hopper-blackwell`, `jaxfluids-cuda-turing-ampere`, `nek-cpu-MiniChannel`, plus `default`.

- [ ] **Step 3: `flake.lock` exists and pins inputs**

```bash
test -f flake.lock && jq '.nodes | keys' flake.lock
```

Expected: lists `["flake-utils", "nixpkgs", "root", ...]`.

- [ ] **Step 4: JAX dev shell smoke test (GPU)**

(Already validated in Task 9; re-run if any flake.nix changes happened in later tasks.)

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work hydrogym/base:dev \
  bash -lc 'cd /opt/hydrogym && nix develop .#jax-cuda-hopper-blackwell -c python -c "import hydrogym.jax, jax; print(jax.devices())"'
```

Expected: a CUDA device appears in the printed list.

- [ ] **Step 5: JAX-Fluids smoke test (GPU)**

(Task 14 covered this; re-run.)

- [ ] **Step 6: NEK MPMD smoke test (CPU)**

(Task 17 step 7 covered this; re-run.)

- [ ] **Step 7: CI `nix-build.yml` workflow passes on the PR branch**

Push the branch. Watch the Actions tab. Expected: `Nix Flake` workflow passes within ~5 minutes (`flake check` + `nek build`).

- [ ] **Step 8: Old `.packaging/` files are untouched**

```bash
ls .packaging/Dockerfile.firedrake_env .packaging/Dockerfile.hydrogym_env .packaging/Dockerfile.hydrogym .packaging/Dockerfile.devpod
```

Expected: all four files listed; no errors.

- [ ] **Step 9: No existing CI workflow regresses**

In the GitHub Actions tab, confirm all of `ruff_lint`, `ruff_format`, `isort`, `codespell`, `build`, `Spelling (codespell)` still pass on the PR.

- [ ] **Step 10: Compile evidence into PR description**

The PR description must include:
- Output of Task 9 step 2 (CUDA device list, Kolmogorov run).
- Output of Task 10 step 2 (Turing/Ampere CUDA device list, Kolmogorov run).
- Output of Task 14 step 2 (JAX-Fluids run).
- Output of Task 17 step 7 (NEK MPMD run).
- Hardware used for each run (GPU model + driver version from `nvidia-smi`).

---

### Task 21: Final commit and PR

**Files:**
- (no file changes — administrative task)

- [ ] **Step 1: Confirm git tree is clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: Confirm submodules are clean and on pinned revs**

```bash
git submodule status
```

Expected: each submodule prefixed with a space (not `+` or `-`).

- [ ] **Step 3: Push the branch**

```bash
git push -u origin <branch-name>
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "Deterministic Nix-flake packaging for JAX, JAX-Fluids, and NEK" --body "$(cat <<'EOF'
## Summary

Replaces ad-hoc Docker packaging with one minimal Ubuntu 22.04 + Nix
base image plus per-backend Nix flake outputs:
- `.#jax-cuda-hopper-blackwell` / `.#jax-cuda-turing-ampere`
- `.#jaxfluids-cuda-hopper-blackwell` / `.#jaxfluids-cuda-turing-ampere`
- `.#nek-cpu-MiniChannel`

`flake.lock` is the reproducibility anchor. Firedrake and MAIA are out
of scope for this PR — the existing Firedrake Dockerfiles and Docker
Hub images keep working unchanged.

## Design

See `docs/superpowers/specs/2026-06-02-nix-flake-packaging-design.md`.

## Verification

[paste outputs collected in Task 20 Step 10 here]

## Test plan

- [x] `Dockerfile.base` builds on a clean Ubuntu host.
- [x] `nix flake show` lists all 5 outputs.
- [x] `nix flake check --no-build` passes in CI.
- [x] JAX Kolmogorov example runs end-to-end on Hopper/Blackwell GPU.
- [x] JAX Kolmogorov example runs end-to-end on Turing/Ampere GPU.
- [x] JAX-Fluids example runs end-to-end on at least one GPU.
- [x] NEK MPMD smoke test passes (CPU).
- [x] Existing CI workflows (ruff, isort, codespell, build) unchanged.

## What's NOT in this PR

- Firedrake (phase 4 follow-up PR).
- MAIA (deferred — no portable binary).
- Removing old `.packaging/Dockerfile.{firedrake_env,hydrogym_env,hydrogym,devpod}` (phase 5).
- Replacing `examples/*/run_*_docker.sh` HPC module scripts (separate PR).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.

---

## Plan Self-Review

### Spec coverage check

- §Motivation — addressed in Task 19 (README) and the PR body of Task 21.
- §Goals (4 goals) — bit-identical environments: Task 8 step 4 (`nix flake check`), Task 20. One source of truth: addressed by removing the 4-Dockerfile chain in phase 5 (out of scope here, called out explicitly). Match clagemann tags: Task 4 (cudaTargets) and Task 18 (README). Co-located edits: Task 6 (shellHook with `pip install -e --no-deps`) and Task 19 README example using `-v $PWD:/opt/hydrogym`.
- §Architecture — Tasks 2, 6, 7, 13, 17 implement the three-layer model.
- §Per-backend output matrix — Tasks 7 (JAX), 13 (JAX-Fluids), 17 (NEK). Firedrake/MAIA explicitly deferred (covered in §Migration phasing).
- §Repository layout — Task 1 scaffolds; subsequent tasks populate.
- §Per-backend risk and mitigation:
  - JAX (jaxlib↔CUDA 12.9 ABI): Task 7 uses `jax[cuda12-local]` from PyPI; Task 9 validates via `jax.devices()`.
  - JAX-Fluids (jaxfluids_rl not on PyPI): Task 12 (submodule + overlay).
  - NEK (case-specific compile): Task 17 (`mkNekShell` factory + per-case directory).
- §User workflow — Tasks 19, 20.
- §Testing and CI — Task 11 (initial CI), Task 17 step 8 (NEK CI build), Task 20 (manual verification).
- §Migration phasing — phases 1–3 in this plan; phase 4 acknowledged in Task 18; phase 5 explicitly omitted in Task 20 step 8.
- §Open questions for implementation phase (5 items):
  - 1. NVHPC URL: Task 5 step 1 (verification step required).
  - 2. Min host driver: Task 3 (.packaging/README.md mentions ≈555+; documented).
  - 3. Cachix: deferred (not in this PR). Plan acknowledges slow first builds in Task 9.
  - 4. NEK case list: MiniChannel only this PR (Task 17). Other cases tracked as future work.
  - 5. jaxlib version pin: Task 7 step 1 picks `jax==0.4.34` with instruction to update during impl.
- §Acceptance criteria (10 items) — all checked by Task 20.

No spec gaps identified.

### Placeholder scan

- Task 2 step 1: `__NIX_VERSION__` and `__NIX_SHA256__` are intentional placeholders for values the engineer must record — Task 2 step 1 explicitly tells them to substitute. **This is allowed** because the steps tell the engineer exactly how to find the values (not "fill in details").
- Task 5 step 1: same pattern for the NVHPC SHA256 — explicit verification command provided.
- Task 12 step 1, Task 16 step 1: submodule URL and rev — explicit lookup procedure provided. Acceptable.
- Task 17 step 1: case file paths from HuggingFace — explicit download command provided.
- No "TBD", "TODO", "implement later", "similar to Task N", "add appropriate error handling" anywhere.

### Type/signature consistency

- `mkBackendShell` signature (Task 6): `{ name, python, pythonDeps, extraInputs ? [], cudaTarget ? null, extraShellHook ? "" }`.
  - JAX backend (Task 7) passes all six, including `cudaTarget`. ✓
  - JAX-Fluids backend (Task 13) same. ✓
  - NEK backend (Task 17) passes `cudaTarget = null`. ✓
- `cudaTargets.nix` attribute names: `hopper-blackwell`, `turing-ampere` — used consistently in `flake.nix` (Task 8), Task 13, and `nix/README.md` (Task 18). ✓
- `mkNekShell` parameter list (Task 17 step 4): `{ case, sizeFile, usrFile, parFile, nek5000Src }` — matches the `nek5000.nix` function signature (Task 17 step 2). ✓
- Output naming convention `<backend>-<compute>-<arch>` — held consistently across all tasks and README. ✓

No type/signature issues found.
