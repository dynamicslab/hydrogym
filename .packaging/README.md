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
