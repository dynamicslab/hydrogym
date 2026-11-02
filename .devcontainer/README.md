# HydroGym devcontainer

Modular devcontainer setup: a shared base image (NVHPC 26.5 + CUDA 13.2,
apt OpenMPI, the parallel HDF5/netCDF/FFTW stacks) plus per-solver
"features" that each pinned `*.devcontainer.json` config composes.
Available features so far:

| Feature              | What it builds                                                   |
|----------------------|------------------------------------------------------------------|
| `./features/petsc`     | PETSc 3.25 + OpenBLAS with the GCC/OpenMPI toolchain             |
| `./features/firedrake` | Firedrake (pinned commit) + a HydroGym venv at `/opt/venvs/firedrake` (requires `petsc`) |

Configs:

| Config                             | Features              | Use                                   |
|------------------------------------|-----------------------|---------------------------------------|
| `petsc-test.devcontainer.json`     | `petsc`               | PETSc build smoke test                |
| `firedrake-test.devcontainer.json` | `petsc` + `firedrake` | Firedrake test suite (`test/`)        |

Build one with the devcontainer CLI, e.g.

```bash
devcontainer up --workspace-folder . \
  --override-config .devcontainer/firedrake-test.devcontainer.json
```

`devcontainer.json` (what VS Code / Codespaces open by default) still
points at the prebuilt `lpaehler/hydrogym-devpod:stable` image and is not
affected by these configs.

## Base image

`base/Dockerfile.base` builds parallel-netcdf 1.14.0, parallel HDF5 1.14.5
and FFTW 3.3.10 from source with the GNU/apt-OpenMPI toolchain into
`/opt/maia_deps_gnu`. A second copy built with NVHPC's own toolchain into
`/opt/maia_deps` is only needed by a GPU MAIA build and is gated behind
the `BUILD_GPU_LIBS` build arg (default `false`): the two library stacks
are not ABI-compatible, and building both roughly doubles the image build
time. The `PSTL_PRESET`/`BUILD_TYPE` build args that the configs pass are
not read by `Dockerfile.base`; they are kept for the GPU MAIA feature's
configs and are no-ops here.

## Workspace layout

`workspaceMount` binds the repo itself (`${localWorkspaceFolder}`) to
`/workspace` in the container, so there are no host-specific paths to
edit. Every feature's `postCreateCommand.sh` runs against that mount once
the container is up (the `install.sh` scripts run at image-build time,
before the mount exists, and only persist feature options under
`/opt/maia-feature-config/`).

## Gotchas

- **Build parallelism is RAM-aware, not a flat `-j16`**: the base image
  build and the feature builds get their `-j` value from
  `.devcontainer/base/compute_parallelism.sh`
  (`min(nproc, available_RAM_GB / 5)`) instead of a number hand-picked for
  one machine; some translation units peak at ~4.2GB RSS each, so a flat
  `-j16` either under-uses a big box or OOM-kills the compiler on a
  smaller one.
- **Dev builds report no `__version__`**: both this HydroGym checkout and
  the Firedrake dev build lack a `__version__` attribute, so version
  checks in the feature scripts use `importlib.metadata.version(...)`
  instead of a bare attribute access.
