# HydroGym devcontainer

Modular devcontainer setup: a shared base image (NVHPC 26.5 + CUDA 13.2,
apt OpenMPI, the parallel HDF5/netCDF/FFTW stacks) plus per-solver
"features" you compose per config (`maia-gpu`, `maia-cpu`, `nek5000`,
`petsc`, `firedrake`, `python-ml`, `hydrogym-gpu`, `jax-fluids`). Start
from `devcontainer-template.json` to see all available features and
example combinations, or open one of the pinned `*.devcontainer.json`
configs directly (`devcontainer.json` is the default: MAIA GPU only).

## You must set your GPU architecture before building

The `maia-gpu` feature compiles MAIA with NVHPC's stdpar for a **specific**
GPU architecture (`-gpu=ccXX`) — it is not a portable binary. Every config
in this directory ships with `pstlPreset` hardcoded to `"ada"` (an RTX
40-series / Ada Lovelace card), because that's the architecture of the
machine this setup was developed on. **If your GPU isn't Ada Lovelace, you
must change this before building**, or MAIA will compile for the wrong
architecture and fail to run (or run without GPU acceleration) on your
hardware.

The one place that matters is the `pstlPreset` option under
`features["./features/maia-gpu"]` in whichever `*.devcontainer.json` you're
using, e.g.:

```jsonc
"features": {
  "./features/maia-gpu": {
    "pstlPreset": "ampere",   // <- change this to match your GPU
    "buildType": "production"
  }
}
```

`build.args.PSTL_PRESET` (in the same files, under `"build"`) is **not**
load-bearing — `Dockerfile.base` declares no matching `ARG`, so it's
silently ignored by Docker. Don't rely on editing it; only the feature
option above actually reaches `configure.py --enable-pstl`.

Valid `pstlPreset` values and the GPUs they correspond to:

| Preset               | Compute capability | Example GPUs                  |
|-----------------------|--------------------|--------------------------------|
| `volta`               | cc70               | V100, Titan V                  |
| `turing`               | cc75               | RTX 20-series, GTX 16-series, T4 |
| `ampere`               | cc80 / cc86        | A100, RTX 30-series             |
| `ada`                  | cc89               | RTX 40-series, L4, L40          |
| `hopper`               | cc90               | H100, H200                      |
| `blackwell`            | cc100              | B100, B200 (datacenter)         |
| `blackwell_consumer`   | cc120              | RTX 50-series                   |
| `multicore`            | —                  | CPU-only stdpar fallback, no GPU |
| `HOST`                 | —                  | let `nvc++` auto-detect the build machine's GPU at compile time |

Not sure which one you have? Run `nvidia-smi --query-gpu=name,compute_cap
--format=csv` on the host before building.

## `third_party/m-AIA` submodule — feature parity caveat

MAIA's development happens in a private tree (`wipmaiaml`) that this
public repo can't vendor directly. `third_party/m-AIA` is instead a
submodule of RWTH's public mirror
(`git.rwth-aachen.de/aia/m-AIA/m-AIA`). **This mirror does not currently
have the RL-relevant features that make MAIA useful for HydroGym's flow
control cases** — specifically the LB jet-actuation boundary conditions
(BC `2007`/`2008`) and the MPMD-based external flow-control channel. Both
`maia-gpu` and `maia-cpu` build successfully against this submodule, but
environments that depend on jet actuation won't work until a fully
RL-featured checkout is public. Swap the submodule (`third_party/m-AIA`)
for that checkout, or repoint `.gitmodules`, once one is available.

The mirror also doesn't ship `auxiliary/hosts/DEVCONTAINER.cmake` (the
NVHPC/CUDA/library-path host config this container needs), since a host
config is inherently environment-specific and was never meant to live
upstream in the first place. `maia-gpu/postCreateCommand.sh` materializes
it at container-start from `.devcontainer/base/DEVCONTAINER.cmake` if the
submodule doesn't already have one, and syncs its `PSTL` value to whatever
`pstlPreset` you set — this becomes a no-op automatically if a future
mirror update adds the file itself.

## Workspace layout

`workspaceMount` binds the repo itself (`${localWorkspaceFolder}`) to
`/workspace` in the container — no host-specific paths to edit. Every
solver's source lives inside this one clone: `third_party/m-AIA` (MAIA,
see caveat above), `third_party/nek5000` (Nek5000 core + KTH Toolbox +
case sources), `third_party/firedrake`.

## Gotchas

- **`-j16`, not higher**: several postCreateCommand steps do native
  parallel compiles (MAIA, PETSc, Firedrake, Nek5000 cases) — some
  translation units peak at ~4.2GB RAM each, so building more of these at
  once than your machine's RAM/16 supports risks the compiler getting
  OOM-killed. `build_all_containers.sh` runs configs sequentially for this
  reason, not in parallel.
- **MAIA host-config auto-detection is hostname-based, not env-var based**:
  `configure.py`'s host detection (`cmake/GetHost.cmake`) pattern-matches
  the container's OS hostname against a hardcoded cluster list — it does
  NOT read `MAIA_HOST`/`HOST` env vars, and this container's hostname
  (`--hostname=LocalGPU` in `runArgs`) doesn't match anything in that list.
  `MAIA_HOST_FILE` (see `maia-gpu`/`maia-cpu` `postCreateCommand.sh`) is the
  real override — it points straight at a host config file and skips
  hostname detection entirely, so none of this matters in practice.
- **Dev builds report no `__version__`**: both this HydroGym checkout and
  the Firedrake dev build lack a `__version__` attribute, so version
  checks here use `importlib.metadata.version(...)` instead of a bare
  attribute access (see `ensure_hydrogym.sh`).
