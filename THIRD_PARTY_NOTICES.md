# Third-party components

HydroGym itself is released under the MIT licence (`LICENSE`). The repository contains, or builds against, the
components below, which keep their own licences. The container images contain the solvers built from them, so the
terms apply to the images as well. This file lists the components that are part of the repository tree or are
referenced as git submodules; the licence texts in the respective directories are authoritative.

## In the repository tree

| Path | Component | Licence |
|---|---|---|
| `third_party/nek5000/solver/DEC_DRL_Nek5000/` | Nek5000 (UChicago Argonne, LLC), fork used by the Nek5000 environments | GNU GPL v3 or later (`LICENSE` in that directory) |
| `third_party/nek5000/cases/*/` (files with a `Copyright : GPL` header, e.g. `naca0012_200k/stats/*.f`, `err_est.f`) | KTH case and statistics sources used by the Nek5000 cases | GPL, as stated in the file headers |
| `hydrogym/jax/env_core.py` (wrappers), `examples/jax/getting_started/3_ppo/run_ppo.py` | adapted from PureJaxRL (Chris Lu, 2023) | Apache License 2.0 (`hydrogym/jax/LICENSE-purejaxrl`) |
| `hydrogym/jax/solvers/channel_fd.py` | re-implementation in JAX of the discretisation and time stepping of `Channel3dLab` of DNSLab (V. Vuorinen, K. Keskinen, Comput. Phys. Commun. 203 (2016) 278-289; CPC Program Library AEZX_v1_0, Mendeley Data doi 10.17632/6gtnjwwg8j.1); no Matlab code is copied | HydroGym MIT licence for the re-implementation; the DNSLab package keeps its CPC Program Library licence |

HydroGym's own Python code and the case-specific files that do not carry another licence header are MIT. The Nek5000
cases are combined with the GPL Nek5000 solver when they are built and run; if you redistribute built Nek5000
binaries, the GPL terms apply to them.

## Git submodules (`.gitmodules`)

`third_party/m-AIA` (m-AIA solver, LGPL-3.0), `third_party/nekrs` (NekRS), `third_party/firedrake` (Firedrake),
`third_party/nek5000/solver/KTH_DRL_Framework/{Nek5000,Toolbox}` (Nek5000, KTH Toolbox) and `third_party/NekStatsPy`
are separate projects with their own licences; see the licence file in each repository.

## Data

The environment data and policies on the Hugging Face Hub are licensed under CC BY 4.0 (see the dataset card).
