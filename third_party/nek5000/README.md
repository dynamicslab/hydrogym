# Nek5000 sources for the HydroGym Nek5000 backend

This tree holds everything `.devcontainer/features/nek5000` needs to build
the solver binaries for HydroGym's Nek5000 cases. It mixes git submodules
and vendored (directly tracked) sources on purpose:

| Path                               | Kind      | Contents |
|------------------------------------|-----------|----------|
| `solver/KTH_DRL_Framework/Nek5000` | submodule | Nek5000 core (github.com/Nek5000/nek5000, `master`) |
| `solver/KTH_DRL_Framework/Toolbox` | submodule | KTH Toolbox (github.com/adampep/KTH_Toolbox): the frame / monitor / runparam / tsrs DRL hooks the channel cases link against |
| `solver/DEC_DRL_Nek5000`           | vendored  | An older, self-contained Nek5000 core that the two airfoil cases' own `compile_script` builds against. Not interchangeable with the KTH core above |
| `cases/mini_channel`               | vendored  | `phill` case, Hub environment `TCFmini_3D_Re180` (KTH core + Toolbox) |
| `cases/large_channel`              | vendored  | `tcf` case, Hub environments `TCFlarge_3D_Re206/207` (KTH core + Toolbox) |
| `cases/naca0012_200k`              | vendored  | `naca_wing` case, Hub environment `NACA0012_3D_Re200000_AOA0` (DEC core) |
| `cases/small_wing`                 | vendored  | `small_wing` case, Hub environment `NACA4412_3D_Re75000_AOA5` (DEC core) |

The build recipes live in `.devcontainer/features/nek5000/postCreateCommand.sh`:
the channel cases go through `nekconfig -build-dep` + `makenek` with the
Toolbox on the include path, the airfoil cases through their committed
`compile_script` with the DEC core as `SOURCE_ROOT`.

Mesh, map and checkpoint inputs (`*.re2`, `*.ma2`, `*.map`, `*.f?????`)
are stored in Git LFS; run `git lfs pull` after cloning. The vendored
trees are excluded from ruff, isort and codespell (`pyproject.toml`) and
are not code HydroGym maintains; fixes belong upstream, with the
exception of the `makefile_usr.inc` object-name fixes needed to build
against the pinned Toolbox.
