# Small wing (NACA4412, Re_c = 75 000, AoA 5 deg, coarse mesh)

Nek5000 case of the small NACA4412 wing with DRL coupling, built against the
vendored core `../../solver/DEC_DRL_Nek5000` (see its VENDORED_README.md). The
DRL and statistics sources follow the naca0012_200k case.

## Files

| file | content |
|---|---|
| `SIZE` | array sizes and run-time parameters |
| `small_wing.rea` | physical and numerical parameters (Re_c = 75 000: p002) |
| `small_wing.usr` | boundary and initial conditions, DRL hooks (makenek generates `small_wing.f` from it) |
| `small_wing.re2`, `small_wing.ma2`, `small_wing.map` | mesh and its partition map |
| `small_wing.wall` | wall definition and global element numbers |
| `small_wing.restart` | restart flag file |
| `mask_small_wing0.f00002` | field with the nearest wall point of every grid point (wall-normal directions) |
| `inc_src/`, `drl/`, `stats/` | include files, DRL coupling, statistics and time series |

The initial fields are part of the HydroGym Hugging Face
dataset (`NACA4412_3D_Re75000_AOA5`), not of this directory.

## Build

    ./compile_script small_wing

compile_script uses the generic compiler flags

    G="-I./inc_src -mcmodel=large -std=legacy"

The HydroGym images build this case with `.devcontainer/features/nek5000/build_nek5000.sh`.

## Observation frame

The observation-frame option of naca0012_200k (PARAM(92), HydroGym
`simulation.frame`) is not available in this case: both sides of the small wing
are actuated, and the observations use the projection of the reference
implementation. A non-zero p092 is reported in the log and has no effect.
