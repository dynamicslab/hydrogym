# Large channel (turbulent channel flow, DRL coupling)

Nek5000 case of the large turbulent channel of HydroGym, built against the
KTH Nek5000 core and Toolbox in `../../solver/KTH_DRL_Framework` (git submodules).

## Build

1. Compilers: [compile_script](./compile_script) sets

        export FC="mpif77 "${TOOLBOX_INC}
        export CC="mpicc"

   Adapt them to the MPI compiler wrappers of your system.

2. Channel parameters (geometry and initial condition): [TCF_PARAM](./TCF_PARAM).
   The domain height (wall-normal) is 2.0.

3. Array sizes and run-time parameters: [SIZE](./SIZE).

4. Compile:

        ./compile_script --all

   Clean the build (makenek asks whether to clean the 3rd-party libraries too,
   which is usually not necessary):

        ./compile_script --clean

The HydroGym images build this case with `.devcontainer/features/nek5000/build_nek5000.sh`.

## Mesh

The mesh is generated from [msh/channel.box](./msh/channel.box) with genbox and genmap:

    cd msh
    bash mesh-script

genbox asks for the input file (`channel.box`); genmap asks for the case name
(`box`) and a tolerance below 0.2 (e.g. `0.00001`). The script moves
`box.re2` and `box.ma2` to `../tcf.re2` and `../tcf.ma2`.

## User routines

The user-defined subroutines (boundary conditions, initial condition, DRL
hooks) are in [tcf.usr](./tcf.usr); makenek generates `tcf.f` from it.
