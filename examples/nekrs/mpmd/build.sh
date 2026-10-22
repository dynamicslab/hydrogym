#!/usr/bin/env bash
# Build nekrs-mpmd: the NekRS 26.0 driver (src/bin/driver.cpp) with nekrs-26.0_driver-mpmd-split.patch applied,
# linked against the installed libnekrs. Run inside the NekRS image (see ../nekrs.sh), from this directory.
set -e
mpicxx -O2 -std=c++17 -I. -I"$NEKRS_HOME/include/lib" driver.cpp -o nekrs-mpmd \
  -L"$NEKRS_HOME/lib" -lnekrs -Wl,-rpath,"$NEKRS_HOME/lib"
echo "built $(pwd)/nekrs-mpmd"
