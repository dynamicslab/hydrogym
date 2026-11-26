#!/bin/bash
# Generate the mesh of the minimal channel from a .box file (run in this directory).
NEK_BIN="../../../solver/KTH_DRL_Framework/Nek5000/bin"
"${NEK_BIN}/genbox"
"${NEK_BIN}/genmap"
mv box.re2 ../phill.re2
mv box.ma2 ../phill.ma2
