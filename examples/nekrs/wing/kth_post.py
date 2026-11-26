"""
KTH post-processing (NekStatsPy, third_party/NekStatsPy) of wing runs: RUN/ZSTAT/int_fld from
`bl_stats.py --int-mesh` -> RUN/stat_nekstat.mat, the same Re200k structure as the KTH benchmark statistics
of the case: rotation into the wall frame, inner scaling, boundary-layer edge
(diagnostic plot), Cf / Cp, Reynolds-stress budgets and the Cl / Cd decomposition, all with the NekStatsPy routines.

    python bl_stats.py runs/zero_trip60 --skip 500 \
        --int-mesh ../../../third_party/NekStatsPy/data/meshes/naca0012/int_mesh.mat
    python kth_post.py runs/zero_trip60 runs/zero_ref1

The export window is that of the benchmark file: profiles 4..80 of the 80 per side (istart 3, iend 80), AoA 0.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEKSTAT = os.path.join(HERE, "..", "..", "..", "third_party", "NekStatsPy")

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--nekstat", default=NEKSTAT, help="NekStatsPy checkout (src/nekstat, data/meshes/naca0012)")
p.add_argument("--mesh", help="default NEKSTAT/data/meshes/naca0012/int_mesh.mat")
p.add_argument("--istart", type=int, default=3)
p.add_argument("--iend", type=int, default=80)
args = p.parse_args()
sys.path.insert(0, os.path.join(args.nekstat, "src"))
from nekstat.pipeline import run  # noqa: E402

mesh = args.mesh or os.path.join(args.nekstat, "data", "meshes", "naca0012", "int_mesh.mat")
for r in args.runs:
    out = os.path.join(r, "stat_nekstat.mat")
    print(f"[KTH] {r}: NekStatsPy on {r}/ZSTAT/int_fld", flush=True)
    run(r, mesh, out_file=out, istart=args.istart, iend=args.iend, long=1, aoa_deg=0.0)
    print(f"[KTH] {out} written", flush=True)
