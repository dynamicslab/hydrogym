"""
Initial field of the wing deployment: the developed uncontrolled flow rs8naca_wing0.f00008 (21 GB, N = 7) of the
HydroGym environment NACA0012_3D_Re200000_AOA0 (Hugging Face dataset dynamicslab/HydroGym-environments,
restart_files/init_1/). It goes to restart/ next to this script, where run.sh expects it (START). The mesh
naca_wing.re2 is part of the repository (third_party/nek5000/cases/naca0012_200k).

    python prepare_case.py                  # download (huggingface_hub; HF_HOME for the cache)
    python prepare_case.py --source DIR     # link (hard link if possible, else symlink) from a local copy of
                                            # NACA0012_3D_Re200000_AOA0
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "restart_files/init_1/rs8naca_wing0.f00008"
CASE = "NACA0012_3D_Re200000_AOA0"

p = argparse.ArgumentParser()
p.add_argument("--source", help="local NACA0012_3D_Re200000_AOA0 directory instead of downloading")
p.add_argument("--repo-id", default="dynamicslab/HydroGym-environments")
args = p.parse_args()

dest = os.path.join(HERE, "restart")
os.makedirs(dest, exist_ok=True)
target = os.path.join(dest, os.path.basename(REL))
if os.path.exists(target):
    print(f"{target} exists")
elif args.source:
    src = os.path.abspath(os.path.join(args.source, REL))
    try:
        os.link(src, target)
        print(f"{target}: hard link to {src}")
    except OSError:
        os.symlink(src, target)
        print(f"{target} -> {src} (symlink; run.sh mounts its directory into the container)")
else:
    from huggingface_hub import hf_hub_download

    # straight into restart/ (local_dir), no second 21 GB copy from the cache
    path = hf_hub_download(args.repo_id, f"{CASE}/{REL}", repo_type="dataset", local_dir=os.path.join(dest, ".hf"))
    os.replace(path, target)
    print(f"{target} written")
