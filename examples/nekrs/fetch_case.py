"""
Fetch the mesh and initial field of the NekRS channel cases from the HydroGym environment TCFlarge_3D_Re207
(Hugging Face dataset dynamicslab/HydroGym-environments): tcf.re2 and restart_files/init_1/rs6tcf0.f00004.
They go to channel/ and are linked into legacy_drl/ and legacy_drl/uncontrolled/.

    python fetch_case.py                      # download
    python fetch_case.py --source DIR         # copy from a local copy of TCFlarge_3D_Re207
"""

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"tcf.re2": "tcf.re2", "rs6tcf0.f00004": "restart_files/init_1/rs6tcf0.f00004"}

p = argparse.ArgumentParser()
p.add_argument("--source", help="local TCFlarge_3D_Re207 directory instead of downloading")
p.add_argument("--repo-id", default="dynamicslab/HydroGym-environments")
args = p.parse_args()

dest = os.path.join(HERE, "channel")
for name, rel in FILES.items():
    target = os.path.join(dest, name)
    if os.path.exists(target):
        print(f"{target} exists")
        continue
    if args.source:
        shutil.copyfile(os.path.join(args.source, rel), target)
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.repo_id, f"TCFlarge_3D_Re207/{rel}", repo_type="dataset")
        shutil.copyfile(path, target)
    print(f"{target} written")

for sub in ("legacy_drl", "legacy_drl/uncontrolled"):
    for name in FILES:
        link = os.path.join(HERE, sub, name)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.join(dest, name), os.path.dirname(link)), link)
            print(f"{link} -> channel/{name}")
