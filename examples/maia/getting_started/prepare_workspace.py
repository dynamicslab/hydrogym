#!/usr/bin/env python3
"""
Prepare MAIA workspace for HPC job submission.

This script should be run BEFORE submitting your job to create the
workspace with proper symlinks. Then your job script can use the workspace.

Usage:
    python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./run_001
"""

import argparse
from hydrogym.maia.workspace import prepare_maia_workspace  # avoids mpi4py init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MAIA workspace")
    parser.add_argument("--env", required=True, help="Environment name (e.g., Cylinder_2D_Re200)")
    parser.add_argument("--work-dir", default=None, help="Work directory (default: auto-generated)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download from HF Hub")
    args = parser.parse_args()

    print(f"Preparing workspace for {args.env}...")

    work_dir, properties_file = prepare_maia_workspace(
        environment_name=args.env,
        work_dir=args.work_dir,
        force_download=args.force_download,
    )

    print("\n" + "="*70)
    print("Workspace ready for HPC job submission!")
    print("="*70)
    print(f"Work directory:   {work_dir}")
    print(f"Properties file:  {properties_file}")
    print("\nNext steps:")
    print(f"  1. Edit your job script to use work_dir: {work_dir}")
    print(f"  2. Submit job: sbatch job.slurm (or equivalent)")
    print("="*70)
