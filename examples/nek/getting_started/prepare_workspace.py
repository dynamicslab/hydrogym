#!/usr/bin/env python3
"""
Prepare NEK5000 workspace for testing packaged environments.

This script prepares a workspace from a local or Hugging Face environment package.

Usage:
    # From local packaged directory
    python prepare_workspace.py \
        --local-dir ../../../packaged_envs \
        --env TCFmini_3D_Re180 \
        --work-dir ./test_run_001 \
        --cache-dir ~/.cache/hydrogym_nek

    # From Hugging Face (when uploaded)
    python prepare_workspace.py \
        --env MiniChannel_Re180 \
        --work-dir ./test_run_001
"""

import argparse
# [YW-MOD] Add import os
import sys, os
# [YW-MOD] End
from pathlib import Path

# Add hydrogym to path if needed
sys.path.insert(0, str(Path(__file__).parents[4] / 'hydrogym'))

from hydrogym.data_manager import HFDataManager


def prepare_nek_workspace(
    env_name: str,
    work_dir: str,
    cache_dir: str = None,
    local_dir: str = None,
    force_download: bool = False,
    restart_index: int = 1, # [YW-MOD] Add restart index
    profile: str = 'NEK5000_v19', # [YW-MOD] Add profile
):
  """
    Prepare NEK5000 workspace with runtime files.

    Args:
        env_name: Environment name (e.g., 'TCFmini_3D_Re180')
        work_dir: Target workspace directory
        cache_dir: Clean cache directory (default: ~/.cache/hydrogym_nek)
        local_dir: Local directory containing packaged envs (for testing)
        force_download: Force re-download from HF

    Returns:
        Tuple of (work_dir, par_file_path, cache_dir, env_path)
    """
  # Set default cache directory
  if cache_dir is None:
    cache_dir = Path.home() / '.cache' / 'hydrogym_nek'
  else:
    cache_dir = Path(cache_dir).expanduser()

  cache_dir = cache_dir.resolve()

  print("=" * 70)
  print("NEK5000 Workspace Preparation")
  print("=" * 70)
  print(f"Environment:      {env_name}")
  print(f"Clean cache:      {cache_dir}")
  print(f"Work directory:   {work_dir}")
  if local_dir:
    print(f"Source:           Local ({local_dir})")
  else:
    print("Source:           Hugging Face Hub")
  print("=" * 70)

  # Create data manager with explicit clean cache (copy mode)
  print("\nInitializing data manager...")
  dm = HFDataManager(
      cache_dir=str(cache_dir),
      local_fallback_dir=local_dir,
      use_clean_cache='copy'  # Copy files to clean cache (not HF hashes)
  )

  # Download/copy environment to clean cache
  print(f"\nStep 1: Getting environment '{env_name}'...")
  environment_path = dm.get_environment_path(
      env_name, force_download=force_download)
  print(f"✓ Environment copied to clean cache: {environment_path}")

  # Prepare workspace with symlinks to clean cache
  print("\nStep 2: Preparing workspace with symlinks...")
  work_paths = dm.prepare_working_directory(
      env_path=environment_path, work_dir=work_dir,)

  work_dir_resolved = work_paths['work_dir']

  # [YW-MOD] Look into the work_dir_resolved, Find the *.par file, and decide the case name
  par_file = None
  for file in os.listdir(work_dir_resolved):
    print(f"Checking file: {file}")
    if profile == 'NEK5000_v17' and file.endswith('.rea') :
      par_file = Path(file)
      case_name = par_file.stem
      break
    elif profile == 'NEK5000_v19' and file.endswith('.par'):
      par_file = Path(file)
      case_name = par_file.stem
      break
  # [YW-MOD] End  

  print(f"par_file: {par_file}")
  par_file = Path(work_dir_resolved) / par_file
  # Create SESSION.NAME file (required by Nek5000)
  print("\nStep 3: Creating Nek5000 session files...")
  session_file = Path(work_dir_resolved) / 'SESSION.NAME'
  with open(session_file, 'w') as f:
    f.write(case_name + '\n')  # [YW-MOD] Case name
    f.write(f'{work_dir_resolved}\n')  # Absolute path to working directory
  print(f"✓ Created SESSION.NAME: {session_file}")

  # [YW-MOD] Symlink int_pos file if it exists (DRL sensor/actuator positions)
  if profile == 'NEK5000_v19':
    int_pos_src = Path(environment_path) / 'int_pos'
    int_pos_dst = Path(work_dir_resolved) / 'int_pos'
  elif profile == 'NEK5000_v17':
    int_pos_src = Path(environment_path) / 'stat_pts.in'
    int_pos_dst = Path(work_dir_resolved) / 'stat_pts.in'
  else:
    raise ValueError(f"Unsupported profile: {profile}")
  if int_pos_src.exists():
    if int_pos_dst.exists():
      int_pos_dst.unlink()
    int_pos_dst.symlink_to(int_pos_src.resolve())
    print(f"✓ Symlinked int_pos: {int_pos_dst}")
  else:
    print("⚠ Warning: int_pos not found (may be needed for DRL coupling)")
  # [YW-MOD] End

  # [YW-MOD] Symlink the mask files if they exist (v17 only)
  if profile == 'NEK5000_v17':
    mask_files = [f for f in os.listdir(environment_path) if f.startswith('mask_') ]
    for mask_file in mask_files:
      mask_src = Path(environment_path) / mask_file
      mask_dst = Path(work_dir_resolved) / mask_file
      if mask_src.exists():
        if mask_dst.exists():
          mask_dst.unlink()
      mask_dst.symlink_to(mask_src.resolve())
      print(f"✓ Symlinked mask file: {mask_dst}")
    else:
      print(f"⚠ Warning: mask file not found: {mask_file}")
  # [YW-MOD] End

  # [YW-MOD] Copy the restart files with from work_dir_resolved/restart_files to work_dir_resolved/
  import shutil
  restart_src = Path(work_dir_resolved) / 'restart_files'/ f'init_{restart_index}'
  if restart_src.exists():
    for rs_file in os.listdir(restart_src):
      shutil.copy(restart_src / rs_file, Path(work_dir_resolved) / rs_file)
    print(f"✓ Copied restarts: {Path(work_dir_resolved) / rs_file} with index {restart_index}")
  else:
    print(f"⚠ Warning: restart_files not found in {work_dir_resolved}")
  # [YW-MOD] End

  # [YW-MOD] Print the case_name
  print("\n" + "=" * 70)
  print("Workspace Ready!")
  print("=" * 70)
  print(f"Clean cache:      {cache_dir}/{env_name}/")
  print(f"  Files copied:   {case_name}.re2, {case_name}.ma2, {case_name}.par, restarts/")
  print(f"\nWork directory:   {work_dir_resolved}")
  print(f"  Symlinks to:    {cache_dir}/{env_name}/")
  print(f"  Parameter file: {case_name}.par")
  print("\nFile structure:")
  print(f"  1. Source:      {local_dir or 'Hugging Face'}/{env_name}/")
  print(f"  2. Clean cache: {cache_dir}/{env_name}/ (copied)")
  print(f"  3. Workspace:   {work_dir_resolved}/ (symlinks + config)")
  print("=" * 70)
  # [YW-MOD] End

  return work_dir_resolved, str(par_file), str(cache_dir), str(environment_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Prepare NEK5000 workspace for MPMD execution",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  # Local packaged environment
  python prepare_workspace.py \\
      --local-dir ../../../packaged_envs \\
      --env TCFmini_3D_Re180 \\
      --work-dir ./test_run_001

  # From Hugging Face
  python prepare_workspace.py \\
      --env MiniChannel_Re180 \\
      --work-dir ./test_run_001

  # Custom cache directory
  python prepare_workspace.py \\
      --local-dir ../../../packaged_envs \\
      --env TCFmini_3D_Re180 \\
      --work-dir ./test_run_001 \\
      --cache-dir /scratch/hydrogym_cache
        """)

  parser.add_argument(
      "--env", required=True, help="Environment name (e.g., TCFmini_3D_Re180)")
  parser.add_argument(
      "--work-dir", required=True, help="Work directory for execution")
  parser.add_argument(
      "--cache-dir",
      default=None,
      help="Clean cache directory (default: ~/.cache/hydrogym_nek)")
  parser.add_argument(
      "--local-dir",
      default=None,
      help="Local directory with packaged environments (for testing, skips HF)")
  parser.add_argument(
      "--force-download",
      action="store_true",
      help="Force re-download/re-copy from source")
  # [YW-MOD] Add restart index
  parser.add_argument(
      "--restart-index",
      type=int,
      default=1,
      help="Restart index (default: 1)")
  parser.add_argument(
      "--profile",
      type=str,
      default='NEK5000_v19',
      help="Profile (default: NEK5000_v19)")
  # [YW-MOD] End

  args = parser.parse_args()

  try:
    work_dir, par_file, cache_dir, env_path = prepare_nek_workspace(
        env_name=args.env,
        work_dir=args.work_dir,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        force_download=args.force_download,
        restart_index=args.restart_index, # [YW-MOD] Add restart index
        profile=args.profile, # [YW-MOD] Add profile
    )

    print("\nNext steps:")
    print(f"  1. Inspect cached files: ls -lh {env_path}")
    print(f"  2. Review configuration: cat {par_file}")
    # [YW-MOD] Add restart index
    print(f"  3. Restart index: {args.restart_index}")
    # [YW-MOD] End
    print("  4. Run test: ./run_example.sh")

  except Exception as e:
    print(f"\n✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
