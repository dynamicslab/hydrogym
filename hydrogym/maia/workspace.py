"""
MAIA Workspace Setup
====================

This module provides utilities for setting up MAIA workspaces with proper
file structure for MPMD coupling. It handles downloading from HF Hub and
creating necessary symbolic links with correct file extensions.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from hydrogym.data_manager import HFDataManager


class MaiaWorkspace:
    """
    Manages MAIA workspace setup including file downloads and symbolic links.

    File preparation is driven by solver profiles in
    :data:`~hydrogym.data_manager.SOLVER_PROFILES`.  The correct profile
    (``'MAIA_LB'`` or ``'MAIA_STRCTRD'``) is auto-detected from sentinel files.
    """

    def __init__(
        self,
        environment_name: str,
        work_dir: Optional[str] = None,
        hf_repo_id: str = "dynamicslab/HydroGym-environments",
        local_fallback_dir: Optional[str] = None,
        use_clean_cache: bool = True,
        solver_type: Optional[str] = None,
    ):
        """
        Initialize the MAIA workspace.

        Args:
            environment_name: Name of the environment (e.g., 'Cylinder_2D_Re200').
            work_dir: Working directory path. If None, uses './run_{environment_name}'.
            hf_repo_id: Hugging Face repository ID.
            local_fallback_dir: Optional local fallback directory.
            use_clean_cache: Whether to use clean cache for HF downloads.
            solver_type: Solver profile key (``'MAIA_LB'`` or ``'MAIA_STRCTRD'``).
                Auto-detected from sentinel files if ``None`` (recommended).
                Defaults to ``'MAIA_LB'`` as fallback for legacy environments.
        """
        self.environment_name = environment_name
        self.work_dir = work_dir or f"./run_{environment_name}"

        self.data_manager = HFDataManager(
            repo_id=hf_repo_id,
            local_fallback_dir=local_fallback_dir,
            use_clean_cache=use_clean_cache,
            fallback_profile=solver_type or "MAIA_LB",
        )

        self.env_data_path: Optional[str] = None
        self.solver_profile: Optional[str] = None
        self.is_setup = False

    def setup(self, force_download: bool = False, verbose: bool = True) -> Dict[str, str]:
        """
        Set up the workspace with all required files.

        This method:
        1. Downloads/locates environment data from HF Hub
        2. Creates work directory structure
        3. Creates symbolic links with proper names/extensions

        Args:
            force_download: Force re-download from HF Hub.
            verbose: Print setup progress.

        Returns:
            Dictionary containing paths to key files:
                - 'work_dir': Path to work directory
                - 'properties_file': Path to properties file for MAIA
                - 'config_file': Path to environment config file
                - 'env_data_path': Path to source environment data
        """
        if verbose:
            print(f"=== Setting up MAIA workspace for {self.environment_name} ===")
            print("1. Fetching environment data...")

        self.env_data_path = self.data_manager.get_environment_path(
            self.environment_name, force_download=force_download
        )

        if verbose:
            print(f"   Environment data ready: {self.env_data_path}")
            print(f"2. Creating workspace: {self.work_dir}")

        paths = self.data_manager.prepare_working_directory(self.env_data_path, self.work_dir)

        # Store detected solver profile for cleanup
        self.solver_profile = paths.get("solver_profile", self.data_manager.fallback_profile)

        self.is_setup = True

        if verbose:
            print("=== Workspace setup complete ===")
            print(f"   Work directory:   {paths['work_dir']}")
            print(f"   Solver profile:   {self.solver_profile}")
            print(f"   Properties file:  {paths.get('properties_file', 'N/A')}")
            print()

        return paths

    def cleanup(self, remove_work_dir: bool = False, verbose: bool = True) -> None:
        """
        Clean up the workspace.

        Args:
            remove_work_dir: If True, removes the entire work directory.
                           If False, only removes symbolic links.
            verbose: Print cleanup messages.
        """
        if not self.is_setup:
            return

        work_path = Path(self.work_dir)

        if remove_work_dir:
            if verbose:
                print(f"Removing work directory: {self.work_dir}")
            if work_path.exists():
                shutil.rmtree(work_path)
        else:
            if verbose:
                print(f"Removing symbolic links in: {self.work_dir}")
            from hydrogym.data_manager import SOLVER_PROFILES

            # Use detected profile instead of hardcoded MAIA_LB
            profile = SOLVER_PROFILES.get(self.solver_profile or "MAIA_LB", {})
            for target_rel in profile.get("workspace_files", {}).values():
                target_path = work_path / target_rel
                if target_path.is_symlink():
                    target_path.unlink()
            for target_rel in profile.get("workspace_dirs", {}).values():
                target_path = work_path / target_rel
                if target_path.is_symlink():
                    target_path.unlink()

        self.is_setup = False


def prepare_maia_workspace(
    environment_name: str,
    work_dir: Optional[str] = None,
    hf_repo_id: str = "dynamicslab/HydroGym-environments",
    force_download: bool = False,
    **kwargs,
) -> Tuple[str, str]:
    """
    Convenience function to prepare a MAIA workspace for MPMD coupling.

    This function handles all the file preparation needed before launching
    mpirun with Python and MAIA. It downloads data from HF Hub (if needed)
    and creates symbolic links with proper file extensions.

    Args:
        environment_name: Name of the environment (e.g., 'Cylinder_2D_Re200').
        work_dir: Working directory path. If None, auto-generates name.
        hf_repo_id: Hugging Face repository ID.
        force_download: Force re-download from HF Hub.
        **kwargs: Additional arguments passed to MaiaWorkspace.

    Returns:
        Tuple of (work_dir_path, properties_file_path) for use with mpirun.

    Example:
        >>> import hydrogym.maia as maia
        >>>
        >>> # Prepare workspace (before submitting HPC job)
        >>> work_dir, props_file = maia.prepare_maia_workspace('Cylinder_2D_Re200')
        >>>
        >>> # Then use work_dir and props_file in your job script:
        >>> # sbatch job.slurm  # where job.slurm references work_dir
    """
    workspace = MaiaWorkspace(environment_name=environment_name, work_dir=work_dir, hf_repo_id=hf_repo_id, **kwargs)
    paths = workspace.setup(force_download=force_download, verbose=True)
    return paths["work_dir"], paths["properties_file"]
