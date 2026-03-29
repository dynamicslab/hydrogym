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

from .hf_data_manager import HFDataManager


class MaiaWorkspace:
    """
    Manages MAIA workspace setup including file downloads and symbolic links.

    This class handles the file preparation needed before launching MPMD coupling,
    ensuring all files have proper names and extensions for the MAIA solver.
    """

    # Required files and their expected names in the work directory
    REQUIRED_FILES = {
        "grid.Netcdf": "out_lb/grid.Netcdf",
        "restart_.Netcdf": "out_lb/restart_.Netcdf",
        "properties_run.toml": "properties_run.toml",
        "geometry.toml": "geometry.toml",
        "environment_config.yaml": "environment_config.yaml",
    }

    # Required directories
    REQUIRED_DIRS = {
        "stl": "stl",
    }

    def __init__(
        self,
        environment_name: str,
        work_dir: Optional[str] = None,
        hf_repo_id: str = "dynamicslab/HydroGym-environments",
        local_fallback_dir: Optional[str] = None,
        use_clean_cache: bool = True,
    ):
        """
        Initialize the MAIA workspace.

        Args:
            environment_name: Name of the environment (e.g., 'Cylinder_2D_Re200').
            work_dir: Working directory path. If None, uses './run_{environment_name}'.
            hf_repo_id: Hugging Face repository ID.
            local_fallback_dir: Optional local fallback directory.
            use_clean_cache: Whether to use clean cache for HF downloads.
        """
        self.environment_name = environment_name
        self.work_dir = work_dir or f"./run_{environment_name}"
        self.hf_repo_id = hf_repo_id

        # Initialize data manager
        self.data_manager = HFDataManager(
            repo_id=hf_repo_id,
            local_fallback_dir=local_fallback_dir,
            use_clean_cache=use_clean_cache,
        )

        self.env_data_path = None
        self.is_setup = False

    def setup(self, force_download: bool = False, verbose: bool = True) -> Dict[str, str]:
        """
        Setup the workspace with all required files.

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

        # Step 1: Get environment data from HF Hub or cache
        if verbose:
            print(f"1. Fetching environment data...")

        # First check ~/.cache/maiagym/ for existing data
        cache_dir = Path.home() / ".cache" / "maiagym" / self.environment_name
        if cache_dir.exists() and not force_download:
            self.env_data_path = str(cache_dir)
            if verbose:
                print(f"   Using cached data: {self.env_data_path}")
        else:
            # Download/get from HF Hub
            self.env_data_path = self.data_manager.get_environment_path(
                self.environment_name, force_download=force_download
            )
            if verbose:
                print(f"   Environment data ready: {self.env_data_path}")

        # Step 2: Create work directory structure
        if verbose:
            print(f"2. Creating work directory: {self.work_dir}")

        work_path = Path(self.work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        # Create output subdirectory for MAIA
        out_lb_path = work_path / "out_lb"
        out_lb_path.mkdir(exist_ok=True)

        # Step 3: Create symbolic links for all required files
        if verbose:
            print(f"3. Creating symbolic links...")

        paths = {
            "work_dir": str(work_path.absolute()),
            "env_data_path": self.env_data_path,
        }

        # Link files
        for source_name, target_rel in self.REQUIRED_FILES.items():
            source_path = Path(self.env_data_path) / source_name
            target_path = work_path / target_rel

            if source_path.exists():
                self._create_link(source_path, target_path, verbose=verbose)

                # Store important file paths
                if source_name == "properties_run.toml":
                    paths["properties_file"] = str(target_path.absolute())
                elif source_name == "environment_config.yaml":
                    paths["config_file"] = str(target_path.absolute())
            else:
                print(f"   WARNING: Source file not found: {source_path}")

        # Link directories
        for source_name, target_rel in self.REQUIRED_DIRS.items():
            source_path = Path(self.env_data_path) / source_name
            target_path = work_path / target_rel

            if source_path.exists():
                self._create_link(source_path, target_path, verbose=verbose)
            else:
                print(f"   WARNING: Source directory not found: {source_path}")

        self.is_setup = True

        if verbose:
            print(f"=== Workspace setup complete ===")
            print(f"   Work directory: {paths['work_dir']}")
            print(f"   Properties file: {paths.get('properties_file', 'N/A')}")
            print()

        return paths

    def _create_link(self, source: Path, target: Path, verbose: bool = True) -> None:
        """
        Create a symbolic link, handling existing files/links.

        Args:
            source: Source file/directory path.
            target: Target link path.
            verbose: Print creation messages.
        """
        # Get absolute path of source
        abs_source = source.resolve()

        # Remove existing link/file if it exists
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()

        # Create parent directory if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Create symlink
        target.symlink_to(abs_source)

        if verbose:
            print(f"   Linked: {target.name} -> {abs_source}")

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

            # Remove only the symlinks we created
            for target_rel in self.REQUIRED_FILES.values():
                target_path = work_path / target_rel
                if target_path.is_symlink():
                    target_path.unlink()

            for target_rel in self.REQUIRED_DIRS.values():
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
    Convenient function to prepare a MAIA workspace for MPMD coupling.

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
