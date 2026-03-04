"""
MAIA Workspace Setup
====================

Provides utilities for setting up MAIA workspaces with the correct file
structure for MPMD coupling.  File preparation (downloading from HF Hub and
creating solver-specific symbolic links) is delegated to
:class:`~hydrogym.data_manager.HFDataManager` so that the workspace layout is
defined in one place — the ``SOLVER_PROFILES`` dict in
:mod:`hydrogym.data_manager` — and automatically stays in sync when profile
definitions are updated.
"""

import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from hydrogym.data_manager import HFDataManager


class MaiaWorkspace:
    """
    Manages MAIA workspace setup including file downloads and symbolic links.

    File preparation is driven by the ``'MAIA_LB'`` solver profile in
    :data:`~hydrogym.data_manager.SOLVER_PROFILES`, so updating the profile
    automatically updates the workspace layout.
    """

    def __init__(
        self,
        environment_name: str,
        work_dir: Optional[str] = None,
        hf_repo_id: str = 'dynamicslab/HydroGym-environments',
        local_fallback_dir: Optional[str] = None,
        use_clean_cache: bool = True,
    ):
        """
        Initialize the MAIA workspace.

        Args:
            environment_name: Name of the environment (e.g. ``'Cylinder_2D_Re200'``).
            work_dir: Working directory path.  Defaults to ``./run_{environment_name}``.
            hf_repo_id: Hugging Face repository ID.
            local_fallback_dir: Optional local fallback directory.
            use_clean_cache: Whether to use clean cache for HF downloads.
        """
        self.environment_name = environment_name
        self.work_dir = work_dir or f"./run_{environment_name}"

        self.data_manager = HFDataManager(
            repo_id=hf_repo_id,
            local_fallback_dir=local_fallback_dir,
            use_clean_cache=use_clean_cache,
            fallback_profile='MAIA_LB',
        )

        self.env_data_path: Optional[str] = None
        self.is_setup = False

    def setup(self, force_download: bool = False, verbose: bool = True) -> Dict[str, str]:
        """
        Set up the workspace with all required files.

        Steps:
        1. Download / locate environment data from HF Hub.
        2. Delegate workspace preparation (directory creation + symlinks) to
           :meth:`~hydrogym.data_manager.HFDataManager.prepare_working_directory`.

        Args:
            force_download: Force re-download from HF Hub.
            verbose: Print setup progress.

        Returns:
            Dictionary with at least:
            - ``'work_dir'``: absolute path to the work directory.
            - ``'env_data_path'``: path to the source environment data.
            - ``'properties_file'``: path to ``properties_run.toml`` symlink (if present).
            - ``'config_file'``: path to ``environment_config.yaml`` symlink (if present).
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

        paths = self.data_manager.prepare_working_directory(
            self.env_data_path, self.work_dir
        )

        self.is_setup = True

        if verbose:
            print("=== Workspace setup complete ===")
            print(f"   Work directory:   {paths['work_dir']}")
            print(f"   Properties file:  {paths.get('properties_file', 'N/A')}")
            print()

        return paths

    def cleanup(self, remove_work_dir: bool = False, verbose: bool = True) -> None:
        """
        Clean up the workspace.

        Args:
            remove_work_dir: If ``True``, remove the entire work directory.
                             If ``False``, only remove the symbolic links created
                             during :meth:`setup`.
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
            profile = SOLVER_PROFILES.get('MAIA_LB', {})
            for target_rel in profile.get('workspace_files', {}).values():
                target_path = work_path / target_rel
                if target_path.is_symlink():
                    target_path.unlink()
            for target_rel in profile.get('workspace_dirs', {}).values():
                target_path = work_path / target_rel
                if target_path.is_symlink():
                    target_path.unlink()

        self.is_setup = False


def prepare_maia_workspace(
    environment_name: str,
    work_dir: Optional[str] = None,
    hf_repo_id: str = 'dynamicslab/HydroGym-environments',
    force_download: bool = False,
    **kwargs
) -> Tuple[str, str]:
    """
    Convenience function to prepare a MAIA workspace for MPMD coupling.

    Downloads data from HF Hub (if needed) and creates symbolic links with
    the layout expected by the MAIA solver.

    Args:
        environment_name: Name of the environment (e.g. ``'Cylinder_2D_Re200'``).
        work_dir: Working directory path.  Auto-generated if ``None``.
        hf_repo_id: Hugging Face repository ID.
        force_download: Force re-download from HF Hub.
        **kwargs: Additional keyword arguments forwarded to :class:`MaiaWorkspace`.

    Returns:
        ``(work_dir_path, properties_file_path)`` for use with ``mpirun``.

    Example::

        import hydrogym.maia as maia

        work_dir, props_file = maia.prepare_maia_workspace('Cylinder_2D_Re200')
        # Then reference work_dir / props_file in your SLURM job script.
    """
    workspace = MaiaWorkspace(
        environment_name=environment_name,
        work_dir=work_dir,
        hf_repo_id=hf_repo_id,
        **kwargs
    )
    paths = workspace.setup(force_download=force_download, verbose=True)
    return paths['work_dir'], paths['properties_file']
