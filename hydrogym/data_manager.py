"""
HydroGym Data Manager
=====================

General Hugging Face data manager for all HydroGym solvers.

Supports three caching strategies:
- use_clean_cache=True:    Create symlinks into HF cache (recommended — clean paths, no duplication)
- use_clean_cache='copy':  Copy to ~/.cache/hydrogym/ (clean paths, duplicated storage)
- use_clean_cache=False:   Use HF cache paths directly (no duplication, messy paths)

Local fallback optimization:
When local_fallback_dir is provided and contains the requested environment, it is used
directly without copying/linking to the cache (cache layer is only used for HF downloads)

Solver profiles
---------------
Each entry in :data:`SOLVER_PROFILES` describes one solver backend:

- ``sentinel``:        Hidden file stored alongside environment data on HF that identifies the
                       solver (e.g. ``.MAIA_LB``). Used for automatic profile detection.
- ``required_files``:  Files that must exist for validation to pass.
- ``required_dirs``:   Directories that must exist and be non-empty.
- ``optional_files``:  Files that are logged but not required.
- ``workspace_files``: ``{source_filename: target_rel_path}`` mapping used by
                       :meth:`HFDataManager.prepare_working_directory` to create
                       solver-specific symlinks in a run directory.
- ``workspace_dirs``:  Same idea for directories.

To add a new solver, add an entry here. No other file needs to change for
basic support (download + validation + workspace prep).

Usage::

    # Symlinks (recommended)
    dm = HFDataManager(use_clean_cache=True)

    # Copy files (if you need to modify them)
    dm = HFDataManager(use_clean_cache='copy')

    # Use HF cache directly
    dm = HFDataManager(use_clean_cache=False)

    # Get environment path (downloads if needed)
    env_path = dm.get_environment_path('Cylinder_2D_Re200')

    # Prepare a solver-specific working directory
    paths = dm.prepare_working_directory(env_path, './run_dir')
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

try:
    from huggingface_hub import snapshot_download, HfApi

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

# ---------------------------------------------------------------------------
# Solver profiles
# ---------------------------------------------------------------------------

SOLVER_PROFILES: Dict[str, dict] = {
    "MAIA_LB": {
        "sentinel": ".MAIA_LB",
        "required_files": [
            "geometry.toml",
            "properties_grid.toml",
            "properties_run.toml",
            "grid.Netcdf",
            "restart_.Netcdf",
        ],
        "required_dirs": ["stl"],
        "optional_files": ["baseline_state.feather"],
        # Workspace symlink map: source filename → relative target path inside run dir
        "workspace_files": {
            "grid.Netcdf": "out_lb/grid.Netcdf",
            "restart_.Netcdf": "out_lb/restart_.Netcdf",
            "properties_run.toml": "properties_run.toml",
            "geometry.toml": "geometry.toml",
            "environment_config.yaml": "environment_config.yaml",
        },
        "workspace_dirs": {
            "stl": "stl",
        },
    },
    "MAIA_STRCTRD": {
        "sentinel": ".MAIA_STRCTRD",
        "required_files": [
            "properties_run.toml",
            "grid.hdf5",
            "restart_.hdf5",
        ],
        "required_dirs": [],
        "optional_files": ["environment_config.yaml"],
        "workspace_files": {
            "grid.hdf5": "grid.hdf5",
            "restart_.hdf5": "out/restart_.hdf5",  # ← Goes in out/ subdirectory
            "properties_run.toml": "properties_run.toml",
            "environment_config.yaml": "environment_config.yaml",
        },
        "workspace_dirs": {},
    },
    # Profiles for future solvers — fill in required_files/workspace_files when ready
    "JAX": {
        "sentinel": ".JAX",
        "required_files": [],
        "required_dirs": [],
        "optional_files": [],
        "workspace_files": {},
        "workspace_dirs": {},
    },
    "JAXFLUIDS": {
        "sentinel": ".JAXFLUIDS",
        "required_files": [],
        "required_dirs": [],
        "optional_files": [],
        "workspace_files": {},
        "workspace_dirs": {},
    },
    "NEK5000": {
        "sentinel": ".NEK5000",
        # Runtime-only files (assumes nek5000 executable is pre-compiled)
        "required_files": [
            "environment_config.yaml",  # Environment configuration (REQUIRED for MAIA pattern)
            "phill.re2",  # Mesh file (read at runtime)
            "phill.ma2",  # Material properties (read at runtime)
            "phill.par",  # Parameter file (read at runtime, overridable)
            "int_pos",  # Time series probe positions (required for TSRS module)
        ],
        "required_dirs": [
            "restart_files",  # Initial condition files (read at startup)
        ],
        "optional_files": [
            "config.yaml",  # Alternative config name (legacy support)
            "README.md",  # Documentation
        ],
        # Workspace: runtime files symlinked to run directory
        "workspace_files": {
            "phill.re2": "phill.re2",
            "phill.ma2": "phill.ma2",
            "phill.par": "phill.par",
            "int_pos": "int_pos",
            "environment_config.yaml": "environment_config.yaml",
        },
        "workspace_dirs": {
            "restart_files": "restart_files",  # Link restart files directly
        },
    },
    "FIREDRAKE": {
        "sentinel": ".FIREDRAKE",
        # Checkpoint files can be either .h5 or .ckpt - validation happens in code
        "required_files": [],
        "required_dirs": [],
        "optional_files": ["environment_config.yaml", "README.md"],
        # No workspace preparation needed - Firedrake uses paths directly
        "workspace_files": {},
        "workspace_dirs": {},
    },
}

# Reverse map: sentinel filename → profile name (built once at import time)
_SENTINEL_TO_PROFILE: Dict[str, str] = {v["sentinel"]: k for k, v in SOLVER_PROFILES.items()}

# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------


class HFDataManager:
    """
    Manages CFD environment data from Hugging Face Hub.

    Handles downloading, caching (via symlinks, copies, or direct HF paths),
    solver-profile detection, validation, and working-directory preparation.

    Solver profiles are detected automatically from sentinel files (e.g. ``.MAIA_LB``,
    ``.MAIA_STRCTRD``) stored alongside environment data on HF. The local cache is
    checked first; if no sentinel is found the HF file listing is queried (lightweight
    — no full download). ``fallback_profile`` is used when neither source yields a
    result (e.g. legacy environments or offline mode).
    """

    def __init__(
        self,
        repo_id: str = "dynamicslab/HydroGym-environments",
        cache_dir: Optional[str] = None,
        local_fallback_dir: Optional[str] = None,
        use_clean_cache: Union[bool, str] = True,
        fallback_profile: str = "MAIA_LB",
    ):
        """
        Initialize the HF Data Manager.

        Args:
            repo_id: Hugging Face repository ID.
            cache_dir: Clean local cache directory (default: ``~/.cache/hydrogym``).
                Only used for HF downloads; local_fallback_dir is used directly.
            local_fallback_dir: Local directory with environment files used when HF
                is unreachable. When available, used directly without cache layer.
            use_clean_cache:
                - ``True``:    Create symlinks into HF cache (recommended).
                - ``'copy'``:  Copy files to clean cache.
                - ``False``:   Use HF cache paths directly.
                Note: cache_dir is only used for HF downloads, not for local_fallback.
            fallback_profile: Solver profile to use when no sentinel file is found.
                Defaults to ``'MAIA_LB'``.  Pass the environment class's
                ``SOLVER_TYPE`` attribute to get the right fallback in offline /
                legacy scenarios.
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/hydrogym")
        self.local_fallback_dir = local_fallback_dir
        self.use_clean_cache = use_clean_cache
        self.fallback_profile = fallback_profile

        if self.use_clean_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if not HF_AVAILABLE:
            self.logger.warning("Hugging Face Hub not available. Using local fallback only.")

    # ------------------------------------------------------------------
    # Solver profile detection
    # ------------------------------------------------------------------

    def _detect_solver_profile(self, env_name: str) -> str:
        """
        Detect the solver profile for *env_name* from sentinel files.

        Resolution order:

        1. Local clean cache (``~/.cache/hydrogym/{env_name}/``) — instant, no network.
        2. ``local_fallback_dir`` — offline alternative when HF is unreachable.
        3. HF repo file listing via ``list_repo_files`` — lightweight, no download,
           requires network.
        4. ``self.fallback_profile`` — last resort (offline / legacy / no sentinel).

        Args:
            env_name: Environment folder name (e.g. ``'Cylinder_2D_Re200'``).

        Returns:
            One of the keys in :data:`SOLVER_PROFILES`.
        """
        # 1. Check local clean cache (fast, no network)
        local_env_dir = os.path.join(self.cache_dir, env_name)
        if os.path.exists(local_env_dir):
            for sentinel, profile_name in _SENTINEL_TO_PROFILE.items():
                if os.path.exists(os.path.join(local_env_dir, sentinel)):
                    self.logger.info(f"Detected solver profile '{profile_name}' from local cache sentinel '{sentinel}'")
                    return profile_name

        # 2. Check local fallback dir (offline alternative, no network)
        if self.local_fallback_dir:
            fallback_env_dir = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(fallback_env_dir):
                for sentinel, profile_name in _SENTINEL_TO_PROFILE.items():
                    if os.path.exists(os.path.join(fallback_env_dir, sentinel)):
                        self.logger.info(
                            f"Detected solver profile '{profile_name}' from local fallback sentinel '{sentinel}'"
                        )
                        return profile_name

        # 3. Query HF file listing (no download, requires network)
        if HF_AVAILABLE:
            try:
                api = HfApi()
                repo_files = api.list_repo_files(self.repo_id, repo_type="dataset")
                for file_path in repo_files:
                    parts = file_path.split("/")
                    if len(parts) == 2 and parts[0] == env_name and parts[1] in _SENTINEL_TO_PROFILE:
                        profile_name = _SENTINEL_TO_PROFILE[parts[1]]
                        self.logger.info(f"Detected solver profile '{profile_name}' from HF sentinel '{parts[1]}'")
                        return profile_name
            except Exception as e:
                self.logger.warning(f"Could not query HF for solver profile: {e}")

        # 4. Last resort
        self.logger.info(f"No sentinel found for '{env_name}', using fallback profile: '{self.fallback_profile}'")
        return self.fallback_profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_environments(self) -> List[str]:
        """
        Get list of available environments from HF Hub or local fallback.

        Returns:
            Sorted list of environment names.
        """
        if HF_AVAILABLE:
            try:
                api = HfApi()
                repo_files = api.list_repo_files(self.repo_id, repo_type="dataset")

                env_names = set()
                for file_path in repo_files:
                    if "/" in file_path:
                        env_names.add(file_path.split("/")[0])

                return sorted(env_names)
            except Exception as e:
                self.logger.warning(f"Could not fetch from HF Hub: {e}")

        if self.local_fallback_dir and os.path.exists(self.local_fallback_dir):
            return [
                d
                for d in os.listdir(self.local_fallback_dir)
                if os.path.isdir(os.path.join(self.local_fallback_dir, d))
            ]

        return []

    def get_environment_path(self, env_name: str, force_download: bool = False) -> str:
        """
        Get path to environment files, downloading from HF Hub if necessary.

        The solver profile is detected automatically from sentinel files before
        any download occurs.

        Args:
            env_name: Environment name (e.g. ``'Cylinder_2D_Re200'``).
            force_download: Force re-download even if cached.

        Returns:
            Path to the local environment directory.
        """
        profile = self._detect_solver_profile(env_name)

        if self.use_clean_cache == "copy":
            return self._get_environment_path_with_copy(env_name, force_download, profile)
        elif self.use_clean_cache is True:
            return self._get_environment_path_with_symlink(env_name, force_download, profile)
        else:
            return self._get_environment_path_direct(env_name, force_download, profile)

    def prepare_working_directory(self, env_path: str, work_dir: str, profile: Optional[str] = None) -> Dict[str, str]:
        """
        Create *work_dir* and populate it with solver-specific symlinks.

        The mapping of source files/directories to their target locations inside
        *work_dir* is defined by the ``workspace_files`` and ``workspace_dirs``
        entries in :data:`SOLVER_PROFILES`.

        If *profile* is ``None`` the profile is auto-detected from sentinel files
        already present in *env_path*, then falls back to ``self.fallback_profile``.

        Args:
            env_path: Path to the cached environment data (as returned by
                :meth:`get_environment_path`).
            work_dir: Target working directory (created if it does not exist).
            profile: Solver profile key (e.g. ``'MAIA_LB'``).  Auto-detected when
                ``None``.

        Returns:
            Dictionary with at least ``'work_dir'`` and ``'env_data_path'`` keys.
            Additional keys may be present depending on the profile (e.g.
            ``'properties_file'`` for MAIA_LB).
        """
        if profile is None:
            env_path_obj = Path(env_path)
            profile = self.fallback_profile
            for sentinel, pname in _SENTINEL_TO_PROFILE.items():
                if (env_path_obj / sentinel).exists():
                    profile = pname
                    break

        solver = SOLVER_PROFILES.get(profile, {})
        workspace_files = solver.get("workspace_files", {})
        workspace_dirs = solver.get("workspace_dirs", {})

        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {
            "work_dir": str(work_path.resolve()),
            "env_data_path": str(env_path),
            "solver_profile": profile,
        }

        for source_name, target_rel in workspace_files.items():
            source_path = Path(env_path) / source_name
            target_path = work_path / target_rel
            if source_path.exists():
                self._link_path(source_path, target_path)
                # Expose well-known paths in the return dict
                if source_name == "properties_run.toml":
                    paths["properties_file"] = str(target_path.resolve())
                elif source_name == "environment_config.yaml":
                    paths["config_file"] = str(target_path.resolve())
            else:
                self.logger.warning(f"Workspace source not found, skipping: {source_path}")

        for source_name, target_rel in workspace_dirs.items():
            source_path = Path(env_path) / source_name
            target_path = work_path / target_rel
            if source_path.exists():
                self._link_path(source_path, target_path)
            else:
                self.logger.warning(f"Workspace source dir not found, skipping: {source_path}")

        self.logger.info(f"Working directory ready at: {work_path.resolve()}")
        return paths

    # ------------------------------------------------------------------
    # Cache strategies (internal)
    # ------------------------------------------------------------------

    def _get_environment_path_with_symlink(self, env_name: str, force_download: bool, profile: str) -> str:
        """Get environment path using symlinks (best option!)."""
        # Check local fallback first - if available, use it directly (skip cache layer)
        if self.local_fallback_dir and not force_download:
            local_fallback_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_fallback_path) and self._validate_environment_files(local_fallback_path, profile):
                print(f"Using local fallback directly (skip cache): {local_fallback_path}")
                return local_fallback_path

        local_env_link = os.path.join(self.cache_dir, env_name)

        if os.path.islink(local_env_link) and not force_download:
            target = os.readlink(local_env_link)
            if os.path.exists(target) and self._validate_environment_files(target, profile):
                print(f"Using cached environment (symlink): {local_env_link} -> {target}")
                return local_env_link
            else:
                print("Symlink target invalid, re-downloading...")
                os.unlink(local_env_link)
        elif os.path.exists(local_env_link):
            print("Replacing old cache with symlink...")
            if os.path.isdir(local_env_link):
                shutil.rmtree(local_env_link)
            else:
                os.remove(local_env_link)

        if HF_AVAILABLE:
            try:
                print(f"Downloading {env_name} from Hugging Face Hub...")
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )
                hf_env_path = os.path.join(hf_cache_path, env_name)
                if os.path.exists(hf_env_path) and os.path.isdir(hf_env_path):
                    real_path = os.path.realpath(hf_env_path)
                    print(f"Creating symlink: {local_env_link} -> {real_path}")
                    os.symlink(real_path, local_env_link)
                    if self._validate_environment_files(local_env_link, profile):
                        print(f"Environment ready at: {local_env_link}")
                        return local_env_link
                    else:
                        os.unlink(local_env_link)
                        raise FileNotFoundError("Validation failed after creating symlink")
                else:
                    raise FileNotFoundError(f"Environment path not found in HF cache: {hf_env_path}")
            except Exception as e:
                self.logger.error(f"Failed to get from HF Hub: {e}")
                import traceback

                self.logger.error(traceback.format_exc())

        # Final fallback: local_fallback_dir (if HF download failed)
        if self.local_fallback_dir:
            local_fallback_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_fallback_path):
                print(f"Using local fallback: {local_fallback_path}")
                return local_fallback_path

        raise FileNotFoundError(f"Environment '{env_name}' not found in HF Hub or local fallback")

    def _get_environment_path_with_copy(self, env_name: str, force_download: bool, profile: str) -> str:
        """Get environment path using clean cache (with copying)."""
        # Check local fallback first - if available, use it directly (skip cache layer)
        if self.local_fallback_dir and not force_download:
            local_fallback_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_fallback_path) and self._validate_environment_files(local_fallback_path, profile):
                print(f"Using local fallback directly (skip cache): {local_fallback_path}")
                return local_fallback_path

        local_env_dir = os.path.join(self.cache_dir, env_name)

        if os.path.exists(local_env_dir) and not force_download:
            if os.path.islink(local_env_dir):
                print("Converting symlink to copy...")
                target = os.readlink(local_env_dir)
                os.unlink(local_env_dir)
                if os.path.exists(target):
                    shutil.copytree(target, local_env_dir)
                    return local_env_dir
            elif self._validate_environment_files(local_env_dir, profile):
                print(f"Using cached environment: {local_env_dir}")
                return local_env_dir
            else:
                print("Cached environment incomplete, re-downloading...")
                shutil.rmtree(local_env_dir)

        if HF_AVAILABLE:
            try:
                print(f"Downloading {env_name} from Hugging Face Hub...")
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )
                hf_env_path = os.path.join(hf_cache_path, env_name)
                if os.path.exists(hf_env_path) and os.path.isdir(hf_env_path):
                    print(f"Copying to clean cache: {local_env_dir}")
                    if os.path.exists(local_env_dir):
                        shutil.rmtree(local_env_dir)
                    shutil.copytree(hf_env_path, local_env_dir)
                    if self._validate_environment_files(local_env_dir, profile):
                        print(f"Environment ready at: {local_env_dir}")
                        return local_env_dir
                    else:
                        raise FileNotFoundError("Validation failed after copy")
                else:
                    raise FileNotFoundError(f"Environment path not found in HF cache: {hf_env_path}")
            except Exception as e:
                self.logger.error(f"Failed to get from HF Hub: {e}")
                import traceback

                self.logger.error(traceback.format_exc())

        # Final fallback: local_fallback_dir (if HF download failed)
        if self.local_fallback_dir:
            local_fallback_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_fallback_path):
                print(f"Using local fallback: {local_fallback_path}")
                return local_fallback_path

        raise FileNotFoundError(f"Environment '{env_name}' not found in HF Hub or local fallback")

    def _get_environment_path_direct(self, env_name: str, force_download: bool, profile: str) -> str:
        """Get environment path using HF cache directly (no local copying)."""
        if HF_AVAILABLE:
            try:
                print(f"Getting {env_name} from Hugging Face Hub...")
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )
                env_path = os.path.join(hf_cache_path, env_name)
                if os.path.exists(env_path) and self._validate_environment_files(env_path, profile):
                    print(f"Using environment at: {env_path}")
                    return env_path
                else:
                    raise FileNotFoundError(f"Environment path not found or invalid: {env_path}")
            except Exception as e:
                self.logger.error(f"Failed to get from HF Hub: {e}")
                import traceback

                self.logger.error(traceback.format_exc())

        if self.local_fallback_dir:
            local_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_path):
                print(f"Using local fallback: {local_path}")
                return local_path

        raise FileNotFoundError(f"Environment '{env_name}' not found in HF Hub or local fallback")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_environment_files(self, env_dir: str, profile: str = "MAIA_LB") -> bool:
        """
        Validate that all required files/directories exist for the given solver profile.

        Args:
            env_dir: Path to environment directory (may be a symlink).
            profile: Solver profile key from :data:`SOLVER_PROFILES`.

        Returns:
            ``True`` if all required files and directories are present.
        """
        solver = SOLVER_PROFILES.get(profile)
        if solver is None:
            self.logger.warning(f"Unknown solver profile '{profile}', skipping validation")
            return True

        if not solver["required_files"] and not solver["required_dirs"]:
            self.logger.info(f"Profile '{profile}' has no required files defined yet, skipping validation")
            return True

        for file_name in solver["required_files"]:
            file_path = os.path.join(env_dir, file_name)
            if not os.path.exists(file_path):
                self.logger.warning(f"Missing required file: {file_path}")
                return False

        for dir_name in solver["required_dirs"]:
            dir_path = os.path.join(env_dir, dir_name)
            if not os.path.exists(dir_path) or not os.listdir(dir_path):
                self.logger.warning(f"Missing or empty required directory: {dir_path}")
                return False

        for file_name in solver["optional_files"]:
            file_path = os.path.join(env_dir, file_name)
            if os.path.exists(file_path):
                self.logger.info(f"Found optional file: {file_name}")
            else:
                self.logger.info(f"Optional file not present: {file_name}")

        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _link_path(self, source: Path, target: Path) -> None:
        """Create a symlink at *target* pointing to the resolved *source* path."""
        abs_source = source.resolve()
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(abs_source)
        self.logger.info(f"Linked: {target} -> {abs_source}")

    def download_environment(self, env_name: str, force_download: bool = False) -> str:
        """Alias for :meth:`get_environment_path` (backwards compatibility)."""
        return self.get_environment_path(env_name, force_download)

    def clear_cache(self, env_name: Optional[str] = None):
        """
        Clear cached environment files.

        Args:
            env_name: Specific environment to clear, or ``None`` for all.
        """
        if not self.use_clean_cache:
            print("Note: HF cache is managed by huggingface_hub. Use 'huggingface-cli delete-cache' to clear.")
            return

        if env_name:
            env_path = os.path.join(self.cache_dir, env_name)
            if os.path.exists(env_path):
                if os.path.islink(env_path):
                    os.unlink(env_path)
                    print(f"Removed symlink for: {env_name}")
                elif os.path.isdir(env_path):
                    shutil.rmtree(env_path)
                    print(f"Cleared cache for: {env_name}")
        else:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                print("Cleared all cached environments")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage HydroGym environment data with Hugging Face Hub")

    parser.add_argument("--list", action="store_true", help="List available environments")
    parser.add_argument("--get", help="Get path to specific environment")
    parser.add_argument("--prepare", metavar="WORK_DIR", help="Prepare a working directory for --get environment")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "direct"],
        default="symlink",
        help="Cache mode: symlink (default), copy, or direct",
    )
    parser.add_argument(
        "--profile",
        choices=list(SOLVER_PROFILES),
        default=None,
        help="Solver profile override (auto-detected by default)",
    )

    args = parser.parse_args()

    use_clean_cache = {"symlink": True, "copy": "copy", "direct": False}[args.mode]

    manager = HFDataManager(use_clean_cache=use_clean_cache)

    if args.list:
        envs = manager.get_available_environments()
        print("Available environments:")
        for env in envs:
            print(f"  - {env}")
    elif args.get:
        path = manager.get_environment_path(args.get)
        print(f"Environment available at: {path}")
        if os.path.islink(path):
            print(f"  (symlink to: {os.readlink(path)})")
        if args.prepare:
            paths = manager.prepare_working_directory(path, args.prepare, profile=args.profile)
            print(f"Working directory ready at: {paths['work_dir']}")
    else:
        print("Use --help for available commands")
