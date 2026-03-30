"""
HF Data Manager with Symbolic Links
====================================

This version supports three approaches:
- use_clean_cache=True: Create symlinks to HF cache (clean paths, no duplication!)
- use_clean_cache='copy': Copy to ~/.cache/maiagym/ (clean paths, duplicate storage)
- use_clean_cache=False: Use HF cache directly (ugly paths, no duplication)

Usage:
    # Symlinks (recommended - best of both worlds!)
    data_manager = HFDataManager(use_clean_cache=True)

    # Copy files (if you need to modify them)
    data_manager = HFDataManager(use_clean_cache='copy')

    # Use HF cache directly
    data_manager = HFDataManager(use_clean_cache=False)
"""

import os
import shutil
from typing import Optional, List, Union
import logging

try:
    from huggingface_hub import snapshot_download, HfApi

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")


class HFDataManager:
    """
    Manages CFD environment data from Hugging Face Hub.

    Can use HF cache directly, create symlinks, or copy to a clean cache structure.
    """

    def __init__(
        self,
        repo_id: str = "dynamicslab/HydroGym-environments",
        cache_dir: Optional[str] = None,
        local_fallback_dir: Optional[str] = None,
        use_clean_cache: Union[bool, str] = True,
    ):
        """
        Initialize the HF Data Manager.

        Args:
            repo_id: Hugging Face repository ID
            cache_dir: Clean local cache directory (default: ~/.cache/maiagym)
            local_fallback_dir: Local directory with environment files as fallback
            use_clean_cache:
                - True: Create symlinks to HF cache (recommended)
                - 'copy': Copy files to clean cache
                - False: Use HF cache directly
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/maiagym")
        self.local_fallback_dir = local_fallback_dir
        self.use_clean_cache = use_clean_cache

        # Ensure cache directory exists (only needed if using clean cache)
        if self.use_clean_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if not HF_AVAILABLE:
            self.logger.warning("Hugging Face Hub not available. Using local fallback only.")

    def get_available_environments(self) -> List[str]:
        """
        Get list of available environments from HF Hub or local fallback.

        Returns:
            List of environment names
        """
        if HF_AVAILABLE:
            try:
                api = HfApi()
                repo_files = api.list_repo_files(self.repo_id, repo_type="dataset")

                # Extract environment names from file paths
                env_names = set()
                for file_path in repo_files:
                    if "/" in file_path:
                        env_name = file_path.split("/")[0]
                        env_names.add(env_name)

                return sorted(list(env_names))
            except Exception as e:
                self.logger.warning(f"Could not fetch from HF Hub: {e}")

        # Fallback to local directory
        if self.local_fallback_dir and os.path.exists(self.local_fallback_dir):
            return [
                d
                for d in os.listdir(self.local_fallback_dir)
                if os.path.isdir(os.path.join(self.local_fallback_dir, d))
            ]

        return []

    def get_environment_path(self, env_name: str, force_download: bool = False) -> str:
        """
        Get path to environment files.

        Args:
            env_name: Environment name (e.g., 'Cylinder_2D_Re200')
            force_download: Force re-download even if cached

        Returns:
            Path to environment directory
        """
        if self.use_clean_cache == "copy":
            return self._get_environment_path_with_copy(env_name, force_download)
        elif self.use_clean_cache is True:
            return self._get_environment_path_with_symlink(env_name, force_download)
        else:
            return self._get_environment_path_direct(env_name, force_download)

    def _get_environment_path_with_symlink(self, env_name: str, force_download: bool) -> str:
        """Get environment path using symlinks (best option!)."""
        # This is our clean cache location (will be a symlink)
        local_env_link = os.path.join(self.cache_dir, env_name)

        # Check if symlink already exists and points to valid location
        if os.path.islink(local_env_link) and not force_download:
            target = os.readlink(local_env_link)
            if os.path.exists(target) and self._validate_environment_files(target):
                print(f"Using cached environment (symlink): {local_env_link} -> {target}")
                return local_env_link
            else:
                print("Symlink target invalid, re-downloading...")
                os.unlink(local_env_link)
        elif os.path.exists(local_env_link):
            # It's a regular directory (maybe from old copy method), remove it
            print("Replacing old cache with symlink...")
            if os.path.isdir(local_env_link):
                shutil.rmtree(local_env_link)
            else:
                os.remove(local_env_link)

        # Try HF Hub first
        if HF_AVAILABLE:
            try:
                print(f"Downloading {env_name} from Hugging Face Hub...")

                # Download to HF cache
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )

                # The environment files are at: hf_cache_path/env_name/
                hf_env_path = os.path.join(hf_cache_path, env_name)

                if os.path.exists(hf_env_path) and os.path.isdir(hf_env_path):
                    # OPTION 1: Use realpath to resolve all symlinks
                    # This gets the actual location of files
                    real_path = os.path.realpath(hf_env_path)
                    print(f"Creating symlink: {local_env_link} -> {real_path}")
                    os.symlink(real_path, local_env_link)

                    if self._validate_environment_files(local_env_link):
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

        # Fallback to local directory (use symlink if possible, copy if needed)
        if self.local_fallback_dir:
            local_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_path):
                print(f"Using local fallback: {local_path}")
                try:
                    os.symlink(local_path, local_env_link)
                    return local_env_link
                except OSError:
                    # Symlink failed, copy instead
                    shutil.copytree(local_path, local_env_link)
                    return local_env_link

        raise FileNotFoundError(f"Environment {env_name} not found in HF Hub or local fallback")

    def _get_environment_path_with_copy(self, env_name: str, force_download: bool) -> str:
        """Get environment path using clean cache (with copying)."""
        # This is our clean cache location
        local_env_dir = os.path.join(self.cache_dir, env_name)

        # Check if already in our clean cache and not forcing download
        if os.path.exists(local_env_dir) and not force_download:
            # If it's a symlink, remove it and copy instead
            if os.path.islink(local_env_dir):
                print("Converting symlink to copy...")
                target = os.readlink(local_env_dir)
                os.unlink(local_env_dir)
                if os.path.exists(target):
                    shutil.copytree(target, local_env_dir)
                    return local_env_dir
            elif self._validate_environment_files(local_env_dir):
                print(f"Using cached environment: {local_env_dir}")
                return local_env_dir
            else:
                print("Cached environment incomplete, re-downloading...")
                shutil.rmtree(local_env_dir)

        # Try HF Hub first
        if HF_AVAILABLE:
            try:
                print(f"Downloading {env_name} from Hugging Face Hub...")

                # Download to HF cache
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )

                # The environment files are at: hf_cache_path/env_name/
                hf_env_path = os.path.join(hf_cache_path, env_name)

                if os.path.exists(hf_env_path) and os.path.isdir(hf_env_path):
                    # Copy from HF cache to our clean cache
                    print(f"Copying to clean cache: {local_env_dir}")

                    if os.path.exists(local_env_dir):
                        shutil.rmtree(local_env_dir)

                    shutil.copytree(hf_env_path, local_env_dir)

                    if self._validate_environment_files(local_env_dir):
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

        # Fallback to local directory
        if self.local_fallback_dir:
            local_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_path):
                print(f"Using local fallback: {local_path}")
                if os.path.exists(local_env_dir):
                    shutil.rmtree(local_env_dir)
                shutil.copytree(local_path, local_env_dir)
                return local_env_dir

        raise FileNotFoundError(f"Environment {env_name} not found in HF Hub or local fallback")

    def _get_environment_path_direct(self, env_name: str, force_download: bool) -> str:
        """Get environment path using HF cache directly (no copying)."""
        # Try HF Hub
        if HF_AVAILABLE:
            try:
                print(f"Getting {env_name} from Hugging Face Hub...")

                # Download to HF cache (or use existing)
                hf_cache_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=f"{env_name}/**",
                    force_download=force_download,
                )

                # The environment files are at: hf_cache_path/env_name/
                env_path = os.path.join(hf_cache_path, env_name)

                if os.path.exists(env_path) and self._validate_environment_files(env_path):
                    print(f"Using environment at: {env_path}")
                    return env_path
                else:
                    raise FileNotFoundError(f"Environment path not found or invalid: {env_path}")

            except Exception as e:
                self.logger.error(f"Failed to get from HF Hub: {e}")
                import traceback

                self.logger.error(traceback.format_exc())

        # Fallback to local directory
        if self.local_fallback_dir:
            local_path = os.path.join(self.local_fallback_dir, env_name)
            if os.path.exists(local_path):
                print(f"Using local fallback: {local_path}")
                return local_path

        raise FileNotFoundError(f"Environment {env_name} not found in HF Hub or local fallback")

    def _validate_environment_files(self, env_dir: str) -> bool:
        """
        Validate that all required files exist in the environment directory.

        Args:
            env_dir: Path to environment directory (can be a symlink)

        Returns:
            True if all required files exist
        """
        required_files = [
            "geometry.toml",
            "properties_grid.toml",
            "properties_run.toml",
            "grid.Netcdf",
            "restart_.Netcdf",
        ]

        for file_name in required_files:
            file_path = os.path.join(env_dir, file_name)
            if not os.path.exists(file_path):
                self.logger.warning(f"Missing required file: {file_path}")
                return False

        # Check for STL directory
        stl_dir = os.path.join(env_dir, "stl")
        if not os.path.exists(stl_dir) or not os.listdir(stl_dir):
            self.logger.warning(f"Missing or empty STL directory: {stl_dir}")
            return False

        # Check for optional baseline file (for baseline reward strategy)
        baseline_file = os.path.join(env_dir, "baseline_state.feather")
        if os.path.exists(baseline_file):
            self.logger.info("Found baseline_state.feather for baseline reward strategy")
        else:
            self.logger.info("Note: baseline_state.feather not found (optional - using running_mean strategy)")

        return True

    # Keep download_environment as an alias for compatibility
    def download_environment(self, env_name: str, force_download: bool = False) -> str:
        """Alias for get_environment_path (for backwards compatibility)"""
        return self.get_environment_path(env_name, force_download)

    def clear_cache(self, env_name: Optional[str] = None):
        """
        Clear cached environment files.

        Args:
            env_name: Specific environment to clear, or None for all
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage MaiaGym environment data with Hugging Face Hub")
    parser.add_argument("--list", action="store_true", help="List available environments")
    parser.add_argument("--get", help="Get path to specific environment")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "direct"],
        default="symlink",
        help="Cache mode: symlink (default), copy, or direct",
    )

    args = parser.parse_args()

    # Convert mode to use_clean_cache value
    if args.mode == "symlink":
        use_clean_cache = True
    elif args.mode == "copy":
        use_clean_cache = "copy"
    else:
        use_clean_cache = False

    if args.list:
        manager = HFDataManager(use_clean_cache=use_clean_cache)
        envs = manager.get_available_environments()
        print("Available environments:")
        for env in envs:
            print(f"  - {env}")
    elif args.get:
        manager = HFDataManager(use_clean_cache=use_clean_cache)
        path = manager.get_environment_path(args.get)
        print(f"Environment available at: {path}")
        if os.path.islink(path):
            print(f"  (symlink to: {os.readlink(path)})")
    else:
        print("Use --help for available commands")
