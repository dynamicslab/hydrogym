---
sidebar_label: data_manager
title: hydrogym.data_manager
---

HydroGym Data Manager
=====================

General Hugging Face data manager for all HydroGym solvers.

Supports three caching strategies:
- use_clean_cache=True:    Create symlinks into HF cache (recommended — clean paths, no duplication)
- use_clean_cache=&#x27;copy&#x27;:  Copy to ~/.cache/hydrogym/ (clean paths, duplicated storage)
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
:meth:``5 to create
solver-specific symlinks in a run directory.
- ``workspace_dirs``:  Same idea for directories.

To add a new solver, add an entry here. No other file needs to change for
basic support (download + validation + workspace prep).

Usage::

# Symlinks (recommended)
dm = HFDataManager(use_clean_cache=True)

# Copy files (if you need to modify them)
dm = HFDataManager(use_clean_cache=&#x27;copy&#x27;)

# Use HF cache directly
dm = HFDataManager(use_clean_cache=False)

# Get environment path (downloads if needed)
env_path = dm.get_environment_path(&#x27;Cylinder_2D_Re200&#x27;)

# Prepare a solver-specific working directory
paths = dm.prepare_working_directory(env_path, &#x27;./run_dir&#x27;)

## HFDataManager Objects

```python
class HFDataManager()
```

Manages CFD environment data from Hugging Face Hub.

Handles downloading, caching (via symlinks, copies, or direct HF paths),
solver-profile detection, validation, and working-directory preparation.

Solver profiles are detected automatically from sentinel files (e.g. ``.MAIA_LB``,
``.MAIA_STRCTRD``) stored alongside environment data on HF. The local cache is
checked first; if no sentinel is found the HF file listing is queried (lightweight
— no full download). ``fallback_profile`` is used when neither source yields a
result (e.g. legacy environments or offline mode).

#### \_\_init\_\_

```python
def __init__(repo_id: str = "dynamicslab/HydroGym-environments",
             cache_dir: Optional[str] = None,
             local_fallback_dir: Optional[str] = None,
             use_clean_cache: Union[bool, str] = True,
             fallback_profile: str = "MAIA_LB")
```

Initialize the HF Data Manager.

**Arguments**:

- `repo_id` - Hugging Face repository ID.
- `cache_dir` - Clean local cache directory (default: ``~/.cache/hydrogym``).
  Only used for HF downloads; local_fallback_dir is used directly.
- `local_fallback_dir` - Local directory with environment files used when HF
  is unreachable. When available, used directly without cache layer.
  use_clean_cache:
  - ``True``:    Create symlinks into HF cache (recommended).
  - ``&#x27;copy&#x27;``:  Copy files to clean cache.
  - ``False``:   Use HF cache paths directly.
- `cache_dir`1 - cache_dir is only used for HF downloads, not for local_fallback.
- `cache_dir`2 - Solver profile to use when no sentinel file is found.
  Defaults to ``&#x27;MAIA_LB&#x27;``.  Pass the environment class&#x27;s
  ``SOLVER_TYPE`` attribute to get the right fallback in offline /
  legacy scenarios.

#### get\_available\_environments

```python
def get_available_environments() -> List[str]
```

Get list of available environments from HF Hub or local fallback.

**Returns**:

  Sorted list of environment names.

#### get\_environment\_path

```python
def get_environment_path(env_name: str, force_download: bool = False) -> str
```

Get path to environment files, downloading from HF Hub if necessary.

The solver profile is detected automatically from sentinel files before
any download occurs.

**Arguments**:

- `env_name` - Environment name (e.g. ``&#x27;Cylinder_2D_Re200&#x27;``).
- `force_download` - Force re-download even if cached.
  

**Returns**:

  Path to the local environment directory.

#### prepare\_working\_directory

```python
def prepare_working_directory(env_path: str,
                              work_dir: str,
                              profile: Optional[str] = None) -> Dict[str, str]
```

Create *work_dir* and populate it with solver-specific symlinks.

The mapping of source files/directories to their target locations inside
*work_dir* is defined by the ``workspace_files`` and ``workspace_dirs``
entries in :data:`SOLVER_PROFILES`.

If *profile* is ``None`` the profile is auto-detected from sentinel files
already present in *env_path*, then falls back to ``self.fallback_profile``.

**Arguments**:

- `env_path` - Path to the cached environment data (as returned by
  :meth:``0).
- ``1 - Target working directory (created if it does not exist).
- ``2 - Solver profile key (e.g. ``&#x27;MAIA_LB&#x27;``).  Auto-detected when
  ``None``.
  

**Returns**:

  Dictionary with at least ``&#x27;work_dir&#x27;`` and ``&#x27;env_data_path&#x27;`` keys.
  Additional keys may be present depending on the profile (e.g.
  ``&#x27;properties_file&#x27;`` for MAIA_LB).

#### download\_environment

```python
def download_environment(env_name: str, force_download: bool = False) -> str
```

Alias for :meth:`get_environment_path` (backwards compatibility).

#### clear\_cache

```python
def clear_cache(env_name: Optional[str] = None)
```

Clear cached environment files.

**Arguments**:

- `env_name` - Specific environment to clear, or ``None`` for all.

