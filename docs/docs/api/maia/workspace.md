---
sidebar_label: workspace
title: hydrogym.maia.workspace
---

MAIA Workspace Setup
====================

This module provides utilities for setting up MAIA workspaces with proper
file structure for MPMD coupling. It handles downloading from HF Hub and
creating necessary symbolic links with correct file extensions.

## MaiaWorkspace Objects

```python
class MaiaWorkspace()
```

Manages MAIA workspace setup including file downloads and symbolic links.

File preparation is driven by solver profiles in
:data:`~hydrogym.data_manager.SOLVER_PROFILES`.  The correct profile
(``&#x27;MAIA_LB&#x27;`` or ``&#x27;MAIA_STRCTRD&#x27;``) is auto-detected from sentinel files.

#### \_\_init\_\_

```python
def __init__(environment_name: str,
             work_dir: Optional[str] = None,
             hf_repo_id: str = "dynamicslab/HydroGym-environments",
             local_fallback_dir: Optional[str] = None,
             use_clean_cache: bool = True,
             solver_type: Optional[str] = None)
```

Initialize the MAIA workspace.

**Arguments**:

- `environment_name` - Name of the environment (e.g., &#x27;Cylinder_2D_Re200&#x27;).
- `work_dir` - Working directory path. If None, uses &#x27;./run_`{environment_name}`&#x27;.
- `hf_repo_id` - Hugging Face repository ID.
- `local_fallback_dir` - Optional local fallback directory.
- `use_clean_cache` - Whether to use clean cache for HF downloads.
- `solver_type` - Solver profile key (``&#x27;MAIA_LB&#x27;`` or ``&#x27;MAIA_STRCTRD&#x27;``).
  Auto-detected from sentinel files if ``None`` (recommended).
  Defaults to ``&#x27;MAIA_LB&#x27;`` as fallback for legacy environments.

#### setup

```python
def setup(force_download: bool = False,
          verbose: bool = True) -> Dict[str, str]
```

Set up the workspace with all required files.

This method:
1. Downloads/locates environment data from HF Hub
2. Creates work directory structure
3. Creates symbolic links with proper names/extensions

**Arguments**:

- `force_download` - Force re-download from HF Hub.
- `verbose` - Print setup progress.
  

**Returns**:

  Dictionary containing paths to key files:
  - &#x27;work_dir&#x27;: Path to work directory
  - &#x27;properties_file&#x27;: Path to properties file for MAIA
  - &#x27;config_file&#x27;: Path to environment config file
  - &#x27;env_data_path&#x27;: Path to source environment data

#### cleanup

```python
def cleanup(remove_work_dir: bool = False, verbose: bool = True) -> None
```

Clean up the workspace.

**Arguments**:

- `remove_work_dir` - If True, removes the entire work directory.
  If False, only removes symbolic links.
- `verbose` - Print cleanup messages.

#### prepare\_maia\_workspace

```python
def prepare_maia_workspace(
        environment_name: str,
        work_dir: Optional[str] = None,
        hf_repo_id: str = "dynamicslab/HydroGym-environments",
        force_download: bool = False,
        **kwargs) -> Tuple[str, str]
```

Convenience function to prepare a MAIA workspace for MPMD coupling.

This function handles all the file preparation needed before launching
mpirun with Python and MAIA. It downloads data from HF Hub (if needed)
and creates symbolic links with proper file extensions.

**Arguments**:

- `environment_name` - Name of the environment (e.g., &#x27;Cylinder_2D_Re200&#x27;).
- `work_dir` - Working directory path. If None, auto-generates name.
- `hf_repo_id` - Hugging Face repository ID.
- `force_download` - Force re-download from HF Hub.
- `**kwargs` - Additional arguments passed to MaiaWorkspace.
  

**Returns**:

  Tuple of (work_dir_path, properties_file_path) for use with mpirun.
  

**Example**:

  &gt;&gt;&gt; import hydrogym.maia as maia
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Prepare workspace (before submitting HPC job)
  &gt;&gt;&gt; work_dir, props_file = maia.prepare_maia_workspace(&#x27;Cylinder_2D_Re200&#x27;)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Then use work_dir and props_file in your job script:
  &gt;&gt;&gt; # sbatch job.slurm  # where job.slurm references work_dir

