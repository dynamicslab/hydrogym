---
sidebar_label: hf_data_manager
title: hydrogym.maia.hf_data_manager
---

HF Data Manager with Symbolic Links

This version supports three approaches:
- use_clean_cache=True: Create symlinks to HF cache (clean paths, no duplication!)
- use_clean_cache=&#x27;copy&#x27;: Copy to ~/.cache/maiagym/ (clean paths, duplicate storage)
- use_clean_cache=False: Use HF cache directly (ugly paths, no duplication)

Usage:
# Symlinks (recommended - best of both worlds!)
data_manager = HFDataManager(use_clean_cache=True)

# Copy files (if you need to modify them)
data_manager = HFDataManager(use_clean_cache=&#x27;copy&#x27;)

# Use HF cache directly
data_manager = HFDataManager(use_clean_cache=False)

## HFDataManager Objects

```python
class HFDataManager()
```

Manages CFD environment data from Hugging Face Hub.

Can use HF cache directly, create symlinks, or copy to a clean cache structure.

#### \_\_init\_\_

```python
def __init__(repo_id: str = "dynamicslab/HydroGym-environments",
             cache_dir: Optional[str] = None,
             local_fallback_dir: Optional[str] = None,
             use_clean_cache: Union[bool, str] = True)
```

Initialize the HF Data Manager.

**Arguments**:

- `repo_id` - Hugging Face repository ID
- `cache_dir` - Clean local cache directory (default: ~/.cache/maiagym)
- `local_fallback_dir` - Local directory with environment files as fallback
  use_clean_cache:
  - True: Create symlinks to HF cache (recommended)
  - &#x27;copy&#x27;: Copy files to clean cache
  - False: Use HF cache directly

#### get\_available\_environments

```python
def get_available_environments() -> List[str]
```

Get list of available environments from HF Hub or local fallback.

**Returns**:

  List of environment names

#### get\_environment\_path

```python
def get_environment_path(env_name: str, force_download: bool = False) -> str
```

Get path to environment files.

**Arguments**:

- `env_name` - Environment name (e.g., &#x27;Cylinder_2D_Re200&#x27;)
- `force_download` - Force re-download even if cached
  

**Returns**:

  Path to environment directory

#### download\_environment

```python
def download_environment(env_name: str, force_download: bool = False) -> str
```

Alias for get_environment_path (for backwards compatibility)

#### clear\_cache

```python
def clear_cache(env_name: Optional[str] = None)
```

Clear cached environment files.

**Arguments**:

- `env_name` - Specific environment to clear, or None for all

