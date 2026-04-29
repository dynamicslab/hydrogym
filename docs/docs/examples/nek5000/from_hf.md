---
sidebar_position: 5
---

# Loading Environments from HuggingFace Hub

`NekEnv.from_hf()` is the recommended way to create a NEK5000 environment for most users. Rather than constructing an `env_config` dictionary manually, you provide only the environment name and the number of solver processes. The method automatically locates the configuration file, prepares the workspace, and returns a ready-to-use [`NekEnv`](../../api/nek/env) instance.

All scripts live in [`examples/nek/getting_started/4_from_hf/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/4_from_hf).

## Creating an environment

```python
from hydrogym.nek import NekEnv

env = NekEnv.from_hf(
    'TCFmini_3D_Re180',
    nproc=10,
    use_clean_cache=False,         # reuse an existing prepared workspace
    local_fallback_dir=None,       # optional: path to a local environment package
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

The `from_hf()` class method:
1. Searches `local_fallback_dir` first (if provided), then the HuggingFace Hub
2. Auto-detects `environment_config.yaml` or `config.yaml` inside the environment package
3. Prepares the workspace and writes the case files before launching the solver

Setting `use_clean_cache=False` reuses a previously prepared workspace, which is useful for iterative development or re-running evaluation scripts.

## Example scripts

| File | Purpose |
|------|---------|
| [`test_nek_DM.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/4_from_hf/test_nek_DM.py) | Smoke-test `from_hf()` loading with zero control |
| [`train_sb3_from_hf.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/4_from_hf/train_sb3_from_hf.py) | SB3 training starting from an HF-managed environment |
| [`run_from_hf_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/4_from_hf/run_from_hf_docker.sh) | Docker/MPI launcher |

## Running the examples

```bash
# Test
mpirun -np 1 python test_nek_DM.py --steps 100 : -np 10 nek5000

# Train
mpirun -np 1 python train_sb3_from_hf.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## Offline use on HPC systems

On clusters without internet access, download the environment package locally and pass the directory via `local_fallback_dir`. The method will resolve the package from the local path and skip any network requests:

```python
env = NekEnv.from_hf(
    'TCFmini_3D_Re180',
    nproc=10,
    local_fallback_dir='/scratch/my_user/nek_envs',
)
```

## Comparison with direct instantiation

| | `from_hf()` | Direct `NekEnv(env_config=...)` |
|---|---|---|
| Configuration required | Environment name + `nproc` | Full `env_config` dictionary |
| Config file detection | Automatic | Manual |
| Workspace preparation | Automatic | Manual |
| Recommended for | Most users | Advanced configuration |

For full control over every environment parameter, use direct instantiation as shown in the [Single Agent](./single_environment) examples. For everything else, `from_hf()` is the simpler choice.
