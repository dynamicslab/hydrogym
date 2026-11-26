---
sidebar_position: 1
---

# Getting Started

The JAX-Fluids backend provides the nozzle environments `Nozzle2D` and `Nozzle3D`: compressible flow through a nozzle
with shock-vector control by secondary injection. The installation is described on the
[JAX-Fluids installation page](../../installation/jaxfluids). The example scripts are in
[`examples/jaxfluids`](https://github.com/dynamicslab/hydrogym/tree/main/examples/jaxfluids).

| File | Description |
|---|---|
| [`gym_make_demo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/jaxfluids/gym_make_demo.py) | The standard entry point, `gym.make()` |
| [`test_jaxfluids_env.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/jaxfluids/test_jaxfluids_env.py) | Direct construction of an environment class and a roll-out with `env.step()` (1000 steps by default) |
| [`environment_config.yaml`](https://github.com/dynamicslab/hydrogym/blob/main/examples/jaxfluids/environment_config.yaml) | Reference of the environment configuration |

## `gym.make()`

```bash
python gym_make_demo.py
python gym_make_demo.py --env hydrogym-jaxfluids/Nozzle3D-v0 --steps 1
```

```python
import gymnasium as gym
import hydrogym.registration

env = gym.make("hydrogym-jaxfluids/Nozzle2D-v0")  # the "_coarse" resolution
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The Hugging Face Hub holds resolution-suffixed variants only (`Nozzle2D_coarse`, `Nozzle2D_fine`, `Nozzle3D_coarse`,
`Nozzle3D_fine`). `gym.make()` uses `_coarse` for both registered ids; pass another one with
`env_config={"environment_name": "Nozzle2D_fine"}`.

## Direct construction

For a custom `environment_config.yaml`, construct the environment class directly:

```python
from hydrogym.jaxfluids.envs import Nozzle2D

env = Nozzle2D(env_config={"environment_name": "Nozzle2D_coarse"})
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Requirements

- The first run downloads the environment data from the Hugging Face Hub (`dynamicslab/HydroGym-environments`).
- `Nozzle2D` runs on any GPU and on the CPU. `Nozzle3D_coarse` needs about 70 GB of GPU memory (JAX reserves 75 % of
  the GPU memory by default, so set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` on an 80-GB or 96-GB GPU), or about 120 GB
  of host memory on the CPU. With 4 injectors one action spans the whole `Nozzle3D_coarse` episode: run the demo with
  `--steps 1`.
- Gymnasium's environment checker may warn that the observations are `float64` (declared `float32`) and that
  `terminated` / `truncated` are JAX array scalars; the values are not affected.
