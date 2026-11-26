# JAX-Fluids Examples

Examples demonstrating HydroGym's JAX-Fluids backend for compressible,
fully-differentiable flow control (nozzle shock-vector control).

## Standard entry point: `gym.make()`

```bash
python gym_make_demo.py
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python gym_make_demo.py --env hydrogym-jaxfluids/Nozzle3D-v0 --steps 1
```

The 3D nozzle needs much more memory than the 2D one: `Nozzle3D_coarse` needs about 70 GB of GPU memory. JAX
reserves 75 % of the GPU memory by default, so on an 80-GB or 96-GB GPU (A100/H100 80 GB, GH200) set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. On the CPU it needs about 120 GB of host memory. `Nozzle2D` runs on any GPU
and on the CPU. With the demo's 4 injectors one action spans the whole `Nozzle3D_coarse` episode, so the first step
returns `terminated=True` (the demo then resets).

```python
import gymnasium as gym
import hydrogym.registration

env = gym.make("hydrogym-jaxfluids/Nozzle2D-v0")  # defaults to the "_coarse" resolution
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

There is no plain `"Nozzle2D"`/`"Nozzle3D"` environment on the Hugging
Face Hub — only resolution-suffixed variants
(`Nozzle2D_coarse`/`_fine`, `Nozzle3D_coarse`/`_fine`). `gym.make()`
defaults both registered ids to `_coarse`; pass a specific one via
`env_config={"environment_name": "Nozzle2D_fine"}`.

## Lower-level entry point

For direct construction (custom `environment_config.yaml`, or anything
`gym.make()`'s registered default doesn't cover), see `test_jaxfluids_env.py`:

```python
from hydrogym.jaxfluids.envs import Nozzle2D

env = Nozzle2D(env_config={"environment_name": "Nozzle2D_coarse"})
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Files

- `gym_make_demo.py` — the standard entry point, `gym.make()`
- `test_jaxfluids_env.py` — interactive test via the lower-level, direct construction path (1000 environment
  steps by default: about 14 h on a CPU for Nozzle2D_coarse)
- `environment_config.yaml` — example config reference

## Notes

- Requires internet access on first run to download environment data from
  the Hugging Face Hub (`dynamicslab/HydroGym-environments`).
- JAX-Fluids' `step()`/`reset()` return values come from the wrapped
  external `jaxfluids_rl.JAXFluidsEnv` package; gymnasium's passive env
  checker may warn about `float64` vs. the declared `float32` observation
  dtype and about `terminated`/`truncated` being JAX array scalars rather
  than plain Python `bool` — informational only, not fatal, and does not
  affect correctness of `obs`/`reward` values.
