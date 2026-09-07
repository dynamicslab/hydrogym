---
sidebar_position: 3
---

# Adding a New Environment

This guide covers the **common** extension case: adding a new flow case or
environment on top of a backend that already exists (a new Firedrake
`FlowConfig` subclass, a new MAIA environment registration, a new Nek
dataset, ...). Adding a whole new *solver backend* is a different, larger
task — see [Adding a New Solver](adding-a-solver.md).

Two things are shared by every backend:

1. **A Python environment class** registered with the backend's discovery
   mechanism (details per backend below).
2. **Data on the Hugging Face Hub** ([dynamicslab/HydroGym-environments](https://huggingface.co/datasets/dynamicslab/HydroGym-environments)):
   one top-level directory per environment, named
   `{Flow}_{dim}_{Re}[_{variant}]` (with a `_FD` suffix for the Firedrake
   environments), containing that environment's meshes/checkpoints/config
   files.

The backend's cache/download machinery (all of it consolidated in
`hydrogym.hf_env_mixin.HFEnvConfigMixin`) looks the environment up by that
name under `~/.cache/<backend-namespace>/<environment_name>/`, falling back
to a `HFDataManager` download. Auto-detected configuration files, in
priority order: `config.yaml`, `environment_config.yaml`, `env_config.yaml`,
`environment.yaml`, `<environment_name>.yaml`, then `config_*.yaml` globs.

---

## Firedrake (new `FlowConfig` subclass)

1. Create a package under `hydrogym/firedrake/envs/<name>/` containing:

   - `flow.py` — your `FlowConfig` subclass:

     ```python
     from hydrogym.firedrake import FlowConfig

     class MyCase(FlowConfig):
         DEFAULT_MESH = "mesh.msh"       # shipped alongside, loaded via load_mesh
         DEFAULT_DT = 1e-4
         DEFAULT_STABILIZATION = "upwind"

         def __init__(self, **config):
             super().__init__(**config)
             # your geometry/boundary-condition setup

         def get_observations(self):
             ...

         def evaluate_objective(self, q=None):
             ...
     ```

   - the mesh files themselves (`*.msh`/`*.geo`), and
   - an `__init__.py` re-exporting the class, e.g.
     `from .flow import MyCase`.

2. Make the class importable where users construct it: `hydrogym.firedrake.envs.__init__`
   re-exports the built-ins explicitly (e.g. `from .cylinder import
   Cylinder, RotaryCylinder`), so add your class there (and to its
   `__all__`). Discovery is explicit, not automatic.

3. Add a test under `test/` mirroring `test/test_cyl.py`: import smoke,
   steady solve, one transient step. Firedrake tests run in the
   Firedrake dev container (see `test/README.md`).

4. Upload the environment's data to the Hub under `MyCase_..._FD/`
   (Firedrake environments carry the `_FD` suffix) with any
   `checkpoint_*.h5`/`mesh.msh` files and, if the case needs one, a
   `config.yaml`.

## MAIA (new `envs/*.py` registration)

MAIA resolves environments **by name prefix** at runtime: the part of the
environment name before the first underscore (`Cylinder` in
`Cylinder_2D_Re200`) is looked up in an in-memory registry.

1. Create `hydrogym/maia/envs/<name>.py`:

   ```python
   from typing import Dict, Tuple

   from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


   class MyCase(MaiaFlowEnv):
       def __init__(self, env_config: Dict):
           super().__init__(env_config)

       def get_reward(self) -> Tuple[float, Dict]:
           ...  # your reward from the solver observations


   # The prefix users will write in the environment name, e.g.
   # MyCase_2D_Re500:
   register_environment("MyCase", MyCase)
   ```

2. Import the module in `hydrogym/maia/envs/__init__.py` so registration
   happens at import time (the registry is populated on import; the
   `from_hf` error message will remind users of the available types).

3. Users then construct it with the prefix-derived name:

   ```python
   import hydrogym.maia as maiaGym
   env = maiaGym.from_hf("MyCase_2D_Re500", nproc=4)
   ```

4. Upload data to the Hub under `MyCase_2D_Re500/` with the solver
   property file(s) and restart data. A MAIA environment needs a
   configuration file (auto-detected by name as listed above) and the
   restart checkpoint(s) referenced by it.

## JAX and JAX-Fluids (new env subclass)

- **JAX**: subclass `hydrogym.jax.envs`' base environment (see
  `channel.py`/`kolmogorov.py`) and implement the gymnax-style
  functional `reset`/`step`. These shipped envs are HF-free; if yours
  needs Hub data, build it on `JAXFlowEnv` (which mixes in
  `HFEnvConfigMixin`, namespace `jaxgym`) instead.
- **JAX-Fluids**: subclass `hydrogym.jaxfluids.env_core.JAXFluidsFlowEnv`
  (see `envs/nozzle.py`). Provide the environment data on the Hub
  including a `config.yaml` — that is the auto-detected configuration
  file for the JAX-Fluids solver setup.

## Nek (new dataset + config)

Nek environments are primarily data: a `NekEnv.from_hf("MyCase_...", nproc=N)`
looks up `~/.cache/nekgym/<name>` (or downloads from the Hub), symlinks
the case files, and expects an `environment_config.yaml` describing the
case (mesh/restart files, control points, observation locations). To add
one:

1. Assemble the case directory: `*.re2`/`*.ma2` mesh, `*.par` parameter
   file, `restart_files/`, and `environment_config.yaml`.
2. Upload it to the Hub under the environment's name.
3. Document the required `nproc` — the MPMD launch must be
   `mpirun -np 1 python ... : -np <nproc> ./nek5000` (world size
   `nproc + 1`), and the validation in `mpi_split` enforces it.

---

## Testing expectations by tier

| What you added | Minimum test tier |
|---|---|
| Config parsing / name resolution only | plain `pytest` (no backend deps) |
| Firedrake case | Firedrake dev container (`pytest test/`) |
| MAIA / Nek environment | dev-container MPMD run (env construction + one `step()` + `close()`); config-level checks in plain CI |
| JAX / JAX-Fluids env | CPU JAX tier; GPU smoke in the GPU dev container |

Add a plain-CI test whenever any part of your environment (name parsing,
config resolution, reward aggregation) can be exercised without the
solver — that is the coverage most easily lost to bit-rot.
