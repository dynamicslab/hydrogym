---
sidebar_position: 2
---

# Adding a New Solver

This guide covers the largest kind of extension: wiring up a **new solver
backend** (a new CFD/physics package behind HydroGym's environment API).
If you instead want to add a *new flow case or environment* on top of an
existing backend, see [Adding a New Environment](adding-an-environment.md)
— it is a much smaller task.

HydroGym deliberately supports **three solver-integration patterns**, and
the first decision you must make is which one fits your solver:

| Pattern | When to use it | Example backend |
|---|---|---|
| **In-process** | Your solver is importable as a Python library and its state can be represented as mutable Python objects | `firedrake` |
| **External process** | Your solver is a compiled binary that must run as its own MPI application | `maia`, `nek` |
| **Functional / JIT** | Your solver is written in JAX (or another array library) and is stepping is `jit`-compilable, state-in/state-out | `jax`, `jaxfluids` |

Each pattern has a minimal, copyable reference skeleton under
`examples/developer_templates/`:

- `minimal_inprocess_solver/` — the in-process pattern
- `minimal_external_solver/` — the external-process pattern

A new contributor can `diff` their work-in-progress backend against a
working minimal example instead of reverse-engineering a production
backend. Both skeletons ship with tests that run in the plain CI runner
(no GPU, no MPI ranks beyond what `mpi4py` provides serially, no solver
binaries).

---

## Pattern 1: In-process solver

This is the lightest integration. HydroGym provides the whole
gymnasium-facing machinery; you only implement the physics interface.

### Step 1 — Subclass `hydrogym.core.PDEBase`

`PDEBase` describes *what the state of your problem is* — fields, boundary
conditions, actuators — not how to advance it. Implement the abstract
methods:

```python
import numpy as np
from hydrogym import core


class TrivialFlow(core.PDEBase):
    # Class-level defaults (all optional):
    MAX_CONTROL = np.inf      # bound on each control input
    DEFAULT_MESH = ""         # mesh name used if none is passed
    DEFAULT_DT = np.inf       # default time step
    TAU = 0.0                 # actuation smoothing timescale

    StateType = list          # whatever type your state has
    MeshType = object
    BCType = object

    def __init__(self, **config):
        # Consume your own config keys with .pop() BEFORE calling super(),
        # so unknown keys can be flagged:
        self.n_dofs = config.pop("n_dofs", 8)
        super().__init__(**config)  # warns about unrecognized keys

    @property
    def num_inputs(self) -> int:
        return 1  # length of the control vector

    @property
    def num_outputs(self) -> int:
        return self.n_dofs  # number of observed scalars

    def load_mesh(self, name):
        return object()  # or a real mesh object

    def initialize_state(self):
        self.q = [0.0] * self.n_dofs

    def init_bcs(self):
        return []  # no boundary conditions

    def get_observations(self):
        return np.array(self.q)

    def evaluate_objective(self, q=None):
        return sum(q ** 2 for q in self.q)  # scalar objective

    # Remaining abstract methods: copy_state, save_checkpoint,
    # load_checkpoint, render. See the skeleton for minimal
    # implementations of each.
```

You inherit working `set_state`, `reset`, `reset_controls`, `set_control`,
`advance_time`, and `dot` implementations. `PDEBase.__init__` handles the
shared `mesh`/`restart` config keys and **warns about unrecognized keys**,
so consume your own keys with `config.pop(...)` before calling
`super().__init__`.

### Step 2 — Subclass `hydrogym.core.TransientSolver`

```python
class TrivialSolver(core.TransientSolver):
    def step(self, iter, control=None, **kwargs):
        """Advance one time step. Return anything your callbacks need."""
        u = self.flow.control_state()   # smoothed actuator state
        # ... advance self.flow.q by dt here ...
        return self.flow
```

If your solver supports adjoints/gradient computation, also implement
`solve(...)` with the appropriate solver options; otherwise the base-class
behavior is a no-op loop that only advances time.

### Step 3 — Reuse `core.FlowEnv` unmodified

Do **not** write a new environment class. `FlowEnv` composes whatever
`PDEBase` and `TransientSolver` you hand it:

```python
env = core.FlowEnv(
    env_config={
        "flow": TrivialFlow,
        "flow_config": {"n_dofs": 8},
        "solver": TrivialSolver,
        "solver_config": {"dt": 0.01},
        "max_steps": 1000,
        "actuation_config": {"num_substeps": 1, "reward_aggregation": "mean"},
    }
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.1])
```

Use the canonical config names `num_substeps` and `reward_aggregation`
(the older `num_sim_substeps_per_actuation` /
`reward_aggreation_rule` spellings are accepted with a
`DeprecationWarning` but should not appear in new code).

### Step 4 — (Optional) Hugging-Face-backed data

If your environments ship meshes/checkpoints on the
[dynamicslab/HydroGym-environments](https://huggingface.co/datasets/dynamicslab/HydroGym-environments)
Hub dataset, mix in `hydrogym.hf_env_mixin.HFEnvConfigMixin` rather than
reimplementing download/cache/config-resolution logic (it is the single
shared implementation all HF-backed backends use):

```python
from hydrogym.hf_env_mixin import HFEnvConfigMixin, ConfigError

class MyFlow(HFEnvConfigMixin, core.PDEBase):
    HF_CACHE_NAMESPACE = "mygym"   # ~/.cache/mygym/<EnvName>
    SOLVER_TYPE = "MYSOLVER"       # key of HFDataManager.SOLVER_PROFILES
```

The mixin provides `_setup_environment_data()`,
`_resolve_configuration_file()`, and `_find_configuration_file()`.

### Step 5 — Register the environment

```python
import gymnasium as gym

gym.register(
    id="hydrogym/Trivial-v0",
    entry_point="my_package.envs:make",   # a thin make(**kwargs) -> Env factory
)
```

so users can discover it with `gym.make("hydrogym/Trivial-v0", ...)`.
Keep the entry point a thin factory; heavy imports belong in the module.

### Step 6 — Tests

Mirror the structure of `test/test_cyl.py`: an import smoke test, a
steady-state solve, one transient `step()`, and one gradient test if your
backend is differentiable. At minimum, a plain-CI test that constructs the
flow, solver, and env with no special dependencies.

---

## Pattern 2: External-process solver

Use this when the solver is a separate compiled application. The controller
(Python/RL side) and the solver run as **two applications in one MPMD MPI
world**:

```bash
mpirun -np 1 python controller.py ... : -np N ./your_solver
```

This is the single biggest "how do I even start" hurdle for new external
backends, so your backend should ship a workspace-preparation helper
mirroring `hydrogym.maia.prepare_maia_workspace` that stages config files,
meshes, and restart data into a run directory, and document the exact
launch command.

### Step 1 — Subclass `gym.Env` directly and mix in the external-process mixin

Do **not** subclass `core.FlowEnv`: it composes `self.flow`/`self.solver`
in-process objects, which has no natural fit for a solver behind an MPI
boundary. Instead:

```python
import gymnasium as gym
from hydrogym.core_external import ExternalProcessEnvMixin


class MySolverEnv(ExternalProcessEnvMixin, gym.Env):
    ...
```

`ExternalProcessEnvMixin` provides:

- **`_split_mpmd_comm(comm_world, nproc=None)`** — performs the
  controller↔solver communicator split. Override the class-level knobs if
  your protocol needs something other than the defaults:
  `CONTROLLER_RANK` (0), `INTERCOMM_TAG` (99),
  `MPI_SPLIT_LOG_PREFIX` ("[MPI_SPLIT] ").
- **A documented `close()` contract**: when the episode or process ends
  you *must* send a termination message over the live communicator and
  tear down your MPI resources — see `hydrogym/nek/env.py` and
  `hydrogym/maia/env_core.py` for the two existing implementations. A
  backend that exits without telling the solver side deadlocks the MPMD
  job.

The split strategies themselves live in `hydrogym.core_external`:

- `mpi_split(comm_world, nproc=None, ...)` — rank-color split (rank 0 =
  controller, ranks 1+ = workers) with an inter-communicator handshake,
  plus world-size validation against `nproc`.
- `split_comm_by_appnum(comm_world)` — APPNUM-based split for true MPMD
  with any controller rank count, discovering the remote application's
  root via group translation.

Use one of these (via `_split_mpmd_comm`) instead of hand-rolling the
world split a third time.

### Step 2 — Implement your solver-specific wire protocol

The observation/action wire format is the **sanctioned escape hatch**:
MAIA's LBM/FV field layout and Nek's SEM node layout are genuinely
different, so each backend owns its own MPI tag set and buffer layout.
Define your `COMMAND_TAGS`-style mapping and send/receive helpers (see
`hydrogym/maia/mpmd_interface.py` for a compact example).

### Step 3 — Hugging-Face-backed environment data

If environments are fetched from the Hub, use `HFEnvConfigMixin` exactly
as in the in-process pattern (Step 4 above) — that is what MAIA and Nek
do.

### Step 4 — Config schema and validation

Keep your backend's config schema in its own shape (dataclass, dict, or
OmegaConf — whatever fits), but: consume recognized keys with `.pop()`,
accept the canonical names (`restart`, `num_substeps`,
`reward_aggregation`) for overlapping concepts, deprecate old names with
warnings rather than breaking them, and warn on unrecognized keys.

### Step 5 — Launch helpers and documentation

Ship a `prepare_<solver>_workspace(...)` helper and document the MPMD
launch command in your examples' README. Without this, nobody can run
your backend.

### Step 6 — Register, then test

Register with `gymnasium.register()` as above. Test in two tiers:

1. **Plain CI tier (no MPI, no solver binary):** a
   `--collect-only`-style test of the config-parsing/validation path.
   This is the coverage external backends most often lack entirely.
2. **Dev-container/HPC tier (MPI + solver binary):** construct the env,
   run one `step()`, and `close()` against a real small case, gated behind
   an MPI marker. See `test/mpmd_smoke_split.py` for a no-binary MPMD
   smoke of the communicator split itself.

---

## Pattern 3: Functional / JIT solver (JAX)

If your solver is a JAX program, do **not** force it into
`core.PDEBase`'s mutable-state shape. Instead subclass
`gymnax.environment.Environment[EnvState, EnvParams]` directly and expose
the standard functional `reset`/`step` API, exactly as
`hydrogym.jax` does. This is a documented third pattern, not a wart: the
whole point of keeping three contracts is that each solver style gets the
natural one. For HF-backed configuration reuse `HFEnvConfigMixin` as
above (the JAX backend does), and register with `gymnasium.register()`
via a thin factory if you want `gym.make()` discovery.

---

## Checklist

- [ ] Chosen the integration pattern (in-process / external-process / functional)
- [ ] Implemented the pattern's abstract surface (see skeleton in `examples/developer_templates/`)
- [ ] Canonical config names used; unknown keys warn; deprecated names shimmed
- [ ] `HFEnvConfigMixin` used instead of hand-rolled HF download logic (if HF-backed)
- [ ] `_split_mpmd_comm` used instead of a hand-rolled world split (if external-process)
- [ ] `close()` sends termination + frees MPI (if external-process)
- [ ] `prepare_<solver>_workspace` helper + documented launch command (if external-process)
- [ ] Registered with `gymnasium.register()`
- [ ] Tests: plain-CI tier at minimum; MPI/solver tier for external-process
