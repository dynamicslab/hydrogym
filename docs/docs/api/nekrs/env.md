---
sidebar_label: env
title: hydrogym.nekrs.env
---

YAML-driven Gymnasium environment for NekRS cases with the HydroGym control interface.

Everything the agent sees and does is defined in the YAML (see examples/nekrs/channel/channel_re207.yml and,
for the wall-unit options of a non-uniform wall, examples/nekrs/wing/wing.yml):


The solver only marks the boundaries that may be actuated ([CASEDATA] hydrogymBoundaries in the par file).
Every solver step is logged with its CFL number (solver_steps.csv), every action with the reward, the CFL range
and the actuation diagnostics (steps.jsonl).
```text
  environment:  nproc, steps_per_action (n or {dt_plus: value}), episode_actions, target_cfl, dt (optional:
                the solver time step of the par file the configuration is made for; checked at start-up)
  wall_units:   nu (default mu / rho of the solver), u_tau: value | {polynomial: [c0, c1, ...], axis: x}
                (needed by the u_tau / plus options below)
  actuators:    boundaries, positions (or an explicit `jets` list), footprint, direction,
                amplitude (value or {u_tau: factor}), node_box [xmin, xmax, ymin, ymax, zmin, zmax] (only these
                boundary nodes are actuated), period [px, py, pz] (periodic footprints, 0 = none),
                znmf (none | uniform | actions, solver side, or
                {group_by: [x, y], tol}: zero net flux on every group of actuators, e.g. per spanwise line),
                resolution_check {mode: error | warn | off, min_nodes_per_width}
  probes:       fields, positions (actuator_offset normal_distance may be {plus: y+}), subtract_plane_mean
                (solver side) or subtract_group_mean: [x, y] (mean over probes sharing x, y, weighted by the
                actuator footprint), frame: global | wall (u, v -> wall tangent u_t (t_x > 0) and inward normal
                v_n) | hg_fortran (u_t = u n_y + v n_x, not a rotation; v_n as for wall), normalization
                (value per field, or u_tau)
  reward:       type wall_shear {boundary, quantity: gradient | force, baseline_dudy, average: legacy | mean}
  monitors:     further [{boundary, quantity: force | gradient}] recorded at every solver step (solver_steps.csv)
  power:        actuator power (hydrogym.power interface, see hydrogym/nekrs/power.py); all keys optional:
                enabled (default true; false skips the wall requests), boundaries (default actuators.boundaries),
                pressure_mean (default [x, y]: p' about the mean over the nodes sharing x and y; [] = full p),
                reference {velocity, length, span} or {velocity, area} (C_P = P / (0.5 rho U^3 A); defaults
                U = 1 and A = area of the power boundaries), drag {boundaries, axis} (drag_coefficient from the
                force monitors, same reference), power_weight (0), power_cost (abs), power_sampling (mean: every
                solver step of the action, one step per request; end: after the last step)
  initial_conditions (episode reset):
                source: continue (default, the flow runs on) | snapshots | files
                snapshots {count, interval, warmup, save, load}: library of exact solver states, built with zero
                    actuation (warmup steps, then one snapshot every interval steps) or read from disk
                files [globs]: NekRS field files (u, p; the time-stepper history is rebuilt from the loaded state)
                interpolate, selection: random | cycle | first, settle_steps (zero-action steps after the reset)
```

#### load\_config

```python
def load_config(config: Union[str, Dict]) -> Dict
```

YAML file or dict; `base: other.yml` (next to the file, else relative to the cwd) is merged underneath.

## NekRSFlowEnv Objects

```python
class NekRSFlowEnv(gym.Env)
```

#### \_\_init\_\_

```python
def __init__(config: Union[str, Dict],
             run_root: str = "runs/nekrs",
             nproc: Optional[int] = None,
             iface: Optional[NekRSInterface] = None,
             power_weight: Optional[float] = None,
             power_cost: Optional[str] = None,
             power_sampling: Optional[str] = None)
```

iface: an interface to one instance of a multi-instance launch (see multisim); default: the only one.
power_weight / power_cost / power_sampling override the keys of the YAML power section (hydrogym.power).

#### u\_tau

```python
def u_tau(xyz: np.ndarray) -> np.ndarray
```

Friction velocity at the points (wall_units.u_tau: a value or a polynomial in one coordinate).

#### actuation\_power\_now

```python
def actuation_power_now() -> Dict
```

C_P forms of the current instant (pressure and imposed wall velocity of the power boundaries).

#### transform\_probes

```python
def transform_probes(vals: np.ndarray) -> np.ndarray
```

Raw probe values [n_probes, n_fields] -> observation (frame, group mean, normalization), probe-major.

#### wall\_velocity

```python
def wall_velocity(action: np.ndarray) -> np.ndarray
```

Actuator wall velocities for actions in [-1, 1] (amplitude, then the per-group zero net flux).

#### reset

```python
def reset(seed=None, options=None)
```

Start an episode from an initial condition of the library (see initial_conditions); with source
continue the flow runs on from its current state (as RSETS in the Nek5000 coupling).
`options={"index": i}` selects a library entry explicitly.

#### step\_async

```python
def step_async(action) -> None
```

First half of step(): set the actuation and start the solver steps without waiting for them.

#### set\_step\_hook

```python
def set_step_hook(every: int, fn=None) -> None
```

Call fn(env, row) after every solver step of an action whose global step index (solver_steps.csv tstep)
is a multiple of `every`; row = [tstep, time, dt, cfl, monitors...] of that step. The solver is idle
during the call, so fn may make requests such as iface.wall_data. With a hook the actions advance one
solver step per request, which gives the same flow. every = 0 (or fn None) removes the hook. Only the
steps of step() / step_async() call it, not those of reset().

#### power\_step\_progress

```python
def power_step_progress() -> bool
```

One-step mode (power_sampling mean or a step hook): wait for the running solver step, sample the power
and call the step hook, then start the next step of the action; True once all steps of the action are
done (then step_wait returns at once). Lets several instances advance in lockstep (multisim).

