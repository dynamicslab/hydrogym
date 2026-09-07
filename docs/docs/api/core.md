---
sidebar_label: core
title: hydrogym.core
---

## ActuatorBase Objects

```python
class ActuatorBase()
```

Base class for a single actuator driving the flow.

An actuator holds a scalar state ``x`` that the controller updates each
step. Subclasses override :meth:`step` to define how the applied state
evolves from the requested input (e.g. first-order smoothing towards the
input over the flow&#x27;s ``TAU`` timescale) rather than setting it directly.

**Attributes**:

- `x` - Current actuator state.

#### \_\_init\_\_

```python
def __init__(state=0.0, **kwargs)
```

Initialize the actuator to a constant state.

**Arguments**:

- `state` - Initial actuator state. Defaults to 0.0.
- `**kwargs` - Ignored; accepted so subclasses can pass through
  shared configuration dictionaries unchanged.

#### state

```python
@property
def state() -> float
```

Return the current actuator state.

**Returns**:

- `float` - Current value of the actuator state ``x``.

#### state

```python
@state.setter
def state(u: float)
```

Set the actuator state directly.

**Arguments**:

- `u` _float_ - New actuator state.

#### step

```python
def step(u: float, dt: float)
```

Update the state of the actuator

## PDEBase Objects

```python
class PDEBase(metaclass=abc.ABCMeta)
```

Basic configuration of the state of the PDE model

Will contain any time-varying flow fields, boundary
conditions, actuation models, etc. Does not contain
any information about solving the time-varying equations

#### \_\_init\_\_

```python
def __init__(**config)
```

Configure the PDE from keyword arguments and prepare it for use.

Loads the mesh, initializes the state, and resets the model to its
initial condition. Also handles the ``restart`` option: a single
checkpoint path (string) is loaded immediately, while a list/tuple of
paths loads the first one as the default state (FlowEnv handles
random selection among the full set at reset time).

Subclasses are expected to ``.pop`` their own config keys before
calling ``super().__init__``; any keys left in ``config`` at this
point are almost certainly typos or unsupported options, so a
warning is emitted rather than silently ignoring them.

**Arguments**:

- `**config` - Configuration options. PDEBase itself consumes:
  - mesh (str): Mesh name passed to ``load_mesh``.
  Defaults to ``DEFAULT_MESH``.
  - restart (str | list | tuple): Checkpoint file(s) to load.
  Defaults to None (start from the initial condition).
  

**Raises**:

- ``3 - If ``restart`` is neither a string nor a list/tuple
  of strings.

#### num\_inputs

```python
@property
@abc.abstractmethod
def num_inputs() -> int
```

Length of the control vector (number of actuators)

#### num\_outputs

```python
@property
@abc.abstractmethod
def num_outputs() -> int
```

Number of scalar observed variables

#### load\_mesh

```python
@abc.abstractmethod
def load_mesh(name: str) -> MeshType
```

Load mesh from the file `name`

#### initialize\_state

```python
@abc.abstractmethod
def initialize_state()
```

Set up mesh, function spaces, state vector, etc

#### init\_bcs

```python
@abc.abstractmethod
def init_bcs()
```

Initialize any boundary conditions for the PDE.

#### set\_state

```python
def set_state(q: StateType)
```

Set the current state fields

Should be overridden if a different assignment
mechanism is used (e.g. `Function.assign`)

**Arguments**:

- `q` _StateType_ - State to be assigned

#### state

```python
def state() -> StateType
```

Return current state field(s) of the PDE

#### copy\_state

```python
@abc.abstractmethod
def copy_state(deepcopy=True)
```

Return a copy of the flow state

#### reset

```python
def reset(q0: StateType = None, t: float = 0.0)
```

Reset the PDE to an initial state

**Arguments**:

  q0 (StateType, optional):
  State to which the PDE fields will be assigned.
  Defaults to None.

#### reset\_controls

```python
def reset_controls()
```

Reset the controls to a zero state

Note that this is broken out from `reset` because
the two are not necessarily called together (e.g.
for linearization or deriving the control vector)

#### collect\_bcs

```python
def collect_bcs() -> Iterable[BCType]
```

Return the set of boundary conditions

#### save\_checkpoint

```python
@abc.abstractmethod
def save_checkpoint(filename: str)
```

Write the current PDE state to a checkpoint file.

**Arguments**:

- `filename` _str_ - Path of the checkpoint file to write.

#### load\_checkpoint

```python
@abc.abstractmethod
def load_checkpoint(filename: str)
```

Load the PDE state from a checkpoint file.

**Arguments**:

- `filename` _str_ - Path of a checkpoint previously written by
  ``save_checkpoint``.

#### get\_observations

```python
@abc.abstractmethod
def get_observations() -> Iterable[ArrayLike]
```

Return the set of measurements/observations

#### evaluate\_objective

```python
@abc.abstractmethod
def evaluate_objective(q: StateType = None) -> ArrayLike
```

Return the objective function to be minimized

**Arguments**:

  q (StateType, optional):
  State to evaluate the objective of, if not
  the current PDE state. Defaults to None.
  

**Returns**:

- `ArrayLike` - objective function (negative of reward)

#### enlist

```python
def enlist(x: Any) -> Iterable[Any]
```

Convert scalar or array-like to a list

#### control\_state

```python
@property
def control_state() -> Iterable[ArrayLike]
```

Return the current control vector (one entry per actuator).

#### set\_control

```python
def set_control(act: ArrayLike = None)
```

Directly set the control state

#### advance\_time

```python
def advance_time(dt: float, act: list[float] = None) -> list[float]
```

Update the current controls state. May involve integrating
a dynamics model rather than directly setting the controls state.
Here, if actual control is `u` and input is `v`, effectively
`du/dt = (1/tau)*(v - u)`

**Arguments**:

- `act` _Iterable[ArrayLike]_ - Action inputs
- `dt` _float_ - Time step
  

**Returns**:

- `Iterable[ArrayLike]` - Updated actuator state

#### dot

```python
def dot(q1: StateType, q2: StateType) -> float
```

Inner product between states q1 and q2

#### render

```python
@abc.abstractmethod
def render(**kwargs)
```

Plot the current PDE state (called by `gymnasium.Env`)

## CallbackBase Objects

```python
class CallbackBase()
```

Base class for things that happen every so often in the simulation.

Concrete callbacks (e.g. saving output for visualization or writing log
entries) are invoked by ``TransientSolver.solve`` each iteration; the
default ``__call__`` acts only every ``interval`` iterations.

TODO: Add a ControllerCallback

#### \_\_init\_\_

```python
def __init__(interval: int = 1)
```

**Arguments**:

- `interval` _int, optional_ - How often to take action. Defaults to 1.

#### \_\_call\_\_

```python
def __call__(iter: int, t: float, flow: PDEBase) -> bool
```

Check if this is an &#x27;iostep&#x27; by comparing to `self.interval`

**Arguments**:

- `iter` _int_ - Iteration number
- `t` _float_ - Time value
- `flow` _PDEBase_ - Underlying PDE model
  

**Returns**:

- `bool` - whether or not to do the thing in this iteration

#### close

```python
def close()
```

Close any open files, etc.

## TransientSolver Objects

```python
class TransientSolver()
```

Time-stepping code for updating the transient PDE

#### \_\_init\_\_

```python
def __init__(flow: PDEBase, dt: float = None)
```

Bind the solver to a flow and select the time step.

**Arguments**:

- `flow` _PDEBase_ - The PDE model to be time-stepped.
- `dt` _float, optional_ - Time step to use. Defaults to the flow&#x27;s
  ``DEFAULT_DT`` if not given.

#### solve

```python
def solve(
    t_span: Tuple[float, float] = None,
    num_steps: int = None,
    callbacks: Iterable[CallbackBase] = [],
    controller: Callable = None,
    collect_rewards: bool = False
) -> Union[PDEBase, Tuple[PDEBase, np.ndarray]]
```

Solve the initial-value problem for the PDE.

Supports both time-span and fixed-step modes:
- If t_span is provided: solve from t_span[0] to t_span[1] with self.dt
- If num_steps is provided: solve for exactly num_steps iterations

**Arguments**:

- `t_span` _Tuple[float, float], optional_ - Tuple of start and end times
  (mutually exclusive with num_steps)
- `num_steps` _int, optional_ - Number of steps to take
  (mutually exclusive with t_span)
  callbacks (Iterable[CallbackBase], optional):
  List of callbacks to evaluate throughout the solve. Defaults to [].
  controller (Callable, optional):
  Feedback/forward controller `u = ctrl(t, y)`
- `collect_rewards` _bool, optional_ - If True, collect and return rewards
  from each step. Defaults to False.
  

**Returns**:

- `PDEBase` - The state of the PDE at the end
  OR
  Tuple[PDEBase, np.ndarray]: (state, rewards) if collect_rewards=True

#### step

```python
def step(iter: int, control: Iterable[float] = None, **kwargs)
```

Advance the transient simulation by one time step

**Arguments**:

- `iter` _int_ - Iteration count
- `control` _Iterable[float], optional_ - Actuation input. Defaults to None.

#### reset

```python
def reset()
```

Reset variables for the timestepper

## FlowEnv Objects

```python
class FlowEnv(gym.Env)
```

Gymnasium environment wrapping a PDE model and its transient solver.

Each ``step`` advances the flow and returns the (negated, time-scaled)
objective as the reward. Reaching the configured step budget is reported
as a truncation, not a termination. Optionally, one ``step`` can advance
the simulation by several solver substeps while holding the action
constant, with per-substep rewards aggregated by a configurable rule.

**Attributes**:

- `flow` - The underlying PDE model.
- `solver` - The transient solver driving the PDE.
- `callbacks` - Callbacks invoked after each step.
- `max_steps` - Episode length in environment steps.
- `iter` - Total number of solver steps taken since the last reset.
- `num_substeps` - Solver steps per environment step.
- ``0 - How per-substep objectives are combined
  (&#x27;mean&#x27;, &#x27;sum&#x27;, or &#x27;median&#x27;).
- ``1 - List of checkpoint paths (if any) available
  for random selection on reset, else None.
- ``2 - Preloaded flow states to reset to.

#### \_\_init\_\_

```python
def __init__(env_config: dict)
```

Build the flow and solver from ``env_config`` and set up spaces.

Multi-substep actuation is configured via ``actuation_config``. The
old keys ``num_sim_substeps_per_actuation`` and
``reward_aggreation_rule`` (misspelled) are still accepted but emit a
DeprecationWarning; their non-deprecated replacements are
``num_substeps`` and ``reward_aggregation``.

The ``restart`` entry of ``flow_config`` selects the initial states:
a string means a single checkpoint (already loaded by the flow), a
list/tuple means several checkpoints, all of which are preloaded as
candidate initial states (reset picks one at random).

**Arguments**:

- ``6 _dict_ - Configuration dictionary containing:
  - flow (type): Callable (usually a PDEBase subclass)
  constructing the flow from ``flow_config``.
  - flow_config (dict, optional): Keyword configuration passed
  to the flow constructor.
  - solver (type): Callable (usually a TransientSolver
  subclass) constructing the solver as
  ``solver(flow, **solver_config)``.
  - solver_config (dict, optional): Keyword configuration
  passed to the solver.
  - callbacks (Iterable[CallbackBase], optional): Callbacks
  invoked after each step. Defaults to [].
  - max_steps (int, optional): Steps per episode. Defaults to
  1e6.
  - actuation_config (dict, optional): Multi-substep options
  (see above). Defaults to one substep and &#x27;mean&#x27;
  aggregation.
  

**Raises**:

- ``1 - If ``num_substeps &lt; 1``, if ``reward_aggregation``
  is not &#x27;mean&#x27;, &#x27;sum&#x27;, or &#x27;median&#x27;, or if ``restart`` is
  neither a string nor a list/tuple.

#### set\_callbacks

```python
def set_callbacks(callbacks: Iterable[CallbackBase])
```

Replace the environment&#x27;s callbacks.

**Arguments**:

- `callbacks` _Iterable[CallbackBase]_ - Callbacks to invoke after
  each step.

#### step

```python
def step(
    action: Iterable[ArrayLike] = None
) -> Tuple[ArrayLike, float, bool, bool, dict]
```

Advance the state of the environment.  See gymnasium.Env documentation

**Arguments**:

- `action` _Iterable[ArrayLike], optional_ - Control inputs. Defaults to None.
  

**Returns**:

  Tuple[ArrayLike, float, bool, bool, dict]: obs, reward, terminated, truncated, info

#### stack\_observations

```python
def stack_observations(obs)
```

Convert observations to numpy array format.

**Arguments**:

- `obs` - Observations in various formats (tuple, list, ndarray, scalar)
  

**Returns**:

- `np.ndarray` - Observations as a numpy array

#### get\_reward

```python
def get_reward()
```

Return the reward for the current flow state.

The reward is the negative of the flow&#x27;s objective function (which
is to be minimized), scaled by the solver time step.

**Returns**:

- `float` - Reward ``-dt * evaluate_objective()``.

#### check\_complete

```python
def check_complete()
```

Check whether the episode has exceeded its step budget.

**Returns**:

- `bool` - True if the number of elapsed steps exceeds ``max_steps``.

#### reset

```python
def reset(seed=None, options=None) -> Tuple[ArrayLike, dict]
```

Reset the environment to initial state.

**Arguments**:

- `seed` - Random seed for reproducibility (gymnasium API).
- `options` - Additional options (gymnasium API).
  

**Returns**:

  Tuple[ArrayLike, dict]: (observation, info)

#### render

```python
def render(mode="human", **kwargs)
```

Render the current PDE state.

**Arguments**:

- `mode` _str, optional_ - Render mode passed through to the flow.
  Defaults to &quot;human&quot;.
- `**kwargs` - Additional keyword arguments forwarded to
  ``flow.render``.

#### close

```python
def close()
```

Close the environment by closing all registered callbacks.

