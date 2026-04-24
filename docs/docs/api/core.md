---
sidebar_label: core
title: hydrogym.core
---

## ActuatorBase Objects

```python
class ActuatorBase()
```

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

#### \_\_init\_\_

```python
def __init__(interval: int = 1)
```

Base class for things that happen every so often in the simulation
(e.g. save output for visualization or write some info to a log file).

**Arguments**:

- `interval` _int, optional_ - How often to take action. Defaults to 1.
  
- `TODO` - Add a ControllerCallback

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

