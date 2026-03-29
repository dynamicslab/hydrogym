import abc
from typing import Any, Callable, Iterable, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike


class ActuatorBase:

  def __init__(self, state=0.0, **kwargs):
    self.x = state

  @property
  def state(self) -> float:
    return self.x

  @state.setter
  def state(self, u: float):
    self.x = u

  def step(self, u: float, dt: float):
    """Update the state of the actuator"""
    raise NotImplementedError


class PDEBase(metaclass=abc.ABCMeta):
  """
    Basic configuration of the state of the PDE model

    Will contain any time-varying flow fields, boundary
    conditions, actuation models, etc. Does not contain
    any information about solving the time-varying equations
    """

  MAX_CONTROL = np.inf
  DEFAULT_MESH = ""
  DEFAULT_DT = np.inf

  # Timescale used to smooth inputs
  #  (should be less than any meaningful timescale of the system)
  TAU = 0.0

  StateType = TypeVar("StateType")
  MeshType = TypeVar("MeshType")
  BCType = TypeVar("BCType")

  def __init__(self, **config):
    self.mesh = self.load_mesh(name=config.get("mesh", self.DEFAULT_MESH))
    self.initialize_state()

    self.reset()

    # Handle both single checkpoint (string) and multiple checkpoints (list)
    restart = config.get("restart")
    if restart is not None:
      if isinstance(restart, str):
        # Single checkpoint - load immediately
        self.load_checkpoint(restart)
      elif isinstance(restart, (list, tuple)):
        # Multiple checkpoints - load the first one as default
        # (FlowEnv will handle random selection)
        if len(restart) > 0:
          self.load_checkpoint(restart[0])
      else:
        raise ValueError(f"restart must be a string or list of strings, got {type(restart)}")

  @property
  @abc.abstractmethod
  def num_inputs(self) -> int:
    """Length of the control vector (number of actuators)"""
    pass

  @property
  @abc.abstractmethod
  def num_outputs(self) -> int:
    """Number of scalar observed variables"""
    pass

  @abc.abstractmethod
  def load_mesh(self, name: str) -> MeshType:
    """Load mesh from the file `name`"""
    pass

  @abc.abstractmethod
  def initialize_state(self):
    """Set up mesh, function spaces, state vector, etc"""
    pass

  @abc.abstractmethod
  def init_bcs(self):
    """Initialize any boundary conditions for the PDE."""
    pass

  def set_state(self, q: StateType):
    """Set the current state fields

        Should be overridden if a different assignment
        mechanism is used (e.g. `Function.assign`)

        Args:
            q (StateType): State to be assigned
        """
    self.q = q

  def state(self) -> StateType:
    """Return current state field(s) of the PDE"""
    return self.q

  @abc.abstractmethod
  def copy_state(self, deepcopy=True):
    """Return a copy of the flow state"""
    pass

  def reset(self, q0: StateType = None, t: float = 0.0):
    """Reset the PDE to an initial state

        Args:
            q0 (StateType, optional):
                State to which the PDE fields will be assigned.
                Defaults to None.
        """
    self.t = t
    if q0 is not None:
      self.set_state(q0)
    self.reset_controls()

  def reset_controls(self):
    """Reset the controls to a zero state

        Note that this is broken out from `reset` because
        the two are not necessarily called together (e.g.
        for linearization or deriving the control vector)

        """
    self.actuators = [ActuatorBase() for _ in range(self.num_inputs)]
    self.init_bcs()

  def collect_bcs(self) -> Iterable[BCType]:
    """Return the set of boundary conditions"""
    return []

  @abc.abstractmethod
  def save_checkpoint(self, filename: str):
    pass

  @abc.abstractmethod
  def load_checkpoint(self, filename: str):
    pass

  @abc.abstractmethod
  def get_observations(self) -> Iterable[ArrayLike]:
    """Return the set of measurements/observations"""
    pass

  @abc.abstractmethod
  def evaluate_objective(self, q: StateType = None) -> ArrayLike:
    """Return the objective function to be minimized

        Args:
            q (StateType, optional):
                State to evaluate the objective of, if not
                the current PDE state. Defaults to None.

        Returns:
            ArrayLike: objective function (negative of reward)
        """
    pass

  def enlist(self, x: Any) -> Iterable[Any]:
    """Convert scalar or array-like to a list"""
    if not isinstance(x, (list, tuple, np.ndarray)):
      x = [x]
    return list(x)

  @property
  def control_state(self) -> Iterable[ArrayLike]:
    return [a.state for a in self.actuators]

  def set_control(self, act: ArrayLike = None):
    """Directly set the control state"""
    if act is None:
      act = np.zeros(self.num_inputs)
    for i, u in enumerate(self.enlist(act)):
      self.actuators[i].state = u

  def advance_time(self, dt: float, act: list[float] = None) -> list[float]:
    """Update the current controls state. May involve integrating
        a dynamics model rather than directly setting the controls state.
        Here, if actual control is `u` and input is `v`, effectively
        `du/dt = (1/tau)*(v - u)`

        Args:
          act (Iterable[ArrayLike]): Action inputs
          dt (float): Time step

        Returns:
          Iterable[ArrayLike]: Updated actuator state
        """
    if act is None:
      act = self.control_state
    self.t += dt

    act = self.enlist(act)
    assert len(act) == self.num_inputs

    for i, u in enumerate(act):
      self.actuators[i].step(u, dt)

    return self.control_state

  def dot(self, q1: StateType, q2: StateType) -> float:
    """Inner product between states q1 and q2"""
    return np.dot(q1, q2)

  @abc.abstractmethod
  def render(self, **kwargs):
    """Plot the current PDE state (called by `gymnasium.Env`)"""
    pass


'''

@ray.remote
class EvaluationActor:
    """To remotely evaluate Firedrake solutions."""

    def __init__(self, firedrake_instance: "Firedrake_instance", index: int, seeds:
            Union[ActorSeeds, tuple], state:dict):
        """
        Initialize a remote runner for a Firedrake problem instance.

        Args:
            firedrake_instance: The Firedrake problem instance to be run.
            index: The index of the actor in question.
            seed: An integer to be used as the seed.
            state: The state dictionary to be loaded.
        """
       
       # Add whole class and the two following to have the evaluation logic.
       # Pipe Firedrake problem into Evotorch's harness, then simmer down the logic needed to have this 
       #   code here running.
'''


class CallbackBase:

  def __init__(self, interval: int = 1):
    """
        Base class for things that happen every so often in the simulation
        (e.g. save output for visualization or write some info to a log file).

        Args:
            interval (int, optional): How often to take action. Defaults to 1.

        TODO: Add a ControllerCallback
        """
    self.interval = interval

  def __call__(self, iter: int, t: float, flow: PDEBase) -> bool:
    """Check if this is an 'iostep' by comparing to `self.interval`

        Args:
            iter (int): Iteration number
            t (float): Time value
            flow (PDEBase): Underlying PDE model

        Returns:
            bool: whether or not to do the thing in this iteration
        """
    return iter % self.interval == 0

  def close(self):
    """Close any open files, etc."""
    pass


class TransientSolver:
  """Time-stepping code for updating the transient PDE"""

  def __init__(self, flow: PDEBase, dt: float = None):
    self.flow = flow
    if dt is None:
      dt = flow.DEFAULT_DT
    self.dt = dt

  def solve(
      self,
      t_span: Tuple[float, float] = None,
      num_steps: int = None,
      callbacks: Iterable[CallbackBase] = [],
      controller: Callable = None,
      collect_rewards: bool = False,
  ) -> Union[PDEBase, Tuple[PDEBase, np.ndarray]]:
    """Solve the initial-value problem for the PDE.

    Supports both time-span and fixed-step modes:
    - If t_span is provided: solve from t_span[0] to t_span[1] with self.dt
    - If num_steps is provided: solve for exactly num_steps iterations

        Args:
            t_span (Tuple[float, float], optional): Tuple of start and end times
                (mutually exclusive with num_steps)
            num_steps (int, optional): Number of steps to take
                (mutually exclusive with t_span)
            callbacks (Iterable[CallbackBase], optional):
                List of callbacks to evaluate throughout the solve. Defaults to [].
            controller (Callable, optional):
                Feedback/forward controller `u = ctrl(t, y)`
            collect_rewards (bool, optional): If True, collect and return rewards
                from each step. Defaults to False.

        Returns:
            PDEBase: The state of the PDE at the end
            OR
            Tuple[PDEBase, np.ndarray]: (state, rewards) if collect_rewards=True
        """
    if (t_span is None) == (num_steps is None):
      raise ValueError("Must provide exactly one of t_span or num_steps")

    rewards = [] if collect_rewards else None

    if num_steps is not None:
      # Fixed-step mode (for multi-substep simulation)
      for iter in range(num_steps):
        t = self.flow.t
        if controller is not None:
          y = self.flow.get_observations()
          u = controller(t, y)
        else:
          u = None
        flow = self.step(iter, control=u)

        if collect_rewards:
          reward_val = self.flow.evaluate_objective()
          rewards.append(reward_val)

        for cb in callbacks:
          cb(iter, t, flow)
    else:
      # Time-span mode (existing behavior)
      for iter, t in enumerate(np.arange(*t_span, self.dt)):
        if controller is not None:
          y = self.flow.get_observations()
          u = controller(t, y)
        else:
          u = None
        flow = self.step(iter, control=u)

        if collect_rewards:
          reward_val = self.flow.evaluate_objective()
          rewards.append(reward_val)

        for cb in callbacks:
          cb(iter, t, flow)

    for cb in callbacks:
      cb.close()

    if collect_rewards:
      return flow, np.array(rewards)
    return flow

  def step(self, iter: int, control: Iterable[float] = None, **kwargs):
    """Advance the transient simulation by one time step

        Args:
            iter (int): Iteration count
            control (Iterable[float], optional): Actuation input. Defaults to None.
        """
    raise NotImplementedError

  def reset(self):
    """Reset variables for the timestepper"""
    pass


class FlowEnv(gym.Env):

  def __init__(self, env_config: dict):
    self.flow: PDEBase = env_config.get("flow")(
        **env_config.get("flow_config", {}))
    self.solver: TransientSolver = env_config.get("solver")(
        self.flow, **env_config.get("solver_config", {}))
    self.callbacks: Iterable[CallbackBase] = env_config.get("callbacks", [])
    self.max_steps: int = env_config.get("max_steps", int(1e6))
    self.iter: int = 0

    # Multi-substep configuration
    import warnings
    actuation_config = env_config.get("actuation_config", {})

    # Support old config keys with deprecation warnings
    if "num_sim_substeps_per_actuation" in actuation_config:
      warnings.warn(
          "num_sim_substeps_per_actuation is deprecated, use num_substeps",
          DeprecationWarning,
          stacklevel=2
      )
      self.num_substeps = actuation_config["num_sim_substeps_per_actuation"]
    else:
      self.num_substeps = actuation_config.get("num_substeps", 1)

    if "reward_aggreation_rule" in actuation_config:
      warnings.warn(
          "reward_aggreation_rule is deprecated (and misspelled), use reward_aggregation",
          DeprecationWarning,
          stacklevel=2
      )
      self.reward_aggregation = actuation_config["reward_aggreation_rule"]
    else:
      self.reward_aggregation = actuation_config.get("reward_aggregation", "mean")

    if self.num_substeps < 1:
      raise ValueError(f"num_substeps must be >= 1, got {self.num_substeps}")

    if self.reward_aggregation not in ["mean", "sum", "median"]:
      raise ValueError(
          f"reward_aggregation must be 'mean', 'sum', or 'median', "
          f"got {self.reward_aggregation}"
      )

    # Multiple checkpoint support
    flow_config = env_config.get("flow_config", {})
    restart = flow_config.get("restart")

    if restart is None:
      # No checkpoints - use current state
      self.restart_checkpoints = None
      self.initial_states = [self.flow.copy_state()]
    elif isinstance(restart, str):
      # Single checkpoint - already loaded by PDEBase
      self.restart_checkpoints = [restart]
      self.initial_states = [self.flow.copy_state()]
    elif isinstance(restart, (list, tuple)):
      # Multiple checkpoints - preload all initial states
      self.restart_checkpoints = list(restart)
      self.initial_states = []

      for ckpt_path in self.restart_checkpoints:
        self.flow.load_checkpoint(ckpt_path)
        self.initial_states.append(self.flow.copy_state())

      # Reset to first checkpoint state
      self.flow.reset(q0=self.initial_states[0])
    else:
      raise ValueError(f"restart must be string or list, got {type(restart)}")

    self.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.flow.num_outputs,),
        dtype=float,
    )
    self.action_space = gym.spaces.Box(
        low=-self.flow.MAX_CONTROL,
        high=self.flow.MAX_CONTROL,
        shape=(self.flow.num_inputs,),
        dtype=float,
    )

  def set_callbacks(self, callbacks: Iterable[CallbackBase]):
    self.callbacks = callbacks

  def step(
      self,
      action: Iterable[ArrayLike] = None
  ) -> Tuple[ArrayLike, float, bool, bool, dict]:
    """Advance the state of the environment.  See gymnasium.Env documentation

        Args:
            action (Iterable[ArrayLike], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ArrayLike, float, bool, bool, dict]: obs, reward, terminated, truncated, info
        """
    if self.num_substeps == 1:
      # Single-step mode (fast path)
      self.solver.step(self.iter, control=action)
      self.iter += 1
      reward = self.get_reward()
    else:
      # Multi-substep mode
      def constant_controller(t, y):
        return action

      _, rewards = self.solver.solve(
          num_steps=self.num_substeps,
          controller=constant_controller,
          collect_rewards=True,
      )

      # Aggregate rewards
      if self.reward_aggregation == "mean":
        aggregated_objective = np.mean(rewards, axis=0)
      elif self.reward_aggregation == "sum":
        aggregated_objective = np.sum(rewards, axis=0)
      else:  # median
        aggregated_objective = np.median(rewards, axis=0)

      self.iter += self.num_substeps
      reward = -self.solver.dt * aggregated_objective

    # Execute callbacks
    t = self.iter * self.solver.dt
    for cb in self.callbacks:
      cb(self.iter, t, self.flow)

    obs = self.flow.get_observations()
    terminated = False  # No terminal state in continuous flow control
    truncated = self.check_complete()  # Episode truncated by max_steps
    info = {}

    obs = self.stack_observations(obs)

    return obs, reward, terminated, truncated, info

  # TODO: Use this to allow for arbitrary returns from collect_observations
  #  That are then converted to a list/tuple/ndarray here
  def stack_observations(self, obs):
    """Convert observations to numpy array format.

    Args:
        obs: Observations in various formats (tuple, list, ndarray, scalar)

    Returns:
        np.ndarray: Observations as a numpy array
    """
    if isinstance(obs, np.ndarray):
      return obs
    elif isinstance(obs, (list, tuple)):
      return np.array(obs, dtype=np.float64)
    else:
      # Scalar case
      return np.array([obs], dtype=np.float64)

  def get_reward(self):
    return -self.solver.dt * self.flow.evaluate_objective()

  def check_complete(self):
    return self.iter > self.max_steps

  def reset(self, seed=None, options=None) -> Tuple[ArrayLike, dict]:
    """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility (gymnasium API).
            options: Additional options (gymnasium API).

        Returns:
            Tuple[ArrayLike, dict]: (observation, info)
        """
    if seed is not None:
      np.random.seed(seed)

    # Randomly select from available initial states
    if len(self.initial_states) > 1:
      idx = np.random.randint(0, len(self.initial_states))
      q0 = self.initial_states[idx]
      info = {"checkpoint_index": idx}
    else:
      q0 = self.initial_states[0]
      info = {}

    t = options.get('t', 0.0) if options else 0.0
    self.iter = 0
    self.flow.reset(q0=q0, t=t)
    self.solver.reset()

    obs = self.flow.get_observations()
    obs = self.stack_observations(obs)

    return obs, info

  def render(self, mode="human", **kwargs):
    self.flow.render(mode=mode, **kwargs)

  def close(self):
    for cb in self.callbacks:
      cb.close()
