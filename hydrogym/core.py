import abc
from typing import Any, Callable, Iterable, Tuple, TypeVar, Union

import gym
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

    if config.get("restart"):
      self.load_checkpoint(config["restart"])

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
    """Plot the current PDE state (called by `gym.Env`)"""
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
      t_span: Tuple[float, float],
      callbacks: Iterable[CallbackBase] = [],
      controller: Callable = None,
  ) -> PDEBase:
    """Solve the initial-value problem for the PDE.

        Args:
            t_span (Tuple[float, float]): Tuple of start and end times
            callbacks (Iterable[CallbackBase], optional):
                List of callbacks to evaluate throughout the solve. Defaults to [].
            controller (Callable, optional):
                Feedback/forward controller `u = ctrl(t, y)`

        Returns:
            PDEBase: The state of the PDE at the end of the solve
        """
    for iter, t in enumerate(np.arange(*t_span, self.dt)):
      if controller is not None:
        y = self.flow.get_observations()
        u = controller(t, y)
      else:
        u = None
      flow = self.step(iter, control=u)
      for cb in callbacks:
        cb(iter, t, flow)

    for cb in callbacks:
      cb.close()

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
    self.q0: self.flow.StateType = self.flow.copy_state()

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
  ) -> Tuple[ArrayLike, float, bool, dict]:
    """Advance the state of the environment.  See gym.Env documentation

        Args:
            action (Iterable[ArrayLike], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ArrayLike, float, bool, dict]: obs, reward, done, info
        """
    self.solver.step(self.iter, control=action)
    self.iter += 1
    t = self.iter * self.solver.dt
    for cb in self.callbacks:
      cb(self.iter, t, self.flow)
    obs = self.flow.get_observations()

    reward = self.get_reward()
    done = self.check_complete()
    info = {}

    obs = self.stack_observations(obs)

    return obs, reward, done, info

  # TODO: Use this to allow for arbitrary returns from collect_observations
  #  That are then converted to a list/tuple/ndarray here
  def stack_observations(self, obs):
    return obs

  def get_reward(self):
    return -self.solver.dt * self.flow.evaluate_objective()

  def check_complete(self):
    return self.iter > self.max_steps

  def reset(self, t=0.0) -> Union[ArrayLike, Tuple[ArrayLike, dict]]:
    self.iter = 0
    self.flow.reset(q0=self.q0, t=t)
    self.solver.reset()

    return self.flow.get_observations()

  def render(self, mode="human", **kwargs):
    self.flow.render(mode=mode, **kwargs)

  def close(self):
    for cb in self.callbacks:
      cb.close()
