from typing import Iterable, Tuple, TypeVar, Union

import gym
import numpy as np

from legacy.hydrogym.core import PDEModel


class ActuatorBase:
    def step(self, u: float, dt: float):
        """Update the state of the actuator"""
        raise NotImplementedError


class PDEBase:
    """
    Basic configuration of the state of the PDE model

    Will contain any time-varying flow fields, boundary
    conditions, actuation models, etc. Does not contain
    any information about solving the time-varying equations
    """

    ACT_DIM = 1  # Number of actuators/actions
    MAX_CONTROL = np.inf
    DEFAULT_MESH = ""
    DEFAULT_DT = np.inf

    # Timescale used to smooth inputs
    #  (should be less than any meaningful timescale of the system)
    TAU = 0.0

    ScalarType = np.float32
    ActType = ScalarType
    ObsType = ScalarType
    StateType = TypeVar("StateType")
    MeshType = TypeVar("MeshType")
    BCType = TypeVar("BCType")

    def __init__(self, **config):
        self.mesh = self.load_mesh(name=config.get("mesh", self.DEFAULT_MESH))
        self.initialize_state()

        if config.get("restart"):
            self.load_checkpoint(config["restart"])

        self.reset()

    def load_mesh(self, name: str) -> MeshType:
        """Load mesh from the file `name`"""
        raise NotImplementedError

    def initialize_state(self):
        """Set up mesh, function spaces, state vector, etc"""
        raise NotImplementedError

    def init_bcs(self, mixed: bool = False):
        """Initialize any boundary conditions for the PDE.

        Args:
            mixed (bool, optional): determines a monolithic vs segregated formulation
                (may not always be necessary). Defaults to False.
        """
        raise NotImplementedError

    def set_state(self, q: StateType):
        """Set the current state fields

        Should be overridden if a different assignment
        mechansim is used (e.g. `Function.assign`)

        Args:
            q (StateType): State to be assigned
        """
        self.q = q

    def state(self) -> StateType:
        """Return current state field(s) of the PDE"""
        return self.q

    def copy_state(self, deepcopy=True):
        """Return a copy of the flow state"""
        raise NotImplementedError

    def reset(self, q0: StateType = None):
        """Reset the PDE to an initial state

        Args:
            q0 (StateType, optional):
                State to which the PDE fields will be assigned.
                Defaults to None.
        """
        if q0 is not None:
            self.set_state(q0)
        self.reset_control()

    def reset_control(self, mixed: bool = False):
        """Reset the controls to a zero state

        Note that this is broken out from `reset` because
        the two are not necessarily called together (e.g.
        for linearization or deriving the control vector)

        Args:
            mixed (bool, optional):
                determines a monolithic vs segregated formulation
                (see `init_bcs`). Defaults to False.

        TODO: Rewrite with ActuatorBase
        """
        self.control = self.enlist_controls(np.zeros(self.ACT_DIM))
        self.init_bcs(mixed=mixed)

    def collect_bcs(self) -> Iterable[BCType]:
        """Return the set of boundary conditions"""
        return []

    def save_checkpoint(self, filename: str):
        raise NotImplementedError

    def load_checkpoint(self, filename: str):
        raise NotImplementedError

    def get_observations(self) -> Iterable[ObsType]:
        """Return the set of measurements/observations"""
        raise NotImplementedError

    def evaluate_objective(self, q: StateType = None) -> ScalarType:
        """Return the objective function to be minimized

        Args:
            q (StateType, optional):
                State to evaluate the objective of, if not
                the current PDE state. Defaults to None.

        Returns:
            ScalarType: objective function (negative of reward)
        """
        raise NotImplementedError

    def enlist_controls(self, control: ActType) -> Iterable[ActType]:
        """Convert scalar or array-like controls to a list of the correct type"""
        if isinstance(control, int) or isinstance(control, float):
            control = [control]
        return [self.ActType(c) for c in control]

    def set_control(self, control: ActType = None):
        """Directly set the control state"""
        if control is None:
            control = np.zeros(self.ACT_DIM)
        self.control = self.enlist_controls(control)

    def update_controls(self, act: Iterable[ActType], dt: float) -> Iterable[ActType]:
        """Update the current controls state.

        May involve integrating a dynamics model rather than
        directly setting the controls state.  Here, if actual
        control is `u` and input is `v`, effectively
            `du/dt = (1/tau)*(v - u)`

        To directly set the control state, see `set_controls`

        Args:
            act (Iterable[ActType]): Action inputs
            dt (float): Time step

        Returns:
            Iterable[ActType]: Updated control state

        TODO: Rewrite with ActuatorBase
        """
        act = self.enlist_controls(act)
        assert len(act) == self.ACT_DIM

        for i, (u, v) in enumerate(zip(self.control, act)):
            self.control[i] += (dt / self.TAU) * (v - u)
        return self.control

    def dot(self, q1: StateType, q2: StateType) -> float:
        """Inner product between states q1 and q2"""
        return np.dot(q1, q2)

    def render(self, **kwargs):
        """Plot the current PDE state (called by `gym.Env`)"""
        raise NotImplementedError


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

    def __init__(self, flow: PDEBase, dt: float):
        self.flow = flow
        self.dt = dt
        self.t = 0.0

    def solve(
        self, t_span: Tuple[float, float], callbacks: Iterable[CallbackBase] = []
    ) -> PDEModel:
        """Solve the initial-value problem for the PDE.

        Args:
            t_span (Tuple[float, float]): Tuple of start and end times
            callbacks (Iterable[CallbackBase], optional):
                List of callbacks to evaluate throughout the solve. Defaults to [].

        Returns:
            PDEModel: The state of the PDE at the end of the solve
        """
        for iter, t in enumerate(np.arange(*t_span, self.dt)):
            flow = self.step(iter)
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


class FlowEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.flow: PDEModel = env_config.get("flow")(env_config)
        self.solver: TransientSolver = env_config.get("solver")(
            self.flow, dt=env_config.get("dt", self.flow.DEFAULT_DT)
        )
        self.callbacks: Iterable[CallbackBase] = env_config.get("callbacks", [])
        self.max_steps: int = env_config.get("max_steps", int(1e6))
        self.iter: int = 0
        self.q0: self.flow.StateType = self.flow.copy_state()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=PDEBase.ScalarType
        )

    def set_callbacks(self, callbacks: Iterable[CallbackBase]):
        self.callbacks = callbacks

    def step(
        self, action: Iterable[PDEBase.ActType] = None
    ) -> Tuple[PDEBase.ObsType, float, bool, dict]:
        """Advance the state of the environment.  See gym.Env documentation

        Args:
            action (Iterable[ActType], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ObsType, float, bool, dict]: obs, reward, done, info
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
        return -self.flow.evaluate_objective()

    def check_complete(self):
        return self.iter > self.max_steps

    def reset(self) -> Union[PDEBase.ObsType, Tuple[PDEBase.ObsType, dict]]:
        self.iter = 0
        self.flow.reset(q0=self.q0)
        self.solver.reset()

        return self.flow.get_observations()

    def render(self, mode="human"):
        self.flow.render(mode=mode)

    def close(self):
        for cb in self.callbacks:
            cb.close()
