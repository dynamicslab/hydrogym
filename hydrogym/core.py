import abc
import warnings
from typing import Any, Callable, Iterable, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from .power import PowerConfig, power_forms, power_info


class ActuatorBase:
    """Base class for a single actuator driving the flow.

    An actuator holds a scalar state ``x`` that the controller updates each
    step. Subclasses override ``step`` to define how the applied state
    evolves from the requested input (e.g. first-order smoothing towards the
    input over the flow's ``TAU`` timescale) rather than setting it directly.

    Attributes:
        x: Current actuator state.
    """

    def __init__(self, state=0.0, **kwargs):
        """Initialize the actuator to a constant state.

        Args:
            state: Initial actuator state. Defaults to 0.0.
            **kwargs: Ignored; accepted so subclasses can pass through
                shared configuration dictionaries unchanged.
        """
        self.x = state

    @property
    def state(self) -> float:
        """Return the current actuator state.

        Returns:
            float: Current value of the actuator state ``x``.
        """
        return self.x

    @state.setter
    def state(self, u: float):
        """Set the actuator state directly.

        Args:
            u (float): New actuator state.
        """
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
        """Configure the PDE from keyword arguments and prepare it for use.

        Loads the mesh, initializes the state, and resets the model to its
        initial condition. Also handles the ``restart`` option: a single
        checkpoint path (string) is loaded immediately, while a list/tuple of
        paths loads the first one as the default state (FlowEnv handles
        random selection among the full set at reset time).

        Subclasses are expected to ``.pop`` their own config keys before
        calling ``super().__init__``; any keys left in ``config`` at this
        point are almost certainly typos or unsupported options, so a
        warning is emitted rather than silently ignoring them.

        Args:
            **config: Configuration options. PDEBase itself consumes:
                - mesh (str): Mesh name passed to ``load_mesh``.
                  Defaults to ``DEFAULT_MESH``.
                - restart (str | list | tuple): Checkpoint file(s) to load.
                  Defaults to None (start from the initial condition).

        Raises:
            ValueError: If ``restart`` is neither a string nor a list/tuple
                of strings.
        """
        # Consume the keys PDEBase itself owns. Subclasses consume their own
        # keys (with .pop) before calling super().__init__; anything left in
        # `config` after that is almost certainly a typo'd or unsupported
        # option, so warn rather than silently ignoring it.
        self.mesh = self.load_mesh(name=config.pop("mesh", self.DEFAULT_MESH))
        self.initialize_state()

        self.reset()

        # Handle both single checkpoint (string) and multiple checkpoints (list)
        restart = config.pop("restart", None)
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

        leftover = sorted(config)
        if leftover:
            warnings.warn(
                f"{type(self).__name__} got unsupported flow_config key(s) {leftover}; "
                "they have no effect and may indicate a misspelled option.",
                stacklevel=2,
            )

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
        """Write the current PDE state to a checkpoint file.

        Args:
            filename (str): Path of the checkpoint file to write.
        """
        pass

    @abc.abstractmethod
    def load_checkpoint(self, filename: str):
        """Load the PDE state from a checkpoint file.

        Args:
            filename (str): Path of a checkpoint previously written by
                ``save_checkpoint``.
        """
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
        """Return the current control vector (one entry per actuator)."""
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


class CallbackBase:
    """Base class for things that happen every so often in the simulation.

    Concrete callbacks (e.g. saving output for visualization or writing log
    entries) are invoked by ``TransientSolver.solve`` each iteration; the
    default ``__call__`` acts only every ``interval`` iterations.
    """

    def __init__(self, interval: int = 1):
        """
        Args:
            interval (int, optional): How often to take action. Defaults to 1.
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
        """Bind the solver to a flow and select the time step.

        Args:
            flow (PDEBase): The PDE model to be time-stepped.
            dt (float, optional): Time step to use. Defaults to the flow's
                ``DEFAULT_DT`` if not given.
        """
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


class _PowerSampler(CallbackBase):
    """Collects actuation power (and drag) after every solver step of an action interval."""

    def __init__(self, env: "FlowEnv"):
        """Sample the power of ``env`` after every solver step."""
        super().__init__(interval=1)
        self.env = env
        self.samples = []

    def __call__(self, iter: int, t: float, flow: PDEBase) -> bool:
        """Record one power sample of the environment."""
        self.samples.append(self.env._power_sample())
        return True


class FlowEnv(gym.Env):
    """Gymnasium environment wrapping a PDE model and its transient solver.

    Each ``step`` advances the flow and returns the (negated, time-scaled)
    objective as the reward. Reaching the configured step budget is reported
    as a truncation, not a termination. Optionally, one ``step`` can advance
    the simulation by several solver substeps while holding the action
    constant, with per-substep rewards aggregated by a configurable rule.

    Attributes:
        flow: The underlying PDE model.
        solver: The transient solver driving the PDE.
        callbacks: Callbacks invoked after each step.
        max_steps: Episode length in solver steps (``iter`` counts solver steps, ``num_substeps`` per environment step).
            The episode is truncated once ``iter > max_steps``, i.e. after ``max_steps // num_substeps + 1`` environment steps.
        iter: Total number of solver steps taken since the last reset.
        num_substeps: Solver steps per environment step.
        reward_aggregation: How per-substep objectives are combined
            ('mean', 'sum', or 'median').
        restart_checkpoints: List of checkpoint paths (if any) available
            for random selection on reset, else None.
        initial_states: Preloaded flow states to reset to.
    """

    def __init__(self, env_config: dict):
        """Build the flow and solver from ``env_config`` and set up spaces.

        Multi-substep actuation is configured via ``actuation_config``. The
        old keys ``num_sim_substeps_per_actuation`` and
        ``reward_aggreation_rule`` (misspelled) are still accepted but emit a
        DeprecationWarning; their non-deprecated replacements are
        ``num_substeps`` and ``reward_aggregation``.

        The solver iteration count runs on across the substeps of all
        environment steps of an episode (multistep solvers such as
        ``SemiImplicitBDF`` take their start-up steps only after a reset).
        ``actuation_config["legacy_substep_iteration"] = True`` reproduces
        HydroGym <= 1.0, which restarts the count at 0 in every
        environment step with ``num_substeps > 1`` (the BDF start-up
        solvers then run again at every action) and evaluates the observations
        before every substep.

        The ``restart`` entry of ``flow_config`` selects the initial states:
        a string means a single checkpoint (already loaded by the flow), a
        list/tuple means several checkpoints, all of which are preloaded as
        candidate initial states (reset picks one at random).

        Args:
            env_config (dict): Configuration dictionary containing:
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
                  (see above). Defaults to one substep and 'mean'
                  aggregation.
                - power_weight, power_cost, power_sampling (optional):
                  actuation-power options (see ``hydrogym.power``):
                  reward <- reward - power_weight * C_P[power_cost];
                  ``power_sampling`` is "mean" (default; mean over the
                  solver substeps of the action) or "end". The default
                  ``power_weight = 0`` leaves the reward unchanged. The
                  power is reported in ``info`` whenever the flow defines
                  ``actuation_power``.

        Raises:
            ValueError: If ``num_substeps < 1``, if ``reward_aggregation``
                is not 'mean', 'sum', or 'median', or if ``restart`` is
                neither a string nor a list/tuple.
        """
        self.flow: PDEBase = env_config.get("flow")(**env_config.get("flow_config", {}))
        self.solver: TransientSolver = env_config.get("solver")(self.flow, **env_config.get("solver_config", {}))
        self.callbacks: Iterable[CallbackBase] = env_config.get("callbacks", [])
        self.max_steps: int = env_config.get("max_steps", int(1e6))
        self.iter: int = 0

        # Multi-substep configuration
        import warnings

        actuation_config = env_config.get("actuation_config", {})

        # Support old config keys with deprecation warnings
        if "num_sim_substeps_per_actuation" in actuation_config:
            warnings.warn(
                "num_sim_substeps_per_actuation is deprecated, use num_substeps", DeprecationWarning, stacklevel=2
            )
            self.num_substeps = actuation_config["num_sim_substeps_per_actuation"]
        else:
            self.num_substeps = actuation_config.get("num_substeps", 1)

        if "reward_aggreation_rule" in actuation_config:
            warnings.warn(
                "reward_aggreation_rule is deprecated (and misspelled), use reward_aggregation",
                DeprecationWarning,
                stacklevel=2,
            )
            self.reward_aggregation = actuation_config["reward_aggreation_rule"]
        else:
            self.reward_aggregation = actuation_config.get("reward_aggregation", "mean")
        self.legacy_substep_iteration = bool(actuation_config.get("legacy_substep_iteration", False))

        if self.num_substeps < 1:
            raise ValueError(f"num_substeps must be >= 1, got {self.num_substeps}")

        if self.reward_aggregation not in ["mean", "sum", "median"]:
            raise ValueError(f"reward_aggregation must be 'mean', 'sum', or 'median', got {self.reward_aggregation}")

        # Actuation power (reported in info; optionally charged in the reward)
        self.power_config = PowerConfig.from_config(env_config)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self._has_power = callable(getattr(self.flow, "actuation_power", None))
        if self._has_power:
            try:
                self.flow.actuation_power()
            except NotImplementedError:
                self._has_power = False
        if self.power_config.weight > 0 and not self._has_power:
            raise ValueError(f"power_weight > 0 but {type(self.flow).__name__} does not implement actuation_power")

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
        """Replace the environment's callbacks.

        Args:
            callbacks (Iterable[CallbackBase]): Callbacks to invoke after
                each step.
        """
        self.callbacks = callbacks

    def step(self, action: Iterable[ArrayLike] = None) -> Tuple[ArrayLike, float, bool, bool, dict]:
        """Advance the state of the environment.  See gymnasium.Env documentation

        Args:
            action (Iterable[ArrayLike], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ArrayLike, float, bool, bool, dict]: obs, reward, terminated, truncated, info
        """
        sampler = _PowerSampler(self) if (self._has_power and self.power_sampling == "mean") else None
        if self.num_substeps == 1:
            # Single-step mode (fast path)
            self.solver.step(self.iter, control=action)
            self.iter += 1
            reward = self.get_reward()
            objective = -reward / self.solver.dt
        else:
            if self.legacy_substep_iteration:
                # HydroGym <= 1.0: solve() numbers the substeps 0..num_substeps-1 in every step
                def constant_controller(t, y):
                    """Return the (fixed) action, holding it constant across substeps."""
                    return action

                _, rewards = self.solver.solve(
                    num_steps=self.num_substeps,
                    controller=constant_controller,
                    collect_rewards=True,
                    callbacks=[sampler] if sampler is not None else [],
                )
                self.iter += self.num_substeps
            else:
                # Multi-substep mode: hold the action over all substeps and keep
                # the iteration count running so multistep solvers (e.g. BDF)
                # only take their start-up steps at the beginning of an episode
                rewards = []
                for _ in range(self.num_substeps):
                    self.solver.step(self.iter, control=action)
                    self.iter += 1
                    rewards.append(self.flow.evaluate_objective())
                    if sampler is not None:
                        sampler(self.iter, self.flow.t, self.flow)

            # Aggregate rewards
            if self.reward_aggregation == "mean":
                aggregated_objective = np.mean(rewards, axis=0)
            elif self.reward_aggregation == "sum":
                aggregated_objective = np.sum(rewards, axis=0)
            else:  # median
                aggregated_objective = np.median(rewards, axis=0)

            reward = -self.solver.dt * aggregated_objective
            objective = aggregated_objective

        info = {}
        if self._has_power:
            if sampler is None or not sampler.samples:
                samples = [self._power_sample()]
            else:
                samples = sampler.samples
            info = self._power_info(samples, objective)
            reward = self.power_config.apply(reward, info["_forms"])
            del info["_forms"]

        # Execute callbacks
        t = self.iter * self.solver.dt
        for cb in self.callbacks:
            cb(self.iter, t, self.flow)

        obs = self.flow.get_observations()
        terminated = False  # No terminal state in continuous flow control
        truncated = self.check_complete()  # Episode truncated by max_steps

        obs = self.stack_observations(obs)

        return obs, reward, terminated, truncated, info

    def _power_sample(self):
        """Per-actuator power coefficients and drag coefficient of the current flow state."""
        if callable(getattr(self.flow, "actuation_power_forms", None)):
            per_actuator, forms = self.flow.actuation_power_forms()
        else:
            per_actuator = self.flow.actuation_power()
            forms = None
        per_actuator = np.asarray(per_actuator, dtype=float).ravel()
        if forms is None:
            forms = power_forms(per_actuator)
        drag = self.flow.drag_coefficient() if callable(getattr(self.flow, "drag_coefficient", None)) else None
        return per_actuator, forms, drag

    def _power_info(self, samples, objective) -> dict:
        """Power info entries from one ("end") or several ("mean") samples.

        The cost forms are formed for every sample (per actuator, or per surface point for jets) and then
        averaged.
        """
        per_actuator = np.mean([s[0] for s in samples], axis=0)
        forms = {k: float(np.mean([s[1][k] for s in samples])) for k in ("signed", "positive", "abs")}
        drags = [s[2] for s in samples if s[2] is not None]
        drag = float(np.mean(drags)) if drags else None
        obj = {
            "objective": float(np.asarray(objective).squeeze()) if np.size(objective) == 1 else np.asarray(objective)
        }
        info = power_info(
            per_actuator,
            sampling=self.power_sampling,
            drag_coefficient=drag,
            objective=obj,
            forms=forms,
        )
        info["_forms"] = forms
        return info

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
        """Return the reward for the current flow state.

        The reward is the negative of the flow's objective function (which
        is to be minimized), scaled by the solver time step.

        Returns:
            float: Reward ``-dt * evaluate_objective()``.
        """
        return -self.solver.dt * self.flow.evaluate_objective()

    def check_complete(self):
        """Check whether the episode has exceeded its step budget.

        Returns:
            bool: True if the number of elapsed steps exceeds ``max_steps``.
        """
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

        t = options.get("t", 0.0) if options else 0.0
        self.iter = 0
        self.flow.reset(q0=q0, t=t)
        self.solver.reset()

        if self._has_power:
            power = self._power_info([self._power_sample()], self.flow.evaluate_objective())
            del power["_forms"]
            info.update(power)

        obs = self.flow.get_observations()
        obs = self.stack_observations(obs)

        return obs, info

    def render(self, mode="human", **kwargs):
        """Render the current PDE state.

        Args:
            mode (str, optional): Render mode passed through to the flow.
                Defaults to "human".
            **kwargs: Additional keyword arguments forwarded to
                ``flow.render``.
        """
        self.flow.render(mode=mode, **kwargs)

    def close(self):
        """Close the environment by closing all registered callbacks."""
        for cb in self.callbacks:
            cb.close()
