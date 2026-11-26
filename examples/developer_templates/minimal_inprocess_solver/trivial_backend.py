"""Minimal in-process solver backend for HydroGym (reference skeleton).

This is the copyable starting point for a new solver backend whose solver
is importable as a Python library with mutable Python state (the pattern
the `firedrake` backend uses). It implements *no real physics* -- the
"flow" is a vector that grows exponentially per step -- so it runs in
plain Python with no Firedrake, no MPI, and no GPU.

The three pieces:

  - `TrivialFlow`    -- a `core.PDEBase` subclass (the state of the problem)
  - `TrivialSolver`  -- a `core.TransientSolver` subclass (how it advances)
  - `make`           -- a thin factory so `gymnasium.register` can find it

Run the accompanying tests with:

    pytest examples/developer_templates/minimal_inprocess_solver/

See docs/docs/developers/adding-a-solver.md for the full guide.
"""

from typing import Iterable

import gymnasium as gym
import numpy as np

from hydrogym.core import ActuatorBase, FlowEnv, PDEBase, TransientSolver


class TrivialActuator(ActuatorBase):
    """First-order-hold actuator: `PDEBase.reset_controls` instantiates the
    actuator list, and `advance_time` calls `step(u, dt)` on each one. The
    base-class `step` is abstract, so every backend supplies its own."""

    def step(self, u: float, dt: float):
        self._state = u


class TrivialFlow(PDEBase):
    """A fake 'PDE': an n-dimensional vector with exponential growth.

    State grows as q <- q + dt * growth_rate * q each step, and the
    objective (minimized) is ||q||^2.
    """

    MAX_CONTROL = np.inf
    DEFAULT_MESH = ""
    DEFAULT_DT = 0.01
    TAU = 0.0

    StateType = np.ndarray
    MeshType = object
    BCType = object

    def __init__(self, **config):
        # Consume your own keys with .pop() BEFORE super().__init__ so the
        # unknown-key warning only fires for genuinely unrecognized keys.
        self.n_dofs = int(config.pop("n_dofs", 8))
        self.growth_rate = float(config.pop("growth_rate", 0.1))
        super().__init__(**config)

    @property
    def num_inputs(self) -> int:
        return 1  # one actuator

    @property
    def num_outputs(self) -> int:
        return self.n_dofs  # the full state is observed

    def load_mesh(self, name: str) -> object:
        return object()  # this backend has no mesh

    def initialize_state(self):
        self.q = np.zeros(self.n_dofs)

    def init_bcs(self):
        return []  # no boundary conditions

    def reset_controls(self):
        # Override: install our concrete actuator instead of the abstract
        # ActuatorBase the parent uses.
        self.actuators = [TrivialActuator() for _ in range(self.num_inputs)]
        self.init_bcs()

    def copy_state(self, deepcopy: bool = True) -> np.ndarray:
        return np.array(self.q, copy=True)

    def save_checkpoint(self, filename: str):
        np.save(filename, self.q)

    def load_checkpoint(self, filename: str):
        self.q = np.load(filename)

    def get_observations(self) -> np.ndarray:
        return np.array(self.q, copy=True)

    def evaluate_objective(self, q: np.ndarray = None) -> float:
        state = self.q if q is None else q
        return float(np.dot(state, state))

    def render(self, **kwargs):
        pass  # nothing to draw


class TrivialSolver(TransientSolver):
    """One explicit Euler step of the fake dynamics, plus actuation."""

    def step(self, iter: int, control: Iterable[float] = None, **kwargs):
        # Apply the (smoothed) actuator state as a control perturbation
        self.flow.set_control(control)
        u = self.flow.advance_time(self.dt, self.flow.control_state)

        self.flow.q = self.flow.q + self.dt * self.flow.growth_rate * self.flow.q
        self.flow.q[0] += self.dt * u[0]  # actuator acts on the first dof
        return self.flow


def make(env_config: dict = None) -> FlowEnv:
    """Thin factory for `gymnasium.register(id=..., entry_point=...).

    Keeps the import light and gives `gym.make` a single callable.
    """
    return FlowEnv(
        env_config={
            "flow": TrivialFlow,
            "flow_config": {"n_dofs": 8},
            "solver": TrivialSolver,
            "solver_config": {"dt": 0.01},
            "max_steps": 100,
            **(env_config or {}),
        }
    )
