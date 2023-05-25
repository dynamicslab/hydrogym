import firedrake as fd
import numpy as np
import pyadjoint

from ..core import ActuatorBase


class DampedActuator(ActuatorBase):
    def __init__(
        self,
        damping: float,
        inertia: float = 1.0,
        integration: str = "explicit",
        state: float = 0.0,
    ):
        self.k = damping
        self.m = inertia
        self._u = pyadjoint.AdjFloat(state)
        self.u = fd.Constant(state)

        assert integration in ("explicit", "implicit")
        self.integration = integration

    @property
    def value(self) -> np.ndarray:
        return self.u.values()[0]

    def set_state(self, u: float):
        self._u = pyadjoint.AdjFloat(u)
        self.u.assign(u)

    def step(self, u: float, dt: float):
        """Update the state of the actuator"""
        if self.integration == "implicit":
            self._u = (self._u + u * dt / self.m) / (1 + self.k * dt / self.m)

        elif self.integration == "explicit":
            self._u = self._u + (self.k / self.m) * dt * (u - self._u)

        self.u.assign(self._u, annotate=True)
