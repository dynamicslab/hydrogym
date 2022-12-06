import firedrake as fd
import pyadjoint

from ..core import ActuatorBase


class DampedActuator(ActuatorBase):
    # TODO: turn on implicit integration as default and
    #   test with RL, differentiability
    def __init__(
        self,
        damping: float,
        inertia: float = 1.0,
        implicit_integration: bool = False,
    ):
        self.k = damping
        self.m = inertia
        self._u = pyadjoint.AdjFloat(0.0)
        self.u = fd.Constant(0.0)
        self.implicit = implicit_integration

    def set_state(self, u: float):
        self.u.assign(u)

    def step(self, u: float, dt: float):
        """Update the state of the actuator"""
        if self.implicit:
            self._u = (self._u + u * dt / self.m) / (1 + self.k * dt / self.m)
        else:
            self._u = self._u + self.k * dt * (u - self._u)

            # Use with fd.Constant
            # self.u.assign(self.u + self.k * dt * (u - self.u), annotate=True)

        self.u.assign(self._u, annotate=True)
