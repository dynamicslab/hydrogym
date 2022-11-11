import pyadjoint

from ..core import ActuatorBase


class DampedActuator(ActuatorBase):
    ActType = pyadjoint.AdjFloat

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
        self.u = self.ActType(0.0)
        self.implicit = implicit_integration

    def step(self, u: float, dt: float):
        """Update the state of the actuator"""
        # self.u = self.ActType(u)

        u = self.ActType(u)  # Cast to appropriate type
        if self.implicit:
            self.u = (self.u + u * dt / self.m) / (1 + self.k * dt / self.m)
        else:
            self.u += self.k * dt * (u - self.u)
