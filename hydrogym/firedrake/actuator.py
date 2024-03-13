import firedrake as fd
import numpy as np
import pyadjoint
from ufl import exp

from ..core import ActuatorBase


class DampedActuator(ActuatorBase):
    """Simple damped actuator model.

    Dynamics are given by the following ODE:

    m * dx/dt = k * (u - x)

    where x is the state of the actuator, u is the control input, k is the damping
    coefficient, and m is the inertia.  Integrating over a time step `dt` with a zero-
    order hold on `u` gives the following exact solution:

    x(t + dt) = u + (x(t) - u) * exp(-k * dt / m)

    Since only the ratio k/m enters the dynamics as a time scale tau = m/k, we can
    think of the dynamics as a low-pass filter with a time constant tau.  The single
    remaining parameter is named `damping`, and corresponds to k/m = 1/tau.
    """

    def __init__(
        self,
        damping: float,
        state: float = 0.0,
    ):
        self.alpha = damping
        self._x = pyadjoint.AdjFloat(state)
        self.x = fd.Constant(state)

    @property
    def state(self) -> np.ndarray:
        return self.x.values()[0]

    @state.setter
    def state(self, u: float):
        self._x = pyadjoint.AdjFloat(u)
        self.x.assign(u)

    def step(self, u: float, dt: float):
        """Update the state of the actuator"""
        self._x = u + (self._x - u) * exp(-self.alpha * dt)
        self.x.assign(self._x, annotate=True)
