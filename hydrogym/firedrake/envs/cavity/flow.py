import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake.pyplot import tricontourf
from ufl import dot, ds, grad

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC


class Cavity(FlowConfig):
    DEFAULT_REYNOLDS = 7500
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-4
    DEFAULT_STABILIZATION = "gls"

    FUNCTIONS = ("q", "qB")  # This flow needs a base flow to compute fluctuation KE

    MAX_CONTROL = 0.1
    TAU = 0.075  # Time constant for controller damping (0.01*instability frequency)

    # Domain labels
    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    SLIP = 5
    WALL = (6, 8)
    CONTROL = 7
    SENSOR = 8

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    @property
    def num_inputs(self) -> int:
        return 1  # Blowing/suction on leading edge

    def configure_observations(
        self, obs_type=None, probe_obs_types={}
    ) -> ObservationFunction:
        if obs_type is None:
            obs_type = "stress_sensor"  # Shear stress at trailing edge

        supported_obs_types = {
            **probe_obs_types,
            "stress_sensor": ObservationFunction(
                self.wall_stress_sensor, num_outputs=1
            ),
        }

        if obs_type not in supported_obs_types:
            raise ValueError(f"Invalid observation type {obs_type}")

        return supported_obs_types[obs_type]

    def init_bcs(self, function_spaces=None):
        if function_spaces is None:
            V, Q = self.function_spaces(mixed=True)
        else:
            V, Q = function_spaces

        # Define static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )
        self.bcu_noslip = fd.DirichletBC(V, fd.Constant((0, 0)), self.WALL)
        # Free-slip on top boundary
        self.bcu_slip = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.SLIP)
        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        # Define time-varying boundary conditions for actuation
        # This matches Barbagallo et al (2009), "Closed-loop control of an open cavity
        # flow using reduced-order models" https://doi.org/10.1017/S0022112009991418
        u_bc = ufl.as_tensor((0.0 * self.x, -self.x * (1600 * self.x + 560) / 147))
        self.bcu_actuation = [ScaledDirichletBC(V, u_bc, self.CONTROL)]

        self.set_control(self.control_state)

    def collect_bcu(self):
        return [
            self.bcu_inflow,
            self.bcu_freestream,
            self.bcu_noslip,
            self.bcu_slip,
            *self.bcu_actuation,
        ]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def linearize_bcs(self, function_spaces=None):
        self.reset_controls()
        self.init_bcs(function_spaces=function_spaces)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))

    def wall_stress_sensor(self, q=None):
        """Integral of wall-normal shear stress (see Barbagallo et al, 2009)"""
        if q is None:
            q = self.q
        u = q.subfunctions[0]
        m = fd.assemble(-dot(grad(u[0]), self.n) * ds(self.SENSOR))
        return (m,)

    def evaluate_objective(self, q=None, qB=None):
        if q is None:
            q = self.q
        if qB is None:
            qB = self.qB
        u = q.subfunctions[0]
        uB = qB.subfunctions[0]
        KE = 0.5 * fd.assemble(fd.inner(u - uB, u - uB) * fd.dx)
        return KE

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        _fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        if clim is None:
            clim = (-10, 10)
        if levels is None:
            levels = np.linspace(*clim, 20)
        tricontourf(
            self.vorticity(),
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            cmap=cmap,
            **kwargs,
        )
        ax.set_xlim([-0.5, 2.5])
        ax.set_ylim([-1, 0.5])
        ax.set_facecolor("grey")
