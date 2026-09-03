import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake.pyplot import tricontourf
from ufl import dot, ds, grad

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC


class Cavity(FlowConfig):
    """Open cavity flow configuration (Re 7500).

    Rectangular open cavity with inflow on the left, free-slip top, and a
    blowing/suction actuator on the leading edge whose velocity profile
    follows Barbagallo et al (2009). The default observation is the
    integral wall-normal shear stress at the trailing edge, and the
    objective is the fluctuation kinetic energy relative to a stored base
    flow ``qB`` (hence ``FUNCTIONS = ("q", "qB")``).
    """

    DEFAULT_REYNOLDS = 7500
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-4
    DEFAULT_STABILIZATION = "none"

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
        """Number of control inputs: one (blowing/suction on the leading edge)."""
        return 1  # Blowing/suction on leading edge

    def configure_observations(self, obs_type=None, probe_obs_types={}) -> ObservationFunction:
        """Select the observation function for the cavity.

        Args:
            obs_type (str, optional): Observation type. Defaults to
                "stress_sensor". Probe-based types passed in
                ``probe_obs_types`` are also supported.
            probe_obs_types (dict, optional): Probe-based observation
                functions provided by ``FlowConfig``.

        Returns:
            ObservationFunction: The selected observation function.

        Raises:
            ValueError: If ``obs_type`` is not a supported type.
        """
        if obs_type is None:
            obs_type = "stress_sensor"  # Shear stress at trailing edge

        supported_obs_types = {
            **probe_obs_types,
            "stress_sensor": ObservationFunction(self.wall_stress_sensor, num_outputs=1),
        }

        if obs_type not in supported_obs_types:
            raise ValueError(f"Invalid observation type {obs_type}")

        return supported_obs_types[obs_type]

    def init_bcs(self, function_spaces=None):
        """Construct and apply the cavity boundary conditions.

        Creates the inflow, freestream, no-slip wall, free-slip top, and
        outflow conditions, plus the time-varying leading-edge actuation
        boundary condition (stored in ``bcu_actuation`` as a
        ``ScaledDirichletBC``), then applies the current control state.

        Args:
            function_spaces (optional): Pair of (velocity, pressure)
                spaces to build conditions on; defaults to the subspaces
                of the mixed space.
        """
        if function_spaces is None:
            V, Q = self.function_spaces(mixed=True)
        else:
            V, Q = function_spaces

        # Define static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.FREESTREAM)
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
        """List of velocity boundary conditions (inflow, freestream, walls, slip, actuation).

        Returns:
            list: All velocity ``DirichletBC`` objects for this flow.
        """
        return [
            self.bcu_inflow,
            self.bcu_freestream,
            self.bcu_noslip,
            self.bcu_slip,
            *self.bcu_actuation,
        ]

    def collect_bcp(self):
        """List of pressure boundary conditions.

        Returns:
            list: Pressure ``DirichletBC`` objects (zero pressure at the outlet).
        """
        return [self.bcp_outflow]

    def linearize_bcs(self, function_spaces=None):
        """Set boundary conditions to zero-amplitude for linearized problems.

        Resets the controls to zero (which scales the actuation BC to
        zero), reinitializes the boundary conditions, and sets the inflow
        velocity to zero.

        Args:
            function_spaces (optional): Pair of (velocity, pressure)
                spaces to rebuild the conditions on.
        """
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
        """Compute the fluctuation kinetic energy relative to a base flow.

        Args:
            q (fd.Function, optional): Flow state to evaluate; defaults to
                the current state.
            qB (fd.Function, optional): Base flow to subtract; defaults to
                the stored base flow ``self.qB``.

        Returns:
            float: ``0.5 * ||u - uB||_L2^2`` of the velocity fields.
        """
        if q is None:
            q = self.q
        if qB is None:
            qB = self.qB
        u = q.subfunctions[0]
        uB = qB.subfunctions[0]
        KE = 0.5 * fd.assemble(fd.inner(u - uB, u - uB) * fd.dx)
        return KE

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        """Render the current vorticity field with matplotlib.

        Args:
            mode (str, optional): Rendering mode; only "human" plotting is
                implemented.
            clim (tuple, optional): (min, max) color limits for the
                vorticity plot. Default (-10, 10).
            levels (array, optional): Contour levels; defaults to 20
                levels spanning ``clim``.
            cmap (str, optional): Matplotlib colormap name. Default "RdBu".
            **kwargs: Additional keyword arguments passed to
                ``tricontourf``.
        """
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
