import os
from typing import Iterable

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake import ds
from firedrake.pyplot import tricontourf
from ufl import atan2, cos, dot, sin

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC


class Pinball(FlowConfig):
    """Flow over three cylinders in a triangular "pinball" arrangement (Re 30).

    Uniform inflow with symmetry conditions top and bottom, and one rotary
    (tangential) actuator per cylinder, so the number of control inputs is
    three. The default observation is the six lift/drag coefficients (one
    lift-drag pair per cylinder) and the objective is the total drag.
    """

    DEFAULT_REYNOLDS = 30
    DEFAULT_MESH = "medium"
    DEFAULT_DT = 1e-2

    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    CYLINDER = (5, 6, 7)

    rad = 0.5
    x0 = [0.0, rad * 1.5 * 1.732, rad * 1.5 * 1.732]
    y0 = [0.0, 1.5 * rad, -1.5 * rad]

    MAX_CONTROL = 10.0  # TODO: Limit this based on literature
    TAU = 0.05  # TODO: Tune this based on vortex shedding period

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    def init_bcs(self, function_spaces=None):
        """Construct and apply the pinball boundary conditions.

        Creates the inflow, freestream (symmetry), and outflow conditions,
        plus one tangential (rotary) actuation boundary condition per
        cylinder (each a ``ScaledDirichletBC`` scaled by the corresponding
        control input), then applies the current control state.

        Args:
            function_spaces (optional): Pair of (velocity, pressure)
                spaces to build conditions on; defaults to the subspaces
                of the mixed space.
        """
        if function_spaces is None:
            V, Q = self.function_spaces(mixed=True)
        else:
            V, Q = function_spaces

        # Define the static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(V.sub(1), fd.Constant(0.0), self.FREESTREAM)  # Symmetry BCs

        self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

        # Define time-varying boundary conditions for the actuation
        # Set up tangential boundaries for each cylinder
        self.rad = fd.Constant(self.rad)
        self.bcu_actuation = []
        for cyl_idx in range(len(self.CYLINDER)):
            theta = atan2(
                ufl.real(self.y - self.y0[cyl_idx]), ufl.real(self.x - self.x0[cyl_idx])
            )  # Angle from center of cylinder

            # Tangential velocity
            u_bc = ufl.as_tensor((-self.rad * sin(theta), self.rad * cos(theta)))
            sub_domain = self.CYLINDER[cyl_idx]
            self.bcu_actuation.append(ScaledDirichletBC(V, u_bc, sub_domain))

        self.set_control(self.control_state)

    @property
    def num_inputs(self) -> int:
        """Number of control inputs: one rotary actuator per cylinder (three)."""
        return len(self.CYLINDER)

    def configure_observations(self, obs_type=None, probe_obs_types={}) -> ObservationFunction:
        """Select the observation function for the pinball.

        Args:
            obs_type (str, optional): Observation type. Defaults to
                "lift_drag". Probe-based types passed in
                ``probe_obs_types`` are also supported.
            probe_obs_types (dict, optional): Probe-based observation
                functions provided by ``FlowConfig``.

        Returns:
            ObservationFunction: The selected observation function.

        Raises:
            ValueError: If ``obs_type`` is not a supported type.
        """
        if obs_type is None:
            obs_type = "lift_drag"

        def _lift_drag(q):
            """Observation payload: flattened lift coefficients followed by drag coefficients."""
            CL, CD = self.compute_forces(q=q)
            return [*CL, *CD]

        supported_obs_types = {
            **probe_obs_types,
            "lift_drag": ObservationFunction(_lift_drag, num_outputs=6),
        }

        if obs_type not in supported_obs_types:
            raise ValueError(f"Invalid observation type {obs_type}")

        return supported_obs_types[obs_type]

    def collect_bcu(self) -> Iterable[fd.DirichletBC]:
        """List of velocity boundary conditions (inflow, freestream, actuation).

        Returns:
            Iterable[fd.DirichletBC]: All velocity ``DirichletBC`` objects,
            one actuation BC per cylinder.
        """
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

    def collect_bcp(self) -> Iterable[fd.DirichletBC]:
        """List of pressure boundary conditions.

        Returns:
            Iterable[fd.DirichletBC]: Pressure ``DirichletBC`` objects
            (zero pressure at the outlet).
        """
        return [self.bcp_outflow]

    def compute_forces(self, q: fd.Function = None) -> Iterable[float]:
        """Compute dimensionless lift/drag coefficients on each cylinder.

        Args:
            q (fd.Function, optional): Flow state to compute forces from;
                defaults to the current state.

        Returns:
            Iterable[float]: Pair of lists ``(CL, CD)`` with one lift and
            one drag value per cylinder, ordered as ``CYLINDER``.
        """
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        # Lift/drag on cylinders
        force = -dot(self.sigma(u, p), self.n)
        CL = [fd.assemble(2 * force[1] * ds(cyl)) for cyl in self.CYLINDER]
        CD = [fd.assemble(2 * force[0] * ds(cyl)) for cyl in self.CYLINDER]
        return CL, CD

    def linearize_bcs(self, function_spaces=None):
        """Set boundary conditions to zero-amplitude for linearized problems.

        Resets the controls to zero (which scales the actuation BCs to
        zero), reinitializes the boundary conditions, and sets the inflow
        velocity and freestream condition to zero.

        Args:
            function_spaces (optional): Pair of (velocity, pressure)
                spaces to rebuild the conditions on.
        """
        self.reset_controls()
        self.init_bcs(function_spaces=function_spaces)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def get_observations(self):
        """Compute the current observation vector of lift/drag coefficients.

        Returns:
            list: Flattened list of lift coefficients followed by drag
            coefficients, one entry per cylinder.
        """
        CL, CD = self.compute_forces()
        return [*CL, *CD]

    def evaluate_objective(self, q=None):
        """Compute the objective: the total drag over all cylinders.

        Args:
            q (fd.Function, optional): Flow state to evaluate; defaults to
                the current state.

        Returns:
            float: Sum of the drag coefficients of all cylinders.
        """
        CL, CD = self.compute_forces(q=q)
        return sum(CD)

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        """Render the current vorticity field with the cylinder circles overlaid.

        Args:
            mode (str, optional): Rendering mode; only "human" plotting is
                implemented.
            clim (tuple, optional): (min, max) color limits for the
                vorticity plot. Default (-2, 2).
            levels (array, optional): Contour levels; defaults to 10
                levels spanning ``clim``.
            cmap (str, optional): Matplotlib colormap name. Default "RdBu".
            **kwargs: Additional keyword arguments passed to
                ``tricontourf``.

        Raises:
            AttributeError: As written this method reads the vorticity and
                cylinder geometry from a ``self.flow`` attribute, which
                ``FlowConfig`` does not define, so rendering a bare
                ``Pinball`` instance will fail.
        """
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = tricontourf(
            vort,
            cmap=cmap,
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            **kwargs,
        )

        for x0, y0 in zip(self.flow.x0, self.flow.y0):
            cyl = plt.Circle((x0, y0), self.flow.rad, edgecolor="k", facecolor="gray")
            im.axes.add_artist(cyl)
