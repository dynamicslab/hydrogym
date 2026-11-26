import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import pyadjoint
import ufl
from firedrake import ds
from firedrake.pyplot import tricontourf
from ufl import as_vector, atan2, cos, dot, sign, sin, sqrt

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC

# Velocity probes
xp = np.linspace(1.0, 10.0, 16)
yp = np.linspace(-2.0, 2.0, 4)
X, Y = np.meshgrid(xp, yp)
DEFAULT_VEL_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

# Pressure probes (spaced equally around the cylinder)
RADIUS = 0.5
DEFAULT_PRES_PROBES = [
    (RADIUS * np.cos(theta), RADIUS * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False)
]


class CylinderBase(FlowConfig):
    """Base class for circular-cylinder flow configurations (Re 100).

    Uniform inflow from the left with symmetry conditions top and bottom,
    outflow on the right, and a single rotary/blowing-suction actuator on
    the cylinder wall implemented as a ``ScaledDirichletBC`` driven by
    ``cyl_velocity_field`` (subclasses define the velocity profile).
    Default observations are the lift and drag coefficients.

    Normalisation (drag and actuation power): density 1, ``U_ref = U_inf = 1``,
    ``L_ref = D = 1`` (radius 0.5), 2D per unit span:
    ``C_D = F_x / (rho U^2 D / 2)`` and ``C_P = P / (rho U^3 D / 2) = 2 P``.
    """

    U_REF = 1.0  # free-stream velocity
    L_REF = 2 * RADIUS  # cylinder diameter

    DEFAULT_REYNOLDS = 100
    DEFAULT_MESH = "medium"
    DEFAULT_DT = 1e-2

    MAX_CONTROL = 0.5 * np.pi
    TAU = 0.0556  # Time constant for controller damping (0.01*vortex shedding period)

    # Domain labels
    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    CYLINDER = 5

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    @property
    def num_inputs(self) -> int:
        """Number of control inputs: one (rotary control on the cylinder)."""
        return 1  # Rotary control

    def configure_observations(self, obs_type=None, probe_obs_types={}) -> ObservationFunction:
        """Select the observation function for the cylinder.

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

        supported_obs_types = {
            **probe_obs_types,
            "lift_drag": ObservationFunction(self.compute_forces, num_outputs=2),
        }

        if obs_type not in supported_obs_types:
            raise ValueError(f"Invalid observation type {obs_type}")

        return supported_obs_types[obs_type]

    def init_bcs(self, function_spaces=None):
        """Construct and apply the cylinder boundary conditions.

        Creates the inflow, freestream (symmetry), and outflow conditions,
        plus the time-varying actuation boundary condition on the cylinder
        wall (``ScaledDirichletBC`` with the subclass's
        ``cyl_velocity_field``), then applies the current control state.

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
        u_bc = self.cyl_velocity_field
        self.bcu_actuation = [ScaledDirichletBC(V, u_bc, self.CYLINDER)]

        # Reset the control with the current mixed (or not) function spaces
        self.set_control(self.control_state)

    @property
    def cyl_velocity_field(self):
        """Velocity vector for the actuation boundary condition on the cylinder.

        Raises:
            NotImplementedError: In the base class; subclasses must
                override this.
        """
        raise NotImplementedError

    def collect_bcu(self) -> list[fd.DirichletBC]:
        """List of velocity boundary conditions (inflow, freestream, actuation).

        Returns:
            list[fd.DirichletBC]: All velocity ``DirichletBC`` objects.
        """
        return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

    def collect_bcp(self) -> list[fd.DirichletBC]:
        """List of pressure boundary conditions.

        Returns:
            list[fd.DirichletBC]: Pressure ``DirichletBC`` objects (zero
            pressure at the outlet).
        """
        return [self.bcp_outflow]

    def compute_forces(self, q: fd.Function = None) -> tuple[float]:
        """Compute dimensionless lift/drag coefficients on cylinder

        Args:
            q (fd.Function, optional):
                Flow state to compute shear force from, if not the current state.

        Returns:
            Iterable[float]: Tuple of (lift, drag) coefficients
        """
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        # Lift/drag on cylinder
        force = -dot(self.sigma(u, p), self.n)
        CL = fd.assemble(2 * force[1] * ds(self.CYLINDER))
        CD = fd.assemble(2 * force[0] * ds(self.CYLINDER))
        return CL, CD

    # get net shear force acting tangential to the surface of the cylinder
    def shear_force(self, q: fd.Function = None) -> float:
        """Net shear force acting tangentially to the cylinder surface

        Implements the general case of the article below:
        http://www.homepages.ucl.ac.uk/~uceseug/Fluids2/Notes_Viscosity.pdf

        Args:
            q (fd.Function, optional):
                Flow state to compute shear force from, if not the current state.

        Returns:
            float: Tangential shear force
        """
        if q is None:
            q = self.q
        (u, p) = fd.split(q)
        (v, s) = fd.TestFunctions(self.mixed_space)

        # der of velocity wrt to the unit normal at the surface of the cylinder
        # equivalent to directional derivative along normal:
        du_dn = dot(self.epsilon(u), self.n)

        # Get unit tangent vector
        # pulled from https://fenics-shells.readthedocs.io/_/downloads/en/stable/pdf/
        t = as_vector((-self.n[1], self.n[0]))

        du_dn_t = (dot(du_dn, t)) * t

        # get the sign from the tangential cmpnt
        direction = sign(dot(du_dn, t))

        return fd.assemble((direction / self.Re * sqrt(du_dn_t[0] ** 2 + du_dn_t[1] ** 2)) * ds(self.CYLINDER))

    def linearize_bcs(self, function_spaces=None):
        """Set boundary conditions to zero-amplitude for linearized problems.

        Resets the controls to zero (which scales the actuation BC to
        zero) and sets the inflow velocity and freestream condition to
        zero.

        Args:
            function_spaces (optional): Pair of (velocity, pressure)
                spaces to rebuild the conditions on.
        """
        self.reset_controls(function_spaces=function_spaces)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def evaluate_objective(self, q: fd.Function = None) -> float:
        """The objective function for this flow is the drag coefficient"""
        CL, CD = self.compute_forces(q=q)
        return CD

    @pyadjoint.no_annotations
    def drag_coefficient(self, q: fd.Function = None) -> float:
        """Drag coefficient ``C_D`` of the cylinder (same normalisation as ``C_P``)."""
        CL, CD = self.compute_forces(q=q)
        return float(CD)

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        """Render the current vorticity field with the cylinder overlaid.

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
        """
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.u), self.pressure_space)
        im = tricontourf(
            vort,
            cmap=cmap,
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            **kwargs,
        )

        cyl = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
        im.axes.add_artist(cyl)


class RotaryCylinder(CylinderBase):
    """Cylinder controlled by tangential (rotary) blowing on the wall.

    The actuation boundary condition is a purely tangential velocity field
    of constant magnitude around the cylinder, so scaling it with the
    control input implements rotational forcing.

    Actuation power (one actuator): the control is the angular velocity
    ``Omega`` (wall speed ``Omega R``), and ``P = Omega M`` is the shaft
    power, ``M`` the moment the cylinder exerts on the fluid (the motor
    torque, minus the fluid torque on the cylinder) from the wall traction;
    ``C_P = 2 P`` (see ``CylinderBase``).
    """

    MAX_CONTROL = 0.5 * np.pi
    DEFAULT_DT = 1e-2

    @property
    def cyl_velocity_field(self):
        """Tangential velocity vector field around the cylinder surface.

        Returns:
            ufl.Tensor: Unit-magnitude tangential velocity
            ``(-rad * sin(theta), rad * cos(theta))`` as a function of the
            angle ``theta`` from the cylinder center.
        """
        # Set up tangential boundaries to cylinder
        theta = atan2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        self.rad = fd.Constant(RADIUS)
        # Tangential velocity
        return ufl.as_tensor((-self.rad * sin(theta), self.rad * cos(theta)))

    def actuation_power(self, q: fd.Function = None) -> np.ndarray:
        """Shaft-power coefficient ``[Omega M / (rho U^3 D / 2)]`` of the rotating cylinder."""
        (omega,) = self.actuator_amplitudes()
        return np.array([omega * self.shaft_moment(self.CYLINDER, (0.0, 0.0), q=q) / self.power_scale])


class Cylinder(CylinderBase):
    """Cylinder controlled by normal blowing/suction jets on the wall.

    Two jets centered at the top and bottom of the cylinder follow
    Rabault et al (2018), https://arxiv.org/abs/1808.07664, with a 10-degree
    angular width each; the actuation BC is scaled by the control input.

    Actuation power (two actuators: upper jet, lower jet):
    ``P_j = int_jet v_n (p - p_ref + |u|^2 / 2) dA`` with ``v_n`` the velocity
    into the fluid and ``p_ref`` the instantaneous mean wall pressure over
    both jets (the actuator group); ``C_P = 2 P`` (see ``CylinderBase``). The
    ``positive``/``abs`` forms clip the pressure and kinetic terms separately
    at every surface quadrature point (``FlowConfig.jet_power``).

    The jets are a zero-net-mass-flux pair: a positive control blows through
    the upper jet and sucks through the lower jet with the same flow rate
    (Rabault et al. 2019), so the power is complete and gauge-independent.
    ``legacy_jets=True`` reproduces HydroGym <= 1.0: both jets blow (or
    both suck) with the same flow rate (net flow ``2 Q``); the reported
    power then does not include the work ``p_ref Q`` of supplying the net flow
    at the mean jet pressure.
    """

    MAX_CONTROL = 0.1
    DEFAULT_DT = 1e-2

    def __init__(self, legacy_jets: bool = False, **config):
        """Create the jet-controlled cylinder.

        Args:
            legacy_jets: If True, both jets blow with the same sign (not
                zero-net-mass-flux), as in HydroGym <= 1.0. Default False:
                upper jet blows, lower jet sucks.
            **config: Forwarded to ``FlowConfig``.
        """
        self.legacy_jets = bool(legacy_jets)
        super().__init__(**config)

    @property
    def cyl_velocity_field(self):
        """Velocity vector for boundary condition

        Blowing/suction actuation on the cylinder wall, following Rabault, et al (2018)
        https://arxiv.org/abs/1808.07664

        The two jets act as a zero-net-mass-flux pair: a positive control blows
        through the upper jet and sucks through the lower jet with equal flow rate.
        With ``legacy_jets`` both jets blow (HydroGym <= 1.0).

        Returns:
            ufl.Tensor: Normal (radial) velocity field on the cylinder
            wall, nonzero only within the jet widths around the top and
            bottom of the cylinder.
        """

        # Set up tangential boundaries to cylinder
        theta = atan2(ufl.real(self.y), ufl.real(self.x))  # Angle from origin
        pi = ufl.pi
        self.rad = fd.Constant(RADIUS)

        omega = pi / 18  # 10 degree jet width

        theta_up = 0.5 * pi
        A_up = ufl.conditional(
            abs(theta - theta_up) < omega / 2,
            pi / (2 * omega * self.rad**2) * ufl.cos((pi / omega) * (theta - theta_up)),
            0.0,
        )

        theta_lo = -0.5 * pi
        A_lo = ufl.conditional(
            abs(theta - theta_lo) < omega / 2,
            pi / (2 * omega * self.rad**2) * ufl.cos((pi / omega) * (theta - theta_lo)),
            0.0,
        )

        if getattr(self, "legacy_jets", False):
            # HydroGym <= 1.0: both jets blow (not zero-net-mass-flux)
            return ufl.as_tensor((self.x, self.y)) * (A_up + A_lo)

        # Normal velocity at the cylinder wall (blowing on top, suction on bottom)
        return ufl.as_tensor((self.x, self.y)) * (A_up - A_lo)

    def actuation_power_forms(self, q: fd.Function = None):
        """Jet power: ``([C_P upper jet, C_P lower jet], forms)`` (see ``FlowConfig.jet_power``)."""
        jets = abs(dot(self.cyl_velocity_field, self.n)) > 0  # surfaces of both jets
        return self.jet_power(self.CYLINDER, regions=[self.y > 0, self.y < 0], support=jets, q=q)

    def actuation_power(self, q: fd.Function = None) -> np.ndarray:
        """Jet power coefficients ``[C_P upper jet, C_P lower jet]``."""
        return self.actuation_power_forms(q=q)[0]
