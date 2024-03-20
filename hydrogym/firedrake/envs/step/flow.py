import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake.petsc import PETSc
from firedrake.pyplot import tricontourf
from ufl import dot, ds, exp, grad

from hydrogym.firedrake import FlowConfig, ObservationFunction, ScaledDirichletBC


class Step(FlowConfig):
  """Backwards-facing step

    Notes on meshes:
    - "medium" - outlet at L=25 (110k elements)
    - "fine" - outlet at L=25 (223k elements)
    - "m1" - "medium" resolution with outlet at L=10 (66k elements)
    - "m2" - "medium" resolution with outlet at L=15 (81k elements)
    - "m3" - "medium" resolution with outlet at L=20 (95k elements)
    - "m4" - "fine" resolution with outlet at L=10 (130k elements)
    - "m5" - "fine" resolution with outlet at L=15 (162k elements)
    - "m6" - "fine" resolution with outlet at L=20 (193k elements)
    """

  DEFAULT_REYNOLDS = 600
  DEFAULT_MESH = "fine"
  DEFAULT_DT = 1e-2

  FUNCTIONS = ("q", "qB"
              )  # This flow needs a base flow to compute fluctuation KE

  MAX_CONTROL = 0.1  # Arbitrary... should tune this
  TAU = 0.005  # Time constant for controller damping (0.01*instability frequency)

  FLUID = 1
  INLET = 2
  OUTLET = 3
  WALL = 4
  CONTROL = 5
  SENSOR = 6

  MESH_DIR = os.path.abspath(f"{__file__}/..")

  def __init__(self, **kwargs):
    # The random forcing is implemented as low-pass-filtered white noise
    # using the DampedActuator class as a filter.  The idea is to limit the
    # dependence of the spectral characteristics of the forcing on the time
    # step of the solver.
    self.noise_amplitude = kwargs.pop("noise_amplitude", 1.0)
    self.noise_tau = kwargs.pop("noise_time_constant", 10 * self.TAU)
    self.noise_seed = kwargs.pop("noise_seed", None)
    self.noise_state = fd.Constant(0.0)
    self.rng = fd.Generator(fd.PCG64(seed=self.noise_seed))
    super().__init__(**kwargs)

  @property
  def num_inputs(self) -> int:
    return 1  # Blowing/suction on edge of step

  @property
  def nu(self):
    return fd.Constant(0.5 / ufl.real(self.Re))

  @property
  def body_force(self):
    delta = 0.1
    x0, y0 = -1.0, 0.25
    w = self.noise_state
    return w * ufl.as_tensor((
        exp(-((self.x - x0)**2 + (self.y - y0)**2) / delta**2),
        exp(-((self.x - x0)**2 + (self.y - y0)**2) / delta**2),
    ))

  def configure_observations(self,
                             obs_type=None,
                             probe_obs_types={}) -> ObservationFunction:
    if obs_type is None:
      obs_type = "stress_sensor"  # Shear stress on downstream wall

    supported_obs_types = {
        **probe_obs_types,
        "stress_sensor":
            ObservationFunction(self.wall_stress_sensor, num_outputs=1),
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
    self.U_inf = ufl.as_tensor(
        (1.0 - ((self.y - 0.25) / 0.25)**2, 0.0 * self.y))
    self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
    self.bcu_noslip = fd.DirichletBC(V, fd.Constant((0, 0)),
                                     (self.WALL, self.SENSOR))
    self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    # Define time-varying boundary conditions for actuation
    u_bc = ufl.as_tensor((0.0 * self.x, -self.x * (1600 * self.x + 560) / 147))
    self.bcu_actuation = [ScaledDirichletBC(V, u_bc, self.CONTROL)]
    self.set_control(self.control_state)

  def advance_time(self, dt, control=None):
    # Generate a noise sample
    comm = fd.COMM_WORLD
    w = np.zeros(1)
    # Generate random noise sample on rank zero
    if comm.rank == 0:
      w[0] = self.noise_amplitude * self.rng.standard_normal()

    # Send the same value to all MPI ranks
    comm.Bcast(w, root=0)

    # Update the noise filter
    x = self.noise_state
    x.assign(x + dt * (w[0] - x) / self.noise_tau)

    return super().advance_time(dt, control)

  def linearize_bcs(self, function_spaces=None):
    self.reset_controls()
    self.init_bcs(function_spaces=function_spaces)
    self.bcu_inflow.set_value(fd.Constant((0, 0)))

  def collect_bcu(self):
    return [
        self.bcu_inflow,
        self.bcu_noslip,
        *self.bcu_actuation,
    ]

  def collect_bcp(self):
    return [self.bcp_outflow]

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

  def render(
      self,
      mode="human",
      axes=None,
      clim=None,
      levels=None,
      cmap="RdBu",
      xlim=None,
      **kwargs,
  ):
    if clim is None:
      clim = (-5, 5)
    if xlim is None:
      xlim = [-2, 10]
    if levels is None:
      levels = np.linspace(*clim, 20)
    if axes is None:
      _fix, axes = plt.subplots(1, 1, figsize=(12, 2))
    tricontourf(
        self.vorticity(),
        levels=levels,
        vmin=clim[0],
        vmax=clim[1],
        extend="both",
        cmap=cmap,
        axes=axes,
        **kwargs,
    )
    axes.set_xlim(*xlim)
    axes.set_ylim([-0.5, 0.5])
    axes.set_facecolor("grey")
