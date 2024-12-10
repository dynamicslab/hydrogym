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
  rad = 0.5
  
  # # Velocity probes
  # xp = np.linspace(3, 9, 6)
  # yp = np.linspace(-1.25, 1.25, 5)
  # X, Y = np.meshgrid(xp, yp)
  # DEFAULT_VEL_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

  # Pressure probes (spaced equally around the cylinder)
  xp = np.linspace(2.0, 4.5, 5)
  yp = np.linspace(-1.25, 1.25, 7)
  X, Y = np.meshgrid(xp, yp)

  DEFAULT_PRES_PROBES = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

  # xp = np.linspace(0.51, 0.89, 3)
  # yp = np.linspace(-1.375, 1.375, 7)
  # X, Y = np.meshgrid(xp, yp)
  # DEFAULT_PRES_PROBES.extend([(x, y) for x, y in zip(X.ravel(), Y.ravel())])

  # xp = np.linspace(1.2, 1.8, 4)
  # yp = np.linspace(-0.24, 0.24, 3)
  # X, Y = np.meshgrid(xp, yp)
  # DEFAULT_PRES_PROBES.extend([(x, y) for x, y in zip(X.ravel(), Y.ravel())])

  # DEFAULT_PRES_PROBES.extend([(0.51, -1.375), (0.51, 1.375),
  #                            (0.89, -1.375), (0.89, 1.375)])

  # print('Length default pressure probes:', len(DEFAULT_PRES_PROBES), flush=True)
  # print('Default pressure probes:', DEFAULT_PRES_PROBES, flush=True)

  # DEFAULT_PRES_PROBES = [
  #   # (rad * 1.5 * 1.732 - 0.51, 1.5 * rad),
  #   # (rad * 1.5 * 1.732, 1.5 * rad + 0.51),
  #   # (rad * 1.5 * 1.732 + 0.5, 1.5 * rad+ 0.51),
  #   # (rad * 1.5 * 1.732 - 0.51, -1.5 * rad),
  #   # (rad * 1.5 * 1.732, -(1.5 * rad + 0.51)),
  #   # (rad * 1.5 * 1.732 + 0.5, -(1.5 * rad+ 0.51)),
  #   # (rad * 1.5 * 1.732 + 1.5, 1.5 * rad - 0.5),
  #   # (rad * 1.5 * 1.732 + 2.5, 1.5 * rad - 0.25),
  #   # (rad * 1.5 * 1.732 + 1.5, 1.5 * rad + 0.5),
  #   # (rad * 1.5 * 1.732 + 2.5, 1.5 * rad + 0.75),
  #   # (rad * 1.5 * 1.732 + 1.5, -(1.5 * rad - 0.5)),
  #   # (rad * 1.5 * 1.732 + 2.5, -(1.5 * rad - 0.25)),
  #   # (rad * 1.5 * 1.732 + 1.5, -(1.5 * rad + 0.5)),
  #   (rad * 1.5 * 1.732 + 2.5, -(1.5 * rad + 0.75))
  #                           ]

  # print('Length default pressure probes:', len(DEFAULT_PRES_PROBES), flush=True)
  # print('Default pressure probes:', DEFAULT_PRES_PROBES, flush=True)

  DEFAULT_VEL_PROBES = DEFAULT_PRES_PROBES
  DEFAULT_VORT_PROBES = DEFAULT_PRES_PROBES

  DEFAULT_REYNOLDS = 30
  DEFAULT_MESH = "medium"
  DEFAULT_DT = 1e-2

  FLUID = 1
  INLET = 2
  FREESTREAM = 3
  OUTLET = 4
  CYLINDER = (5, 6, 7)

  x0 = [0.0, rad * 1.5 * 1.732, rad * 1.5 * 1.732]
  y0 = [0.0, 1.5 * rad, -1.5 * rad]

  MAX_CONTROL_LOW = -1.0
  MAX_CONTROL_UP = 1.0
  CONTROL_SCALING = 7.0
  TAU = 0.05  # TODO: Tune this based on vortex shedding period
  # TAU = 0.04  # Time constant for controller damping (0.01*vortex shedding period)

  MESH_DIR = os.path.abspath(f"{__file__}/..")

  def init_bcs(self, function_spaces=None):
    if function_spaces is None:
      V, Q = self.function_spaces(mixed=True)
    else:
      V, Q = function_spaces

    # Define the static boundary conditions
    self.U_inf = fd.Constant((1.0, 0.0))
    self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
    self.bcu_freestream = fd.DirichletBC(
        V.sub(1), fd.Constant(0.0), self.FREESTREAM)  # Symmetry BCs

    self.bcp_outflow = fd.DirichletBC(Q, fd.Constant(0), self.OUTLET)

    # Define time-varying boundary conditions for the actuation
    # Set up tangential boundaries for each cylinder
    self.rad = fd.Constant(self.rad)
    self.bcu_actuation = []
    for cyl_idx in range(len(self.CYLINDER)):
      theta = atan2(
          ufl.real(self.y - self.y0[cyl_idx]),
          ufl.real(self.x - self.x0[cyl_idx]))  # Angle from center of cylinder

      # Tangential velocity
      u_bc = ufl.as_tensor((-self.rad * sin(theta), self.rad * cos(theta)))
      sub_domain = self.CYLINDER[cyl_idx]
      self.bcu_actuation.append(ScaledDirichletBC(V, u_bc, sub_domain))

    self.set_control(self.control_state)

  @property
  def num_inputs(self) -> int:
    return len(self.CYLINDER)

  def configure_observations(self,
                             obs_type=None,
                             probe_obs_types={}) -> ObservationFunction:
    if obs_type is None:
      obs_type = "lift_drag"

    def _lift_drag(q):
      CL, CD = self.compute_forces(q=q)
      return [*CL, *CD]

    supported_obs_types = {
        **probe_obs_types,
        "lift_drag":
            ObservationFunction(_lift_drag, num_outputs=6),
    }

    if obs_type not in supported_obs_types:
      raise ValueError(f"Invalid observation type {obs_type}")

    return supported_obs_types[obs_type]

  def collect_bcu(self) -> Iterable[fd.DirichletBC]:
    return [self.bcu_inflow, self.bcu_freestream, *self.bcu_actuation]

  def collect_bcp(self) -> Iterable[fd.DirichletBC]:
    return [self.bcp_outflow]

  def compute_forces(self, q: fd.Function = None) -> Iterable[float]:
    if q is None:
      q = self.q
    (u, p) = fd.split(q)
    # Lift/drag on cylinders
    force = -dot(self.sigma(u, p), self.n)
    CL = [fd.assemble(2 * force[1] * ds(cyl)) for cyl in self.CYLINDER]
    CD = [fd.assemble(2 * force[0] * ds(cyl)) for cyl in self.CYLINDER]
    return CL, CD

  def linearize_bcs(self, function_spaces=None):
    self.reset_controls()
    self.init_bcs(function_spaces=function_spaces)
    self.bcu_inflow.set_value(fd.Constant((0, 0)))
    self.bcu_freestream.set_value(fd.Constant(0.0))

  # def get_observations(self):
  #   CL, CD = self.compute_forces()
  #   return [*CL, *CD]

  def evaluate_objective(self, q: fd.Function = None, averaged_objective_values=None, return_objective_values=False) -> float:
    """The objective function for this flow is the drag coefficient"""
    if averaged_objective_values is None:
        CL, CD = self.compute_forces(q=q)
        if return_objective_values:
          return [*CL, *CD]
    else:
        CL, CD = averaged_objective_values[:3], averaged_objective_values[3:]
    # print('pinball lambda', self.reward_lambda, CD, sum(CD), flush=True)
    # return sum(np.square(CD)) + self.reward_lambda * sum(np.square(CL))
    return sum(CD) + self.reward_lambda * sum(CL)
    # return sum(CD)
    # return -(1.5 - (sum(CD) + self.reward_lambda * np.abs(sum(CL))))

  def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
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
