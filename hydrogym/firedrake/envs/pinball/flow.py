import os
from typing import Iterable

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
from firedrake import ds
from firedrake.pyplot import tricontourf
from ufl import atan2, cos, dot, sin

from hydrogym.firedrake import FlowConfig, ScaledDirichletBC


class Pinball(FlowConfig):
    DEFAULT_REYNOLDS = 30
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-2
    DEFAULT_VELOCITY_ORDER = 1
    DEFAULT_STABILIZATION = "gls"

    FLUID = 1
    INLET = 2
    FREESTREAM = 3
    OUTLET = 4
    CYLINDER = (5, 6, 7)

    rad = 0.5
    x0 = [0.0, rad * 1.5 * 1.866, rad * 1.5 * 1.866]
    y0 = [0.0, 1.5 * rad, -1.5 * rad]

    ACT_DIM = len(CYLINDER)
    OBS_DIM = 2 * len(CYLINDER)  # [CL, CD] for each cyliner
    MAX_CONTROL = 0.5 * np.pi
    TAU = 1.0  # TODO: Tune this based on vortex shedding period

    MESH_DIR = os.path.abspath(f"{__file__}/..")

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define the static boundary conditions
        self.U_inf = fd.Constant((1.0, 0.0))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_freestream = fd.DirichletBC(
            V.sub(1), fd.Constant(0.0), self.FREESTREAM
        )  # Symmetry BCs

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
            u_bc = ufl.as_tensor((self.rad * sin(theta), self.rad * cos(theta)))
            sub_domain = self.CYLINDER[cyl_idx]
            self.bcu_actuation.append(ScaledDirichletBC(V, u_bc, sub_domain))

        self.set_control(self.control_state)

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

    def linearize_bcs(self, mixed=True):
        self.reset_controls(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))
        self.bcu_freestream.set_value(fd.Constant(0.0))

    def get_observations(self):
        CL, CD = self.compute_forces()
        return [*CL, *CD]

    def evaluate_objective(self, q=None):
        CL, CD = self.compute_forces(q=q)
        return sum(CD)

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
