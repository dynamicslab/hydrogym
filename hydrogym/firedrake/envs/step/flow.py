import os

import firedrake as fd
import numpy as np
import ufl
from firedrake.petsc import PETSc
from ufl import dot, ds, exp, grad

from hydrogym.firedrake import FlowConfig, ScaledDirichletBC


class Step(FlowConfig):
    DEFAULT_REYNOLDS = 600
    DEFAULT_MESH = "fine"
    DEFAULT_DT = 1e-2

    FUNCTIONS = ("q", "qB")  # This flow needs a base flow to compute fluctuation KE

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
    def nu(self):
        return fd.Constant(0.5 / ufl.real(self.Re))

    @property
    def body_force(self):
        delta = 0.1
        x0, y0 = -1.0, 0.25
        w = self.noise_state
        return w * ufl.as_tensor(
            (
                exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / delta**2),
                exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / delta**2),
            )
        )

    def init_bcs(self, mixed=False):
        V, Q = self.function_spaces(mixed=mixed)

        # Define static boundary conditions
        self.U_inf = ufl.as_tensor((1.0 - ((self.y - 0.25) / 0.25) ** 2, 0.0 * self.y))
        self.bcu_inflow = fd.DirichletBC(V, self.U_inf, self.INLET)
        self.bcu_noslip = fd.DirichletBC(
            V, fd.Constant((0, 0)), (self.WALL, self.SENSOR)
        )
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

    def linearize_bcs(self, mixed=True):
        self.reset_controls(mixed=mixed)
        self.init_bcs(mixed=mixed)
        self.bcu_inflow.set_value(fd.Constant((0, 0)))

    def collect_bcu(self):
        return [
            self.bcu_inflow,
            self.bcu_noslip,
            *self.bcu_actuation,
        ]

    def collect_bcp(self):
        return [self.bcp_outflow]

    def get_observations(self, q=None):
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
