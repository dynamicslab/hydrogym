from functools import partial
from typing import Callable, Iterable, NamedTuple

import firedrake as fd
import numpy as np
import pyadjoint
import ufl
from firedrake import dx, logging
from firedrake.__future__ import interpolate
from mpi4py import MPI
from numpy.typing import ArrayLike
from ufl import curl, dot, inner, nabla_grad, sqrt, sym

from ..core import ActuatorBase, PDEBase
from .actuator import DampedActuator
from .utils import print


class ScaledDirichletBC(fd.DirichletBC):
    def __init__(self, V, g, sub_domain, method=None):
        self.unscaled_function_arg = g
        self._scale = fd.Constant(1.0)
        super().__init__(V, self._scale * g, sub_domain, method)

    def set_scale(self, c):
        self._scale.assign(c)


class ObservationFunction(NamedTuple):
    func: Callable
    num_outputs: int

    def __call__(self, q: fd.Function) -> np.ndarray:
        return self.func(q)


class FlowConfig(PDEBase):
    DEFAULT_REYNOLDS = 1
    DEFAULT_VELOCITY_ORDER = 2  # Taylor-Hood elements
    DEFAULT_STABILIZATION = "none"
    MESH_DIR = ""

    FUNCTIONS = ("q",)  # tuple of functions necessary for the flow

    def __init__(self, velocity_order=None, **config):
        self.Re = fd.Constant(ufl.real(config.get("Re", self.DEFAULT_REYNOLDS)))

        if velocity_order is None:
            velocity_order = self.DEFAULT_VELOCITY_ORDER
        self.velocity_order = velocity_order

        probes = config.pop("probes", None)
        if probes is None:
            probes = []

        probe_obs_types = {
            "velocity_probes": ObservationFunction(
                partial(self.velocity_probe, probes), num_outputs=2 * len(probes)
            ),
            "pressure_probes": ObservationFunction(
                partial(self.pressure_probe, probes), num_outputs=len(probes)
            ),
            "vorticity_probes": ObservationFunction(
                partial(self.vorticity_probe, probes), num_outputs=len(probes)
            ),
        }

        self.obs_fun = self.configure_observations(
            obs_type=config.pop("observation_type", None),
            probe_obs_types=probe_obs_types,
        )

        super().__init__(**config)

    def load_mesh(self, name: str) -> ufl.Mesh:
        return fd.Mesh(f"{self.MESH_DIR}/{name}.msh", name="mesh")

    def save_checkpoint(self, filename: str, write_mesh=True, idx=None):
        with fd.CheckpointFile(filename, "w") as chk:
            if write_mesh:
                chk.save_mesh(self.mesh)  # optional
            for f_name in self.FUNCTIONS:
                chk.save_function(getattr(self, f_name), idx=idx)

            act_state = np.array([act.state for act in self.actuators])
            chk.set_attr("/", "act_state", act_state)

    def load_checkpoint(self, filename: str, idx=None, read_mesh=True):
        with fd.CheckpointFile(filename, "r") as chk:
            if read_mesh:
                self.mesh = chk.load_mesh("mesh")
                self.initialize_state()
            else:
                assert hasattr(self, "mesh")
            for f_name in self.FUNCTIONS:
                try:
                    f_load = chk.load_function(self.mesh, f_name, idx=idx)
                    f_self = getattr(self, f_name)

                    # If the checkpoint saved on a different function space,
                    # approximate the same field on the current function space
                    # by projecting the checkpoint field onto the current space
                    V_chk = f_load.function_space().ufl_element()
                    V_self = f_self.function_space().ufl_element()
                    if V_chk.ufl_element() != V_self.ufl_element():
                        f_load = fd.project(f_load, V_self)

                    f_self.assign(f_load)
                except RuntimeError:
                    logging.log(
                        logging.WARN,
                        f"Function {f_name} not found in checkpoint, defaulting to zero.",
                    )

            if chk.has_attr("/", "act_state"):
                act_state = chk.get_attr("/", "act_state")
                for i in range(self.num_inputs):
                    self.actuators[i].state = act_state[i]

        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def configure_observations(
        self, obs_type=None, probe_obs_types={}
    ) -> ObservationFunction:
        raise NotImplementedError

    def get_observations(self) -> np.ndarray:
        return self.obs_fun(self.q)

    @property
    def num_outputs(self) -> int:
        # This may be lift/drag, a stress "sensor", or a set of probe locations
        return self.obs_fun.num_outputs

    def initialize_state(self):
        # Set up UFL objects referring to the mesh
        self.n = fd.FacetNormal(self.mesh)
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(
            self.mesh, "CG", self.velocity_order
        )
        self.pressure_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.mixed_space = fd.MixedFunctionSpace(
            [self.velocity_space, self.pressure_space]
        )
        for f_name in self.FUNCTIONS:
            setattr(self, f_name, fd.Function(self.mixed_space, name=f_name))

        self.split_solution()  # Break out and rename main solution

    def set_state(self, q: fd.Function):
        """Set the current state fields

        Args:
            q (fd.Function): State to be assigned
        """
        self.q.assign(q)

    def copy_state(self, deepcopy: bool = True) -> fd.Function:
        """Return a copy of the current state fields

        Returns:
            q (fd.Function): copy of the flow state
        """
        return self.q.copy(deepcopy=deepcopy)

    def create_actuator(self, tau=None) -> ActuatorBase:
        """Create a single actuator for this flow"""
        if tau is None:
            tau = self.TAU
        return DampedActuator(1 / tau)

    def reset_controls(self):
        """Reset the controls to a zero state

        Note that this is broken out from `reset` because
        the two are not necessarily called together (e.g.
        for linearization or deriving the control vector)

        TODO: Allow for different kinds of actuators
        """
        self.actuators = [self.create_actuator() for _ in range(self.num_inputs)]
        self.init_bcs()

    @property
    def nu(self):
        return fd.Constant(1 / ufl.real(self.Re))

    def split_solution(self):
        self.u, self.p = self.q.subfunctions
        self.u.rename("u")
        self.p.rename("p")

    def vorticity(self, u: fd.Function = None) -> fd.Function:
        """Compute the vorticity field `curl(u)` of the flow

        Args:
            u (fd.Function, optional):
                If given, compute the vorticity of this velocity
                field rather than the current state.

        Returns:
            fd.Function: vorticity field
        """
        if u is None:
            u = self.u
        vort = fd.project(curl(u), self.pressure_space)
        vort.rename("vort")
        return vort

    def function_spaces(self, mixed: bool = True):
        """Function spaces for velocity and pressure

        Args:
            mixed (bool, optional):
                If True (default), return subspaces of the mixed velocity/pressure
                space. Otherwise return the segregated velocity and pressure spaces.

        Returns:
            Tuple[fd.FunctionSpace, fd.FunctionSpace]: Velocity and pressure spaces
        """
        if mixed:
            V = self.mixed_space.sub(0)
            Q = self.mixed_space.sub(1)
        else:
            V = self.velocity_space
            Q = self.pressure_space
        return V, Q

    def collect_bcu(self) -> Iterable[fd.DirichletBC]:
        """List of velocity boundary conditions"""
        return []

    def collect_bcp(self) -> Iterable[fd.DirichletBC]:
        """List of pressure boundary conditions"""
        return []

    def collect_bcs(self) -> Iterable[fd.DirichletBC]:
        """List of all boundary conditions"""
        return self.collect_bcu() + self.collect_bcp()

    def epsilon(self, u) -> ufl.Form:
        """Symmetric gradient (strain) tensor"""
        return sym(nabla_grad(u))

    def sigma(self, u, p) -> ufl.Form:
        """Newtonian stress tensor"""
        return 2 * self.nu * self.epsilon(u) - p * fd.Identity(len(u))

    @pyadjoint.no_annotations
    def max_cfl(self, dt) -> float:
        """Estimate of maximum CFL number"""
        h = fd.CellSize(self.mesh)
        CFL = fd.assemble(
            interpolate(dt * sqrt(dot(self.u, self.u)) / h, self.pressure_space)
        )
        return self.mesh.comm.allreduce(CFL.vector().max(), op=MPI.MAX)

    @property
    def body_force(self):
        return fd.Function(self.velocity_space).assign(fd.Constant((0.0, 0.0)))

    def linearize_bcs(self):
        """Sets the boundary conditions appropriately for linearized flow"""
        raise NotImplementedError

    def set_control(self, act: ArrayLike = None):
        """
        Directly sets the control state

        Note that for time-varying controls it will be better to adjust the controls
        in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
        to change control for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        if act is not None:
            super().set_control(act)

            if hasattr(self, "bcu_actuation"):
                for i in range(self.num_inputs):
                    u = np.clip(
                        self.actuators[i].state, -self.MAX_CONTROL, self.MAX_CONTROL
                    )
                    self.bcu_actuation[i].set_scale(u)

    def dot(self, q1: fd.Function, q2: fd.Function) -> float:
        """Energy inner product between two fields"""
        u1 = q1.subfunctions[0]
        u2 = q2.subfunctions[0]
        return fd.assemble(inner(u1, u2) * dx)

    def velocity_probe(self, probes, q: fd.Function = None) -> list[float]:
        """Probe velocity in the wake.

        Returns a list of velocities at the probe locations, ordered as
        (u1, u2, ..., uN, v1, v2, ..., vN) where N is the number of probes.
        """
        if q is None:
            q = self.q
        u = q.subfunctions[0]
        return np.stack(u.at(probes)).flatten("F")

    def pressure_probe(self, probes, q: fd.Function = None) -> list[float]:
        """Probe pressure around the cylinder"""
        if q is None:
            q = self.q
        p = q.subfunctions[1]
        return np.stack(p.at(probes))

    def vorticity_probe(self, probes, q: fd.Function = None) -> list[float]:
        """Probe vorticity in the wake."""
        if q is None:
            u = None
        else:
            u = q.subfunctions[0]
        vort = self.vorticity(u=u)
        return np.stack(vort.at(probes))
