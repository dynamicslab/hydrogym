from typing import Iterable

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
                    getattr(self, f_name).assign(
                        chk.load_function(self.mesh, f_name, idx=idx)
                    )
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

    def create_actuator(self) -> ActuatorBase:
        """Create a single actuator for this flow"""
        return DampedActuator(1 / self.TAU)

    def reset_controls(self, mixed: bool = False):
        """Reset the controls to a zero state

        Note that this is broken out from `reset` because
        the two are not necessarily called together (e.g.
        for linearization or deriving the control vector)

        Args:
            mixed (bool, optional):
                determines a monolithic vs segregated formulation
                (see `init_bcs`). Defaults to False.

        TODO: Allow for different kinds of actuators
        """
        self.actuators = [self.create_actuator() for _ in range(self.num_inputs)]
        self.init_bcs(mixed=mixed)

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
        """_summary_

        Args:
            mixed (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_

        TODO: Is this necessary in Firedrake?
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

    def control_vec(self, mixed=False):
        """Functions corresponding to the columns of the control matrix"""
        V, Q = self.function_spaces(mixed=mixed)

        B = []
        for bcu in self.bcu_actuation:
            domain = bcu.sub_domain
            u_ctrl = bcu.unscaled_function_arg
            bc_function = fd.assemble(interpolate(u_ctrl, V))
            bcs = [fd.DirichletBC(V, bc_function, domain)]

            # Control as Function
            B.append(fd.project(fd.Constant((0, 0)), V, bcs=bcs))

        return B

    def dot(self, q1: fd.Function, q2: fd.Function) -> float:
        """Energy inner product between two fields"""
        u1 = q1.subfunctions[0]
        u2 = q2.subfunctions[0]
        return fd.assemble(inner(u1, u2) * dx)
