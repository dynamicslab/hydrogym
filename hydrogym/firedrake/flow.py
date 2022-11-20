from typing import Iterable

import firedrake as fd
import numpy as np

# import pyadjoint
import ufl
from firedrake import dx

# from firedrake import logging
from ufl import curl, inner, nabla_grad, sym

from ..core import ActuatorBase, PDEBase
from .actuator import DampedActuator


class FlowConfig(PDEBase):
    DEFAULT_REYNOLDS = 1
    MESH_DIR = ""

    ScalarType = fd.utils.ScalarType
    # ActType = pyadjoint.AdjFloat  # TODO
    ActType = fd.Constant
    # ActType = float
    ObsType = float

    def __init__(self, **config):
        super().__init__(**config)
        self.Re = fd.Constant(ufl.real(config.get("Re", self.DEFAULT_REYNOLDS)))

    def load_mesh(self, name: str) -> ufl.Mesh:
        return fd.Mesh(f"{self.MESH_DIR}/{name}.msh", name="mesh")

    def save_checkpoint(self, filename: str, write_mesh=True, idx=None):
        with fd.CheckpointFile(filename, "w") as chk:
            if write_mesh:
                chk.save_mesh(self.mesh)  # optional
            chk.save_function(self.q, idx=idx)

    def load_checkpoint(self, filename: str, idx=None, read_mesh=True):
        with fd.CheckpointFile(filename, "r") as chk:
            if read_mesh:
                self.mesh = chk.load_mesh("mesh")
                self.initialize_state()
            else:
                assert hasattr(self, "mesh")
            self.q.assign(chk.load_function(self.mesh, "q", idx=idx))
        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def initialize_state(self):
        # Set up UFL objects referring to the mesh
        self.n = fd.FacetNormal(self.mesh)
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        # Set up Taylor-Hood elements
        self.velocity_space = fd.VectorFunctionSpace(self.mesh, "CG", 2)
        self.pressure_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.mixed_space = fd.MixedFunctionSpace(
            [self.velocity_space, self.pressure_space]
        )
        self.q = fd.Function(self.mixed_space, name="q")
        self.split_solution()  # Break out and rename solution

        self.u_ctrl = [None]

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
        return DampedActuator(damping=1 / self.TAU)

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
        self.actuators = [self.create_actuator() for _ in range(self.ACT_DIM)]
        self.init_bcs(mixed=mixed)

    @property
    def nu(self):
        return fd.Constant(1 / ufl.real(self.Re))

    def split_solution(self):
        self.u, self.p = self.q.split()
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

    @property
    def body_force(self):
        return fd.interpolate(fd.Constant((0.0, 0.0)), self.velocity_space)

    def linearize_bcs(self):
        """Sets the boundary conditions appropriately for linearized flow"""
        raise NotImplementedError

    def set_control(self, act: ActType = None):
        """
        Directly sets the control state

        Note that for time-varying controls it will be better to adjust the controls
        in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
        to change control for a steady-state solve, for instance, and is also used
        internally to compute the control matrix
        """
        super().set_control(act)

        if hasattr(self, "bcu_actuation"):
            for i in range(self.ACT_DIM):
                c = fd.Constant(self.actuators[i].get_state())
                self.bcu_actuation[i]._function_arg.assign(
                    fd.interpolate(c * self.u_ctrl[i], self.velocity_space)
                )

    def control_vec(self, mixed=False):
        """Return a list of PETSc.Vecs corresponding to the columns of the control matrix"""
        (v, _) = fd.TestFunctions(self.mixed_space)
        self.linearize_bcs()
        # self.linearize_bcs() should have reset control, need to perturb it now

        fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())

        B = []
        for i in range(self.ACT_DIM):
            c = np.zeros(self.ACT_DIM)
            c[i] = 1.0  # Perturb the ith control
            self.set_control(c)

            # Control as Function
            B.append(
                fd.assemble(inner(fd.Constant((0, 0)), v) * dx, bcs=self.collect_bcs())
            )

            # Have to have mixed function space for computing B functions
            self.reset_controls(mixed=True)

        # At the end the BC function spaces could be mixed or not
        self.reset_controls(mixed=mixed)
        return B

    def dot(self, q1: fd.Function, q2: fd.Function) -> float:
        """Energy inner product between two fields"""
        u1, _ = q1.split()
        u2, _ = q2.split()
        return fd.assemble(inner(u1, u2) * dx)
