from typing import Optional

import firedrake as fd
import numpy as np

# import pyadjoint
from firedrake import logging


class PDEModel:
    ACT_DIM = 1

    # Timescale used to smooth inputs
    #  (should be less than any meaningful timescale of the system)
    TAU = 0.0

    def __init__(self, mesh, restart=None):
        self.mesh = mesh
        self.n = fd.FacetNormal(self.mesh)
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        self.initialize_state()  # Set up function spaces and state

        # TODO: Do this without having to reinitialize everything?
        if restart is not None:
            self.load_checkpoint(restart)

        self.reset()  # Initializes control

    def initialize_state(self):
        """Set up function spaces, state vector, and any other"""
        pass

    def reset(self, q0=None):
        if q0 is not None:
            self.q.assign(q0)
        self.reset_control()

    def save_checkpoint(self, h5_file, write_mesh=True, idx=None):
        with fd.CheckpointFile(h5_file, "w") as chk:
            if write_mesh:
                chk.save_mesh(self.mesh)  # optional
            chk.save_function(self.q, idx=idx)

    def load_checkpoint(self, h5_file, idx=None, read_mesh=True):
        with fd.CheckpointFile(h5_file, "r") as chk:
            if read_mesh:
                mesh = chk.load_mesh("mesh")
                PDEModel.__init__(self, mesh)  # Reinitialize with new mesh
            else:
                assert hasattr(self, "mesh")
            self.q.assign(chk.load_function(self.mesh, "q", idx=idx))
        self.split_solution()  # Reset functions so self.u, self.p point to the new solution

    def init_bcs(self):
        """Define all boundary conditions"""
        pass

    def collect_bcs(self):
        pass

    def get_observations(self):
        pass

    def evaluate_objective(self, q=None):
        pass

    def set_control(self):
        pass

    def enlist_controls(self, control):
        if isinstance(control, int) or isinstance(control, float):
            control = [control]
        # return [pyadjoint.AdjFloat(c) for c in control]
        # return [fd.Constant(c) for c in control]
        return control

    def update_controls(self, act, dt):
        """Adds a damping factor to the controller response

        If actual control is u and input is v, effectively
            du/dt = (1/tau)*(v - u)
        """
        act = self.enlist_controls(act)
        assert len(act) == self.ACT_DIM

        # for i, (u, v) in enumerate(zip(self.control, act)):
        #     # self.control[i].assign(u + (dt/self.TAU)*(v - u))
        #     self.control[i] += (dt / self.TAU) * (v - u)
        for i, u in enumerate(act):
            self.control[i] = u
        logging.log(logging.DEBUG, self.control)
        return self.control

    def reset_control(self, mixed=False):
        self.control = self.enlist_controls(np.zeros(self.ACT_DIM))
        self.init_bcs(mixed=mixed)

    def dot(self, q1, q2):
        """Inner product between states q1 and q2"""
        pass


class CallbackBase:
    def __init__(self, interval: Optional[int] = 1):
        """
        Base class for things that happen every so often in the simulation
        (e.g. save output for Paraview or write some info to a log file).
        See also `utils/io.py`

        Parameters:
            interval - how often to take action
        """
        self.interval = interval

    def __call__(self, iter: int, t: float, flow: PDEModel):
        """
        Check if this is an 'iostep' by comparing to `self.interval`
            This assumes that a child class will do something with this information
        """
        iostep = iter % self.interval == 0
        return iostep

    def close(self):
        pass
