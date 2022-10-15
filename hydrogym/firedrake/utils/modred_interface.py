import firedrake as fd
import modred as mr
import numpy as np

from .utils import set_from_array


class Snapshot(mr.VecHandle):
    """Interface between time series in numpy binaries and modred"""

    def __init__(self, filename: str, base_vec_handle=None, scale=None):
        super().__init__(base_vec_handle=base_vec_handle, scale=scale)

        if filename[-4:] != ".npy":
            filename += ".npy"
        self.filename = filename

    def _get(self, filename=None):
        # with fd.CheckpointFile(self.filename, 'r') as chk:
        #     q = chk.load_function(self.flow.mesh, name=self.name, idx=self.idx)
        if filename is None:
            filename = self.filename
        return np.load(filename)

    def _put(self, q, filename=None):
        if filename is None:
            filename = self.filename
        np.save(filename, q)

    def as_function(self, flow):
        if fd.COMM_WORLD.size > 1:
            raise NotImplementedError

        q = fd.Function(flow.mixed_space)
        set_from_array(q, self.get())
        return q


def vec_handle_mean(vec_handles):
    """Compute the mean of the vector handles and return as SnapshotVector"""
    from functools import reduce

    data = [h.get() for h in vec_handles]
    return reduce(lambda x1, x2: x1 + x2, data) * (1 / len(data))
