import firedrake as fd
import modred as mr
from .utils import get_array, set_from_array

# Type suggestions
from ..core import FlowConfig

from modred import PODHandles

class SnapshotVector(mr.Vector):
    """Wrapper for fd.Vector objects"""
    def __init__(self, flow: FlowConfig, f: fd.Function):
        self.flow = flow
        self._function = f.copy(deepcopy=True)
        with f.dat.vec_ro as vec:
            self.data_array = vec

    def dot(self, other):
        return self.flow.dot(self.as_function(), other.as_function())

    def as_function(self):
        # Does not work in parallel
        # with self._function.dat.vec as vec:
        #     vec.setArray(self.data_array.array)

        set_from_array(self._function, self.data_array)  # Does not work in parallel

        return self._function

    def __add__(self, other):
        """Return a new object that is the sum of self and other"""
        sum_vec = self.copy()

        sum_vec.data_array = self.data_array + other.data_array  # Does not work in parallel

        # sum_vec._function += other._function
        return sum_vec

    def __mul__(self, scalar):
        """Return a new object that is the ``self * scalar`` """
        mul_vec = self.copy()

        mul_vec.data_array = self.data_array * scalar  # Does not work in parallel

        # mul_vec._function *= scalar
        return mul_vec

    def copy(self):
        f = self.as_function().copy(deepcopy=True)
        return SnapshotVector(self.flow, f)

class Snapshot(mr.VecHandle):
    """Interface between time series in CheckpointFiles and modred"""
    def __init__(self, filename: str, flow: FlowConfig, idx=None, base_vec_handle=None, scale=None, name='q'):
        super().__init__(base_vec_handle=base_vec_handle, scale=scale)
        self.flow = flow
        self.filename = filename
        self.name = name
        self.idx = idx

    def _get(self, filename=None):
        # with fd.CheckpointFile(self.filename, 'r') as chk:
        #     q = chk.load_function(self.flow.mesh, name=self.name, idx=self.idx)
        if filename is None: filename = self.filename
        self.flow.load_checkpoint(f'{filename}.h5', idx=self.idx)
        return SnapshotVector(self.flow, self.flow.q)

    def _put(self, q_vec: SnapshotVector, filename=None):
        if filename is None: filename = self.filename
        q = q_vec.as_function()
        with fd.CheckpointFile(f'{filename}.h5', 'w') as chk:
            chk.save_function(q, name=self.name, idx=self.idx)

def vec_handle_mean(vec_handles):
    """Compute the mean of the vector handles and return as SnapshotVector"""
    from functools import reduce
    do_sum = lambda x1, x2: x1 + x2
    
    data = [h.get() for h in vec_handles]
    return reduce(do_sum, data) * (1/len(data))

class ComplexSnapshot(Snapshot):
    def _get(self):
        real_part = super()._get(filneame=f'{self.filename}_real.h5')
        imag_part = super()._get(filneame=f'{self.filename}_imag.h5')
        return real_part + 1j*imag_part

    def _put(self, vec):
        set_from_array(self.flow.q)